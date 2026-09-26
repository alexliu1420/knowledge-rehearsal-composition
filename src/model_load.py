"""Loading a model at a pinned revision, optionally in 4-bit.

Study 4's stage-0 screen is inference only, which is what makes a scale comparison possible
on one 8 GB card at all. Qwen2.5-3B fits in fp16 (~6.2 GB); 7B does not, and needs NF4.

The 3B point is therefore the **uncontaminated** scale comparison and the 7B point is the
extended one. Any result that appears only in 4-bit is reported as such, because quantisation
is a change to the model and not a neutral convenience -- this programme measures small
differences in factual access, which is exactly where quantisation error lands.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")


def load_model(model_name: str, precision: str = "fp16", load_4bit: bool = False):
    """The pinned model on cuda, in eval mode."""
    import torch
    from transformers import AutoModelForCausalLM

    from model_pin import revision_for

    rev = revision_for(model_name)
    if load_4bit:
        from transformers import BitsAndBytesConfig

        # NF4 with double quantisation and an fp16 compute dtype: the configuration the
        # QLoRA paper reports as closest to the unquantised model. compute_dtype matches the
        # fp16 used everywhere else in the programme so the arithmetic path is unchanged.
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        m = AutoModelForCausalLM.from_pretrained(
            model_name, revision=rev, quantization_config=qc, device_map={"": 0})
        return m.eval()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[precision]
    m = AutoModelForCausalLM.from_pretrained(model_name, revision=rev, dtype=dtype)
    return m.to("cuda").eval()


def load_tag(model_name: str, precision: str, load_4bit: bool) -> str:
    """Short label for cache keys and reports, so a 4-bit run cannot be confused for fp16."""
    return f"{model_name.split('/')[-1]}.{'nf4' if load_4bit else precision}"
