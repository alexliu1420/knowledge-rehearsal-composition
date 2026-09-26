# T4 Exposure accounting per epoch (injected rows identical across conditions)
Rows and token counts are for the rehearsal rows only; content tokens are matched by design; processed = chat-templated prompt + answer; supervised = answer tokens (answer-only loss). Steps/epoch counts ALL training rows, injection plus rehearsal (Qwen 390 injection rows, Falcon 831): floor(rows/8), because the trainer accumulates 8 rows per optimizer step and discards the incomplete last group of each epoch.
| family | condition | rows | content tok | processed tok | supervised tok | optimizer steps/epoch (injection + rehearsal) | mean mem100 epoch | steps to mem100 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | route | 461 | 8740 | 22654 | 1520 | 106 | 6.7 (6 seeds) | 707 |
| Qwen2.5-3B | atomic | 637 | 8738 | 28048 | 2644 | 128 | 6.8 (6 seeds) | 875 |
| Falcon3-3B | route | 819 | 14926 | 42807 | 1878 | 206 | 8.7 (6 seeds) | 1785 |
| Falcon3-3B | atomic | 1112 | 14929 | 52937 | 4025 | 242 | 8.8 (6 seeds) | 2138 |
