# T11 Sensitivity: primary contrast restricted by atomic pairing and by baseline second-hop knowledge (exploratory, not declared)
The route builder does not require the second hop to be known when asked directly. `hop2_known` = fraction of six direct second-hop phrasings the base model answers (files `v2_g4_hop2_base6.json`, `falcon_hop2_base6.json`). Token matching trims atomic rehearsal rows, not routes, so some routes have only one of their two facts rehearsed (`preserve_atomic.json`, field `atom`). route - atomic at mem100 on the BOTH set, six seeds, t(5).
| family | subset | n routes | route - atomic, 6 seeds [95% CI] |
|---|---|---|---|
| Qwen2.5-3B | all comparison routes | 229 | +0.166 [+0.032, +0.301] |
| Qwen2.5-3B | both facts rehearsed under atomic | 208 | +0.171 [+0.036, +0.307] |
| Qwen2.5-3B | both facts rehearsed and second hop known on all six | 162 | +0.153 [+0.010, +0.296] |
| Qwen2.5-3B | second hop known on all six phrasings | 179 | +0.148 [+0.007, +0.289] |
| Qwen2.5-3B | second hop known on at least one phrasing | 208 | +0.153 [+0.019, +0.287] |
| Qwen2.5-3B | second hop known on none | 21 | +0.299 [+0.151, +0.448] |
| Falcon3-3B | all comparison routes | 414 | +0.204 [+0.097, +0.311] |
| Falcon3-3B | both facts rehearsed under atomic | 414 | +0.204 [+0.097, +0.311] |
| Falcon3-3B | both facts rehearsed and second hop known on all six | 231 | +0.175 [+0.047, +0.302] |
| Falcon3-3B | second hop known on all six phrasings | 231 | +0.175 [+0.047, +0.302] |
| Falcon3-3B | second hop known on at least one phrasing | 297 | +0.183 [+0.067, +0.299] |
| Falcon3-3B | second hop known on none | 117 | +0.257 [+0.151, +0.363] |
