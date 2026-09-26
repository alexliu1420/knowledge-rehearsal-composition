# T5 Failure-matched rescue (G12) and natural-activation transfer (G14)
Files: `{qwen,falcon}_g12_s*_L*.json`, `{qwen,falcon}_natact_s*_L*.json`. Rates on failing held-out prompts whose unpatched run reproduced the stored miss.
| family | quantity | seed mean [95% CI] | per seed |
|---|---|---|---|
| Qwen2.5-3B | own bridge (explicit) | +0.857 [+0.733, +0.980] | 0.851 / 0.810 / 0.909 |
| Qwen2.5-3B | self control | +0.037 [-0.034, +0.109] | 0.041 / 0.007 / 0.064 |
| Qwen2.5-3B | wrong bridge -> donor answer | +0.564 [+0.419, +0.708] | 0.603 / 0.497 / 0.591 |
| Qwen2.5-3B | bridge-emission failures rescued | +0.893 [+0.769, +1.016] | 0.836 / 0.914 / 0.929 |
| Qwen2.5-3B | base state, subject position | +0.252 [+0.183, +0.321] | 0.227 / 0.248 / 0.282 |
| Qwen2.5-3B | route-adapter state, subject position | +0.304 [+0.136, +0.472] | 0.268 / 0.261 / 0.382 |
| Qwen2.5-3B | base state, last token | +0.063 [-0.020, +0.146] | 0.026 / 0.072 / 0.091 |
| Qwen2.5-3B | base state, wrong subject | +0.117 [+0.081, +0.154] | 0.114 / 0.133 / 0.105 |
| Falcon3-3B | own bridge (explicit) | +0.610 [+0.310, +0.911] | 0.672 / 0.471 / 0.688 |
| Falcon3-3B | self control | +0.103 [+0.071, +0.135] | 0.117 / 0.092 / 0.100 |
| Falcon3-3B | wrong bridge -> donor answer | +0.458 [+0.216, +0.699] | 0.485 / 0.350 / 0.538 |
| Falcon3-3B | bridge-emission failures rescued | +0.602 [+0.241, +0.963] | 0.756 / 0.467 / 0.582 |
| Falcon3-3B | base state, subject position | +0.363 [+0.235, +0.490] | 0.411 / 0.309 / 0.369 |
| Falcon3-3B | route-adapter state, subject position | +0.379 [+0.179, +0.579] | 0.435 / 0.287 / 0.416 |
| Falcon3-3B | base state, last token | +0.278 [+0.099, +0.457] | 0.352 / 0.274 / 0.208 |
| Falcon3-3B | base state, wrong subject | +0.204 [+0.167, +0.240] | 0.201 / 0.191 / 0.220 |
