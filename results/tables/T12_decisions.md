# T12 Decision rules of the format ablations (G16 bridge-token mask, G17 answer-only, G18 seeds 3-5)
Rehearsed routes (Rall) unless stated; acquisition-matched checkpoint unless stated. f = (answer-only - masked) / (bridge-as-context - masked). Per-family intervals pair by seed (t(k-1)). Seeds 0-2 are the declared G17 cohort; seeds 3-5 were added after it was seen (G18, post hoc) and are shown alone and combined. The pooled row pairs seeds across both families and was not declared (exploratory).
| family | checkpoint | set | f (seeds 0-2) | seed pairs | bridge-as-context - answer-only [95% CI] | sign |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | mem100 | Rall | -0.21 | 3, seeds 0-2 (declared, G17) | +0.105 [-0.027, +0.236] | 3/3 |
| Qwen2.5-3B | mem100 | Rall | -0.21 | 3, seeds 3-5 alone (G18) | +0.093 [-0.056, +0.242] | 3/3 |
| Qwen2.5-3B | mem100 | Rall | -0.21 | 6, seeds 0-5 (post hoc extension, G18) | +0.099 [+0.045, +0.152] | 6/6 |
| Qwen2.5-3B | mem100 | L | +0.16 | 3, seeds 0-2 (declared, G17) | +0.057 [-0.082, +0.196] | 3/3 |
| Qwen2.5-3B | mem100 | L | +0.16 | 3, seeds 3-5 alone (G18) | +0.042 [-0.078, +0.162] | 3/3 |
| Qwen2.5-3B | mem100 | L | +0.16 | 6, seeds 0-5 (post hoc extension, G18) | +0.049 [-0.000, +0.099] | 6/6 |
| Qwen2.5-3B | final | Rall | -1.21 | 3, seeds 0-2 (declared, G17) | +0.042 [-0.114, +0.199] | 2/3 |
| Qwen2.5-3B | final | Rall | -1.21 | 3, seeds 3-5 alone (G18) | +0.123 [+0.014, +0.231] | 3/3 |
| Qwen2.5-3B | final | Rall | -1.21 | 6, seeds 0-5 (post hoc extension, G18) | +0.082 [+0.014, +0.151] | 5/6 |
| Qwen2.5-3B | final | L | +0.77 | 3, seeds 0-2 (declared, G17) | -0.002 [-0.205, +0.201] | 2/3 |
| Qwen2.5-3B | final | L | +0.77 | 3, seeds 3-5 alone (G18) | +0.077 [-0.030, +0.184] | 3/3 |
| Qwen2.5-3B | final | L | +0.77 | 6, seeds 0-5 (post hoc extension, G18) | +0.037 [-0.039, +0.114] | 5/6 |
| Falcon3-3B | mem100 | Rall | +0.26 | 3, seeds 0-2 (declared, G17) | +0.117 [-0.039, +0.272] | 3/3 |
| Falcon3-3B | mem100 | Rall | +0.26 | 3, seeds 3-5 alone (G18) | +0.066 [-0.125, +0.258] | 2/3 |
| Falcon3-3B | mem100 | Rall | +0.26 | 6, seeds 0-5 (post hoc extension, G18) | +0.091 [+0.019, +0.163] | 5/6 |
| Falcon3-3B | mem100 | L | +0.23 | 3, seeds 0-2 (declared, G17) | +0.107 [-0.106, +0.319] | 3/3 |
| Falcon3-3B | mem100 | L | +0.23 | 3, seeds 3-5 alone (G18) | +0.072 [-0.143, +0.287] | 3/3 |
| Falcon3-3B | mem100 | L | +0.23 | 6, seeds 0-5 (post hoc extension, G18) | +0.089 [+0.006, +0.172] | 6/6 |
| Falcon3-3B | final | Rall | +0.67 | 3, seeds 0-2 (declared, G17) | +0.077 [-0.027, +0.181] | 3/3 |
| Falcon3-3B | final | Rall | +0.67 | 3, seeds 3-5 alone (G18) | +0.139 [-0.101, +0.379] | 3/3 |
| Falcon3-3B | final | Rall | +0.67 | 6, seeds 0-5 (post hoc extension, G18) | +0.108 [+0.029, +0.186] | 6/6 |
| Falcon3-3B | final | L | +0.73 | 3, seeds 0-2 (declared, G17) | +0.061 [-0.094, +0.215] | 3/3 |
| Falcon3-3B | final | L | +0.73 | 3, seeds 3-5 alone (G18) | +0.140 [-0.051, +0.331] | 3/3 |
| Falcon3-3B | final | L | +0.73 | 6, seeds 0-5 (post hoc extension, G18) | +0.100 [+0.020, +0.180] | 6/6 |

## Pooled across families (exploratory)
| checkpoint | set | seed pairs | bridge-as-context - answer-only [95% CI] | sign |
|---|---|---|---|---|
| mem100 | Rall | 6 | +0.111 [+0.056, +0.166] | 6/6 |
| mem100 | L | 6 | +0.082 [+0.008, +0.155] | 6/6 |
| final | Rall | 6 | +0.060 [+0.006, +0.114] | 5/6 |
| final | L | 6 | +0.029 [-0.048, +0.106] | 5/6 |
