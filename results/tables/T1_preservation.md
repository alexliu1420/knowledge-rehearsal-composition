# T1 Preservation: route - atomic rehearsal, BOTH-replayed set
Files: `{pre}preserve_{cond}_{ck}_post_X_s{seed}.json`. Held-out access = (5 non-canonical phrasings, biomedical persona + 6 phrasings, neutral persona) / 11. Seed-level paired t. mem100 = first epoch at which a 40-item probe of the injected facts reaches >= 0.99 strict recall; full-set recall over all injected routes at that checkpoint is reported in the last column.
| family | checkpoint | seeds | n routes | route - atomic [95% CI] | per seed | full-set injected recall route / atomic |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | mem100 | 3 seeds | 229 | +0.119 [+0.035, +0.203] | +0.157 / +0.107 / +0.092 | 0.973 / 0.966 (min 0.967 / 0.941) |
| Qwen2.5-3B | mem100 | 6 seeds | 229 | +0.166 [+0.032, +0.301] | +0.157 / +0.107 / +0.092 / +0.062 / +0.164 / +0.416 | 0.979 / 0.972 (min 0.967 / 0.941) |
| Qwen2.5-3B | final | 3 seeds | 229 | +0.148 [-0.036, +0.331] | +0.218 / +0.154 / +0.071 | 0.979 / 0.935 (min 0.956 / 0.879) |
| Qwen2.5-3B | final | 6 seeds | 229 | +0.126 [-0.006, +0.258] | +0.218 / +0.154 / +0.071 / -0.104 / +0.210 / +0.208 | 0.975 / 0.957 (min 0.941 / 0.879) |
| Falcon3-3B | mem100 | 3 seeds | 414 | +0.232 [-0.143, +0.607] | +0.176 / +0.403 / +0.118 | 0.974 / 0.963 (min 0.951 / 0.930) |
| Falcon3-3B | mem100 | 6 seeds | 414 | +0.204 [+0.097, +0.311] | +0.176 / +0.403 / +0.118 / +0.147 / +0.184 / +0.195 | 0.976 / 0.978 (min 0.951 / 0.930) |
| Falcon3-3B | final | 3 seeds | 414 | +0.212 [-0.068, +0.491] | +0.201 / +0.329 / +0.105 | 0.994 / 0.993 (min 0.989 / 0.983) |
| Falcon3-3B | final | 6 seeds | 414 | +0.183 [+0.098, +0.267] | +0.201 / +0.329 / +0.105 / +0.120 / +0.155 / +0.186 | 0.990 / 0.971 (min 0.974 / 0.859) |

## Scoring-rule robustness (mem100, seeds 0-2)
| family | scorer | route - atomic [95% CI] |
|---|---|---|
| Qwen2.5-3B | study | +0.119 [+0.035, +0.203] |
| Qwen2.5-3B | substring | +0.119 [+0.035, +0.203] |
| Qwen2.5-3B | word-boundary | +0.119 [+0.035, +0.203] |
| Qwen2.5-3B | exact | +0.123 [+0.034, +0.211] |
| Falcon3-3B | study | +0.232 [-0.143, +0.607] |
| Falcon3-3B | substring | +0.232 [-0.143, +0.607] |
| Falcon3-3B | word-boundary | +0.232 [-0.143, +0.607] |
| Falcon3-3B | exact | +0.233 [-0.145, +0.611] |
