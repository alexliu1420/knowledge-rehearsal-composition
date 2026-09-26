# T7 Shortcut test: bridge-dependence on never-rehearsed (L) and rehearsed (Rall) routes, Qwen L12, seed-0 mem100 adapters
Files: `bridge_causal_L12.json` (base), `shortcut_{none,route,bridgectx,atomic}_s0_L12.json`. identity = acc(own bridge) - acc(wrong bridge) on routes answered unpatched; strict follow = wrong-bridge output equals the actual donor's answer.
| condition | set | n answered | answered unpatched | own bridge | wrong bridge | identity | identity - base | strict follow |
|---|---|---|---|---|---|---|---|---|
| base | L | 320 | 1.000 | 0.825 | 0.322 | +0.503 | +0.000 | 0.439 |
| base | Rall | 228 | 1.000 | 0.899 | 0.333 | +0.566 | +0.000 | 0.370 |
| none | L | 244 | 0.762 | 0.852 | 0.307 | +0.545 | +0.042 | 0.498 |
| none | Rall | 151 | 0.662 | 0.894 | 0.258 | +0.636 | +0.070 | 0.497 |
| route | L | 310 | 0.969 | 0.910 | 0.313 | +0.597 | +0.094 | 0.634 |
| route | Rall | 228 | 1.000 | 0.934 | 0.342 | +0.592 | +0.026 | 0.595 |
| bridgectx | L | 311 | 0.972 | 0.932 | 0.251 | +0.682 | +0.179 | 0.671 |
| bridgectx | Rall | 226 | 0.991 | 0.978 | 0.261 | +0.717 | +0.151 | 0.680 |
| atomic | L | 274 | 0.856 | 0.872 | 0.270 | +0.602 | +0.099 | 0.608 |
| atomic | Rall | 182 | 0.798 | 0.984 | 0.264 | +0.720 | +0.154 | 0.547 |
