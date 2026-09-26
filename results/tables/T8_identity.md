# T8 Causal screens on successful computations: bridge-representation identity by layer (supplementary S1)
Files: Qwen `bridge_causal_L{6,12,24}.json` (base) and `mech_{route,atomic}_s0_L{6,12,24}.json`; Falcon `falcon_causal_{base,route_mem100,atomic_mem100}[_s{1,2}]_L{7,11,15}.json`. identity = acc(own-bridge transplant) - acc(wrong-bridge transplant) at the subject position, on comparison-set routes the adapter answers unpatched. Falcon: three seeds, t(2); Qwen: seed 0 only.

## Qwen2.5-3B
| layer | base | route rehearsal (per seed) | atomic rehearsal (per seed) | route - atomic |
|---|---|---|---|---|
| 6 | +0.449 | +0.529 | +0.576 | -0.047 (1 seed) |
| 12 | +0.498 | +0.599 | +0.668 | -0.069 (1 seed) |
| 24 | +0.626 | +0.542 | +0.659 | -0.117 (1 seed) |

## Falcon3-3B
| layer | base | route rehearsal (per seed) | atomic rehearsal (per seed) | route - atomic |
|---|---|---|---|---|
| 7 | +0.393 | +0.370 / +0.345 / +0.292 | +0.497 / +0.521 / +0.516 | -0.176 [-0.296, -0.056] |
| 11 | +0.396 | +0.355 / +0.291 / +0.265 | +0.492 / +0.508 / +0.547 | -0.212 [-0.393, -0.031] |
| 15 | +0.063 | +0.044 / +0.041 / +0.056 | +0.161 / +0.089 / +0.153 | -0.087 [-0.176, +0.001] |
