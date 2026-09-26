# T9 Every atomic-rehearsal adapter: held-out compositional access on its comparison set, both checkpoints
Collapse is defined as held-out access below 0.7 on the comparison set; the other atomic adapters lie at 0.705-0.928, so the count depends on a threshold in a narrow gap. Files: the atomic post files of T1 (BOTH set) and T6 (rehearsed-in-all set); route adapters listed for comparison.
| family | experiment | condition | checkpoint | seed | held-out access |  | retention delta-ppl at checkpoint | mem100 epoch | loss, epoch 12 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | preservation | atomic | mem100 | 0 | 0.840 |  | -0.01 | 6 | 0.077 |
| Qwen2.5-3B | preservation | atomic | mem100 | 1 | 0.873 |  | -0.20 | 4 | 0.035 |
| Qwen2.5-3B | preservation | atomic | mem100 | 2 | 0.902 |  | -0.18 | 7 | 0.036 |
| Qwen2.5-3B | preservation | atomic | mem100 | 3 | 0.928 |  | +0.11 | 6 | 0.051 |
| Qwen2.5-3B | preservation | atomic | mem100 | 4 | 0.823 |  | +0.03 | 9 | 0.005 |
| Qwen2.5-3B | preservation | atomic | mem100 | 5 | 0.584 | collapse | +0.27 | 9 | 0.004 |
| Qwen2.5-3B | preservation | route | mem100 | 0 | 0.997 |  | -0.11 | 7 | 0.050 |
| Qwen2.5-3B | preservation | route | mem100 | 1 | 0.980 |  | +0.02 | 5 | 0.004 |
| Qwen2.5-3B | preservation | route | mem100 | 2 | 0.994 |  | -0.09 | 6 | 0.003 |
| Qwen2.5-3B | preservation | route | mem100 | 3 | 0.989 |  | +0.02 | 6 | 0.056 |
| Qwen2.5-3B | preservation | route | mem100 | 4 | 0.987 |  | +0.08 | 7 | 0.006 |
| Qwen2.5-3B | preservation | route | mem100 | 5 | 1.000 |  | +0.22 | 9 | 0.005 |
| Qwen2.5-3B | preservation | atomic | final | 0 | 0.736 |  | +0.52 | 6 | 0.077 |
| Qwen2.5-3B | preservation | atomic | final | 1 | 0.844 |  | +0.32 | 4 | 0.035 |
| Qwen2.5-3B | preservation | atomic | final | 2 | 0.924 |  | +0.07 | 7 | 0.036 |
| Qwen2.5-3B | preservation | atomic | final | 3 | 0.913 |  | +0.51 | 6 | 0.051 |
| Qwen2.5-3B | preservation | atomic | final | 4 | 0.755 |  | +0.24 | 9 | 0.005 |
| Qwen2.5-3B | preservation | atomic | final | 5 | 0.792 |  | +0.38 | 9 | 0.004 |
| Qwen2.5-3B | preservation | route | final | 0 | 0.955 |  | +0.07 | 7 | 0.050 |
| Qwen2.5-3B | preservation | route | final | 1 | 0.998 |  | +0.10 | 5 | 0.004 |
| Qwen2.5-3B | preservation | route | final | 2 | 0.995 |  | +0.14 | 6 | 0.003 |
| Qwen2.5-3B | preservation | route | final | 3 | 0.809 |  | +0.29 | 6 | 0.056 |
| Qwen2.5-3B | preservation | route | final | 4 | 0.965 |  | +0.23 | 7 | 0.006 |
| Qwen2.5-3B | preservation | route | final | 5 | 1.000 |  | +0.24 | 9 | 0.005 |
| Falcon3-3B | preservation | atomic | mem100 | 0 | 0.822 |  | +2.61 | 7 | 0.007 |
| Falcon3-3B | preservation | atomic | mem100 | 1 | 0.594 | collapse | +2.75 | 8 | 0.003 |
| Falcon3-3B | preservation | atomic | mem100 | 2 | 0.866 |  | +3.12 | 8 | 0.001 |
| Falcon3-3B | preservation | atomic | mem100 | 3 | 0.849 |  | +3.64 | 9 | 0.025 |
| Falcon3-3B | preservation | atomic | mem100 | 4 | 0.814 |  | +3.19 | 11 | 0.004 |
| Falcon3-3B | preservation | atomic | mem100 | 5 | 0.773 |  | +4.16 | 10 | 0.002 |
| Falcon3-3B | preservation | route | mem100 | 0 | 0.998 |  | +2.97 | 11 | 0.002 |
| Falcon3-3B | preservation | route | mem100 | 1 | 0.997 |  | +2.25 | 7 | 0.001 |
| Falcon3-3B | preservation | route | mem100 | 2 | 0.984 |  | +2.60 | 8 | 0.018 |
| Falcon3-3B | preservation | route | mem100 | 3 | 0.996 |  | +2.56 | 8 | 0.041 |
| Falcon3-3B | preservation | route | mem100 | 4 | 0.999 |  | +2.81 | 9 | 0.003 |
| Falcon3-3B | preservation | route | mem100 | 5 | 0.968 |  | +1.92 | 9 | 0.001 |
| Falcon3-3B | preservation | atomic | final | 0 | 0.798 |  | +3.04 | 7 | 0.007 |
| Falcon3-3B | preservation | atomic | final | 1 | 0.668 | collapse | +3.37 | 8 | 0.003 |
| Falcon3-3B | preservation | atomic | final | 2 | 0.879 |  | +4.72 | 8 | 0.001 |
| Falcon3-3B | preservation | atomic | final | 3 | 0.863 |  | +2.89 | 9 | 0.025 |
| Falcon3-3B | preservation | atomic | final | 4 | 0.808 |  | +3.10 | 11 | 0.004 |
| Falcon3-3B | preservation | atomic | final | 5 | 0.791 |  | +4.39 | 10 | 0.002 |
| Falcon3-3B | preservation | route | final | 0 | 0.999 |  | +3.24 | 11 | 0.002 |
| Falcon3-3B | preservation | route | final | 1 | 0.997 |  | +3.17 | 7 | 0.001 |
| Falcon3-3B | preservation | route | final | 2 | 0.984 |  | +3.38 | 8 | 0.018 |
| Falcon3-3B | preservation | route | final | 3 | 0.983 |  | +2.31 | 8 | 0.041 |
| Falcon3-3B | preservation | route | final | 4 | 0.963 |  | +3.50 | 9 | 0.003 |
| Falcon3-3B | preservation | route | final | 5 | 0.977 |  | +2.34 | 9 | 0.001 |
| Qwen2.5-3B | transfer | atomic | mem100 | 0 | 0.317 | collapse | +0.16 | 7 | 0.010 |
| Qwen2.5-3B | transfer | atomic | mem100 | 1 | 0.852 |  | -0.22 | 6 | 0.056 |
| Qwen2.5-3B | transfer | atomic | mem100 | 2 | 0.775 |  | +0.02 | 11 | 0.008 |
| Qwen2.5-3B | transfer | route | mem100 | 0 | 0.992 |  | +0.11 | 7 | 0.005 |
| Qwen2.5-3B | transfer | route | mem100 | 1 | 0.969 |  | +0.28 | 6 | 0.006 |
| Qwen2.5-3B | transfer | route | mem100 | 2 | 0.993 |  | -0.12 | 7 | 0.004 |
| Qwen2.5-3B | transfer | atomic | final | 0 | 0.705 |  | +0.25 | 7 | 0.010 |
| Qwen2.5-3B | transfer | atomic | final | 1 | 0.382 | collapse | -0.01 | 6 | 0.056 |
| Qwen2.5-3B | transfer | atomic | final | 2 | 0.793 |  | +0.10 | 11 | 0.008 |
| Qwen2.5-3B | transfer | route | final | 0 | 0.995 |  | +0.30 | 7 | 0.005 |
| Qwen2.5-3B | transfer | route | final | 1 | 0.999 |  | +0.25 | 6 | 0.006 |
| Qwen2.5-3B | transfer | route | final | 2 | 0.996 |  | +0.02 | 7 | 0.004 |
| Falcon3-3B | transfer | atomic | mem100 | 0 | 0.874 |  | +3.54 | 7 | 0.086 |
| Falcon3-3B | transfer | atomic | mem100 | 1 | 0.783 |  | +3.78 | 8 | 0.013 |
| Falcon3-3B | transfer | atomic | mem100 | 2 | 0.892 |  | +4.01 | 9 | 0.007 |
| Falcon3-3B | transfer | route | mem100 | 0 | 0.993 |  | +2.89 | 8 | 0.013 |
| Falcon3-3B | transfer | route | mem100 | 1 | 0.982 |  | +2.71 | 9 | 0.001 |
| Falcon3-3B | transfer | route | mem100 | 2 | 0.941 |  | +3.23 | 10 | 0.003 |
| Falcon3-3B | transfer | atomic | final | 0 | 0.850 |  | +3.51 | 7 | 0.086 |
| Falcon3-3B | transfer | atomic | final | 1 | 0.669 | collapse | +3.72 | 8 | 0.013 |
| Falcon3-3B | transfer | atomic | final | 2 | 0.910 |  | +3.76 | 9 | 0.007 |
| Falcon3-3B | transfer | route | final | 0 | 0.986 |  | +3.17 | 8 | 0.013 |
| Falcon3-3B | transfer | route | final | 1 | 0.986 |  | +3.06 | 9 | 0.001 |
| Falcon3-3B | transfer | route | final | 2 | 0.961 |  | +3.05 | 10 | 0.003 |

Atomic adapters below 0.7: mem100 3 of 18, final 3 of 18. Route adapters below 0.7: mem100 0 of 18, final 0 of 18.
