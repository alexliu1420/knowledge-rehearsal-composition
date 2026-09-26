# T6 Transfer experiment: rehearse half the routes, measure the entity-disjoint other half; 3 seeds per family
Files: `results/g15_transfer.json` (Qwen), `results/g15_transfer_falcon.json` (Falcon), from `analyze_transfer.py`. L = never rehearsed in any condition; Rall = rehearsed in every condition. Seed-level paired t(2); sign = seeds with the contrast > 0.

## Qwen2.5-3B (340 rehearsed / 320 never-rehearsed), mem100
| set | condition | held-out | per seed | canonical | bridge emission | per seed |
|---|---|---|---|---|---|---|
| L | none | 0.819 | 0.841 / 0.829 / 0.787 | 0.713 | 0.007 | 0.003 / 0.009 / 0.008 |
| L | route | 0.948 | 0.947 / 0.938 / 0.959 | 0.965 | 0.004 | 0.004 / 0.004 / 0.002 |
| L | atomic | 0.743 | 0.477 / 0.878 / 0.873 | 0.929 | 0.026 | 0.052 / 0.008 / 0.018 |
| L | coherent | 0.858 | 0.835 / 0.884 / 0.856 | 0.922 | 0.006 | 0.009 / 0.002 / 0.006 |
| L | bridgectx | 0.947 | 0.958 / 0.934 / 0.949 | 0.975 | 0.003 | 0.004 / 0.001 / 0.004 |
| Rall | none | 0.762 | 0.765 / 0.786 / 0.736 | 0.611 | 0.011 | 0.002 / 0.012 / 0.021 |
| Rall | route | 0.985 | 0.992 / 0.969 / 0.993 | 0.994 | 0.003 | 0.006 / 0.002 / 0.000 |
| Rall | atomic | 0.648 | 0.317 / 0.852 / 0.775 | 0.911 | 0.287 | 0.590 / 0.097 / 0.175 |
| Rall | coherent | 0.831 | 0.783 / 0.873 / 0.838 | 0.908 | 0.049 | 0.092 / 0.006 / 0.049 |
| Rall | bridgectx | 0.970 | 0.980 / 0.955 / 0.977 | 0.989 | 0.005 | 0.003 / 0.008 / 0.005 |
| set | contrast | mean [95% CI] | sign |
|---|---|---|---|
| L | route - atomic | +0.206 [-0.365, +0.776] | 3/3 |
| L | route - none | +0.129 [+0.036, +0.222] | 3/3 |
| L | bridgectx - none | +0.128 [+0.052, +0.204] | 3/3 |
| L | coherent - none | +0.039 [-0.060, +0.138] | 2/3 |
| L | atomic - none | -0.076 [-0.696, +0.543] | 2/3 |
| L | route - bridgectx | +0.001 [-0.025, +0.027] | 2/3 |
| L | route - coherent | +0.090 [+0.012, +0.169] | 3/3 |
| L | bridgectx - coherent | +0.089 [-0.003, +0.181] | 3/3 |
| Rall | bridgectx - coherent | +0.139 [-0.005, +0.283] | 3/3 |
| Rall | route - atomic | +0.337 [-0.402, +1.075] | 3/3 |
| Rall | bridgectx - atomic | +0.323 [-0.420, +1.065] | 3/3 |
| Rall | coherent - atomic | +0.184 [-0.426, +0.793] | 3/3 |
| Rall | emission: atomic - none | +0.276 [-0.401, +0.953] | 3/3 |
| Rall | emission: coherent - none | +0.038 [-0.084, +0.159] | 2/3 |
| Rall | emission: bridgectx - none | -0.006 [-0.027, +0.014] | 1/3 |
| Rall | emission: bridgectx - atomic | -0.282 [-0.946, +0.381] | 0/3 |

## Qwen2.5-3B (340 rehearsed / 320 never-rehearsed), final
| set | condition | held-out | per seed | canonical | bridge emission | per seed |
|---|---|---|---|---|---|---|
| L | none | 0.844 | 0.860 / 0.829 / 0.841 | 0.766 | 0.005 | 0.002 / 0.009 / 0.004 |
| L | route | 0.962 | 0.956 / 0.963 / 0.966 | 0.973 | 0.002 | 0.005 / 0.001 / 0.001 |
| L | atomic | 0.701 | 0.840 / 0.393 / 0.868 | 0.951 | 0.026 | 0.018 / 0.049 / 0.009 |
| L | coherent | 0.847 | 0.814 / 0.836 / 0.891 | 0.920 | 0.007 | 0.006 / 0.008 / 0.008 |
| L | bridgectx | 0.916 | 0.942 / 0.839 / 0.968 | 0.944 | 0.003 | 0.003 / 0.004 / 0.001 |
| Rall | none | 0.793 | 0.800 / 0.786 / 0.792 | 0.652 | 0.007 | 0.001 / 0.012 / 0.009 |
| Rall | route | 0.997 | 0.995 / 0.999 / 0.996 | 1.000 | 0.002 | 0.006 / 0.000 / 0.000 |
| Rall | atomic | 0.627 | 0.705 / 0.382 / 0.793 | 0.930 | 0.286 | 0.259 / 0.441 / 0.159 |
| Rall | coherent | 0.848 | 0.832 / 0.845 / 0.865 | 0.931 | 0.036 | 0.035 / 0.034 / 0.037 |
| Rall | bridgectx | 0.945 | 0.959 / 0.885 / 0.992 | 0.978 | 0.004 | 0.000 / 0.012 / 0.000 |
| set | contrast | mean [95% CI] | sign |
|---|---|---|---|
| L | route - atomic | +0.261 [-0.403, +0.926] | 3/3 |
| L | route - none | +0.118 [+0.069, +0.168] | 3/3 |
| L | bridgectx - none | +0.073 [-0.074, +0.220] | 3/3 |
| L | coherent - none | +0.004 [-0.116, +0.123] | 2/3 |
| L | atomic - none | -0.143 [-0.776, +0.490] | 1/3 |
| L | route - bridgectx | +0.045 [-0.125, +0.216] | 2/3 |
| L | route - coherent | +0.115 [+0.027, +0.202] | 3/3 |
| L | bridgectx - coherent | +0.069 [-0.088, +0.226] | 3/3 |
| Rall | bridgectx - coherent | +0.098 [-0.027, +0.223] | 3/3 |
| Rall | route - atomic | +0.370 [-0.173, +0.913] | 3/3 |
| Rall | bridgectx - atomic | +0.319 [-0.084, +0.721] | 3/3 |
| Rall | coherent - atomic | +0.221 [-0.305, +0.747] | 3/3 |
| Rall | emission: atomic - none | +0.279 [-0.072, +0.630] | 3/3 |
| Rall | emission: coherent - none | +0.028 [+0.013, +0.043] | 3/3 |
| Rall | emission: bridgectx - none | -0.003 [-0.016, +0.010] | 0/3 |
| Rall | emission: bridgectx - atomic | -0.282 [-0.622, +0.058] | 0/3 |

## Falcon3-3B (571 rehearsed / 541 never-rehearsed), mem100
| set | condition | held-out | per seed | canonical | bridge emission | per seed |
|---|---|---|---|---|---|---|
| L | none | 0.654 | 0.592 / 0.675 / 0.697 | 0.681 | 0.000 | 0.000 / 0.000 / 0.000 |
| L | route | 0.923 | 0.945 / 0.926 / 0.899 | 0.961 | 0.000 | 0.000 / 0.000 / 0.000 |
| L | atomic | 0.893 | 0.917 / 0.845 / 0.916 | 0.938 | 0.000 | 0.000 / 0.000 / 0.000 |
| L | bridgectx | 0.905 | 0.901 / 0.920 / 0.894 | 0.953 | 0.000 | 0.000 / 0.000 / 0.000 |
| Rall | none | 0.681 | 0.644 / 0.692 / 0.707 | 0.700 | 0.002 | 0.002 / 0.004 / 0.001 |
| Rall | route | 0.972 | 0.993 / 0.982 / 0.941 | 0.997 | 0.000 | 0.001 / 0.001 / 0.000 |
| Rall | atomic | 0.850 | 0.874 / 0.783 / 0.892 | 0.920 | 0.048 | 0.036 / 0.103 / 0.005 |
| Rall | bridgectx | 0.948 | 0.946 / 0.959 / 0.938 | 0.980 | 0.000 | 0.001 / 0.000 / 0.001 |
| set | contrast | mean [95% CI] | sign |
|---|---|---|---|
| L | route - atomic | +0.031 [-0.091, +0.152] | 2/3 |
| L | route - none | +0.269 [+0.078, +0.460] | 3/3 |
| L | bridgectx - none | +0.251 [+0.111, +0.390] | 3/3 |
| L | atomic - none | +0.238 [+0.041, +0.436] | 3/3 |
| L | route - bridgectx | +0.018 [-0.036, +0.073] | 3/3 |
| Rall | route - atomic | +0.122 [-0.063, +0.308] | 3/3 |
| Rall | bridgectx - atomic | +0.098 [-0.073, +0.269] | 3/3 |
| Rall | emission: atomic - none | +0.046 [-0.075, +0.166] | 3/3 |
| Rall | emission: bridgectx - none | -0.002 [-0.007, +0.003] | 0/3 |
| Rall | emission: bridgectx - atomic | -0.048 [-0.173, +0.078] | 0/3 |

## Falcon3-3B (571 rehearsed / 541 never-rehearsed), final
| set | condition | held-out | per seed | canonical | bridge emission | per seed |
|---|---|---|---|---|---|---|
| L | none | 0.684 | 0.681 / 0.674 / 0.698 | 0.722 | 0.000 | 0.000 / 0.000 / 0.000 |
| L | route | 0.931 | 0.935 / 0.936 / 0.921 | 0.962 | 0.000 | 0.000 / 0.000 / 0.000 |
| L | atomic | 0.824 | 0.861 / 0.696 / 0.915 | 0.913 | 0.000 | 0.000 / 0.000 / 0.000 |
| L | bridgectx | 0.913 | 0.914 / 0.931 / 0.894 | 0.947 | 0.000 | 0.000 / 0.000 / 0.000 |
| Rall | none | 0.694 | 0.701 / 0.704 / 0.678 | 0.741 | 0.002 | 0.002 / 0.002 / 0.003 |
| Rall | route | 0.978 | 0.986 / 0.986 / 0.961 | 0.997 | 0.001 | 0.001 / 0.001 / 0.001 |
| Rall | atomic | 0.810 | 0.850 / 0.669 / 0.910 | 0.908 | 0.069 | 0.012 / 0.179 / 0.015 |
| Rall | bridgectx | 0.954 | 0.953 / 0.967 / 0.942 | 0.979 | 0.001 | 0.000 / 0.000 / 0.002 |
| set | contrast | mean [95% CI] | sign |
|---|---|---|---|
| L | route - atomic | +0.107 [-0.193, +0.407] | 3/3 |
| L | route - none | +0.247 [+0.194, +0.299] | 3/3 |
| L | bridgectx - none | +0.229 [+0.152, +0.306] | 3/3 |
| L | atomic - none | +0.140 [-0.118, +0.397] | 3/3 |
| L | route - bridgectx | +0.018 [-0.010, +0.046] | 3/3 |
| Rall | route - atomic | +0.168 [-0.169, +0.506] | 3/3 |
| Rall | bridgectx - atomic | +0.145 [-0.198, +0.487] | 3/3 |
| Rall | emission: atomic - none | +0.067 [-0.171, +0.305] | 3/3 |
| Rall | emission: bridgectx - none | -0.001 [-0.002, -0.001] | 0/3 |
| Rall | emission: bridgectx - atomic | -0.068 [-0.307, +0.170] | 0/3 |
