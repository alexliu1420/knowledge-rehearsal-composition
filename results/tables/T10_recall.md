# T10 Full-set injected-fact recall per adapter, both checkpoints
Strict recall of the injected fact over every injected route in the measured half (not the 40-item training-time probe that defines mem100). Files: every post file used in T1, T2 and T6.
| family | experiment | condition | checkpoint | per seed | mean | min |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | preservation | none | mem100 | 0.990 / 0.990 / 0.985 | 0.988 | 0.985 |
| Qwen2.5-3B | preservation | none | final | 0.990 / 0.990 / 0.985 | 0.988 | 0.985 |
| Qwen2.5-3B | preservation | route | mem100 | 0.979 / 0.967 / 0.972 / 0.987 / 0.977 / 0.990 | 0.979 | 0.967 |
| Qwen2.5-3B | preservation | route | final | 0.956 / 0.990 / 0.990 / 0.941 / 0.985 / 0.990 | 0.975 | 0.941 |
| Qwen2.5-3B | preservation | atomic | mem100 | 0.977 / 0.941 / 0.979 / 0.972 / 0.987 / 0.977 | 0.972 | 0.941 |
| Qwen2.5-3B | preservation | atomic | final | 0.879 / 0.949 / 0.977 / 0.959 / 0.987 / 0.990 | 0.957 | 0.879 |
| Qwen2.5-3B | transfer | route | mem100 | 0.985 / 0.987 / 0.990 | 0.987 | 0.985 |
| Qwen2.5-3B | transfer | route | final | 0.987 / 0.990 / 0.990 | 0.989 | 0.987 |
| Qwen2.5-3B | transfer | atomic | mem100 | 0.969 / 0.964 / 0.977 | 0.970 | 0.964 |
| Qwen2.5-3B | transfer | atomic | final | 0.990 / 0.962 / 0.990 | 0.980 | 0.962 |
| Qwen2.5-3B | transfer | coherent | mem100 | 0.982 / 0.979 / 0.990 | 0.984 | 0.979 |
| Qwen2.5-3B | transfer | coherent | final | 0.987 / 0.990 / 0.990 | 0.989 | 0.987 |
| Qwen2.5-3B | transfer | bridgectx | mem100 | 0.956 / 0.985 / 0.982 | 0.974 | 0.956 |
| Qwen2.5-3B | transfer | bridgectx | final | 0.979 / 0.982 / 0.990 | 0.984 | 0.979 |
| Falcon3-3B | preservation | none | mem100 | 0.976 / 0.940 / 0.952 | 0.956 | 0.940 |
| Falcon3-3B | preservation | none | final | 0.994 / 0.995 / 0.995 | 0.995 | 0.994 |
| Falcon3-3B | preservation | route | mem100 | 0.996 / 0.951 / 0.974 / 0.969 / 0.971 / 0.993 | 0.976 | 0.951 |
| Falcon3-3B | preservation | route | final | 0.996 / 0.996 / 0.989 / 0.974 / 0.989 / 0.998 | 0.990 | 0.974 |
| Falcon3-3B | preservation | atomic | mem100 | 0.930 / 0.984 / 0.974 / 0.988 / 0.996 / 0.994 | 0.978 | 0.930 |
| Falcon3-3B | preservation | atomic | final | 0.998 / 0.983 / 0.998 / 0.859 / 0.990 / 0.998 | 0.971 | 0.859 |
| Falcon3-3B | transfer | route | mem100 | 0.980 / 0.996 / 0.993 | 0.990 | 0.980 |
| Falcon3-3B | transfer | route | final | 0.987 / 0.998 / 0.998 | 0.994 | 0.987 |
| Falcon3-3B | transfer | atomic | mem100 | 0.951 / 0.982 / 0.984 | 0.972 | 0.951 |
| Falcon3-3B | transfer | atomic | final | 0.954 / 0.978 / 0.957 | 0.963 | 0.954 |
| Falcon3-3B | transfer | bridgectx | mem100 | 0.996 / 0.888 / 0.995 | 0.960 | 0.888 |
| Falcon3-3B | transfer | bridgectx | final | 0.998 / 0.987 / 0.994 | 0.993 | 0.987 |
