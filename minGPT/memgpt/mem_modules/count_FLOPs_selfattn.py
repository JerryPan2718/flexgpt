#### operation cost in unit ####
MM = 1
MFS = 1
ZSC = 1
TD = 10

#### configs ####
# Parameters: layers, d_model
# 117M: 12, 768
# 345M: 24, 1024
# 762M: 36, 1280
# 1542M: 48, 1600
B, H = (24, 1024)
K, Tc, Tg = (8, 128, 512)

#### Calculation
M = 2 * B * Tc ** 3 * MM + B * Tc ** 2 * MFS + sum([2 * B * (Tc + i) ** 2 * MM + 2 * B * (Tc + i) ** 2 * ZSC + B * (Tc + i) ** 2 * MFS for i in range(1, Tg)])
N = sum([2 * B * (Tc + i) ** 3 * MM + B * (Tc + i) ** 2 for i in range(0, Tg)])
C = sum([B * K * (Tc + i) ** 2 * TD for i in range(0, Tg)])
P = N / (N + C)
S = N / M

Speedup = (N + C) / (C + M)
print(f"M: {M}")
print(f"N: {N}")
print(f"C: {C}")
print(f"P: {P}")
print(f"S: {S}")
print(f"Speedup: {Speedup}")
