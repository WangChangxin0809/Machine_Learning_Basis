import random

# 1.设置随机种子，a为None时，以系统时间为种子
random.seed(a=None)

random.random()  # 生成0 <= x < 1
random.randint(a=1, b=10)  # 生成1 <= x <= 10
random.uniform(a=2.3, b=3.5)  # 生成a与b之间的浮点数

# Functions for sequences
s = [1, 3, 4, 5, 3, 25, 0, 43]
random.choice(s)  # 从s中随机选一个
random.choices(population=s, weights=None, cum_weights=None, k=5)  # weight是权重，k是个数，这是独立实验，可重复
random.sample(population=s, k=5, counts=[2] * 8)  # k是抽取个数，counts是单个元素抽取次数上限
random.shuffle(s)  # 重排

# 常见分布
random.gauss(mu=0, sigma=1.0)  # 正态分布
random.binomialvariate(n=1, p=0.5)  # 二项分布

# numpy中的random
import numpy as np

arr = np.array([[1, 2, 3], [2, 3, 4]])

rng = np.random.default_rng()  # 设置局部随机种子
rng.random()
rng.shuffle(x=arr, axis=1)
rng.choice(a=arr, size=(2, 3), replace=True, p=[0.5] * 2, axis=0)  # a是arr,size是抽取数目，replace是能否被抽取，p是抽取概率，axis是抽取轴
rng.permutation(x=arr, axis=0)  # 打乱顺序
rng.uniform(low=0.0, high=20.0, size=(2, 3))  # 均匀分布
rng.integers(low=1, high=200, size=(2, 3), endpoint=True)  # endpoint是指是否包含high
