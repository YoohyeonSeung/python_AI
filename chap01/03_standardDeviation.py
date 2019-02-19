import matplotlib.pyplot as plt
import numpy as np
import math

xs = [1, 2]
ys = [2, 5]

plt.plot(xs, ys, 'rx')
plt.xlim(0, 3)
plt.ylim(0.1, 6)
 # w 예상치
w = 1
 # b 예상치
b = 2

#numpy를 이용해서 y라는 배열을 xs 배열을 이용해 쉽게 계산.
y = w * np.array([1, 2]) +b


# 편차의 합
sumOfDeviation = 0.0

# 분산의 합
sumOfVariance = 0.0

# 표준 편차의 합
sumOfStandardDeviation = 0.0

for i, x in enumerate(xs):
    deviation = ys[i]-y[i]       # 편차
    variance  = deviation ** 2
    standardDeviation = math.sqrt(variance)  # 표준 편차

    sumOfVariance += variance
    sumOfDeviation += deviation
    sumOfStandardDeviation +=standardDeviation

    print("x가 ", x, "일때, 실제값 : ", ys[i], ", 예상값 : ", y[i])
    print("      편차 : ", deviation)
    print("      분산 : ", variance)
    print("      표준 편차 : ", standardDeviation)

print("편차의 합 : ", sumOfDeviation)
print("분산의 합 : ", sumOfVariance)
print("표준 편차의 합 : ", sumOfStandardDeviation)









plt.plot(xs, y, 'b')
plt.show()

