import matplotlib.pyplot as plt
import numpy as np

# (1, 2)점 하나를 갖는 경우
# xs = [1]
# ys = [2]
#
# plt.plot(xs, ys, 'ro')
# plt.show()


# (1, 2), (2, 5) 두 점을 갖는 경우이다.

xs = [1, 2]
ys = [2, 5]

# 빨간 점 찍기
plt.plot(xs, ys, 'ro')

# 두점 사이에 파란색 긋기

plt.plot(xs, ys, 'b')
plt.xlim(0, 3)
plt.ylim(0.1, 6)
plt.show()