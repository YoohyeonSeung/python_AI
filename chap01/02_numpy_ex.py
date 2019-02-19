import matplotlib.pyplot as plt
import numpy as np
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

plt.plot(xs, y, 'b')
plt.show()

