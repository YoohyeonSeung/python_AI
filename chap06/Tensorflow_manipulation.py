# Tensor Manipulation
# https://www.tensorflow.org/api_docs/python/


import tensorflow as tf
import numpy as np
import pprint

tf.set_random_seed(777)

pp = pprint.PrettyPrinter(indent=4)

sess = tf.InteractiveSession()

t = np.array([0. , 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t)
print(t.shape)
print(t.ndim)
print(t[0], t[1], t[-1])
print(t[2:5],t[4:-1])
print(t[:2],t[3:])


# 2차원 배열(Matrix:행렬)
t = np.array([  [1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]   ])

pp.pprint(t)
print(t.ndim)
print(t.shape)

## shape, Rank, Axis
t = tf.constant([1, 2, 3, 4])
print(tf.shape(t))       # Tensor("Shape:0", shape=(1,), dtype=int32)
print(tf.shape(t).eval())   # [4]


t=tf.constant([[1, 2],[3, 4]])
print(tf.shape(t))
print(tf.shape(t).eval())


t= tf.constant(     [        [[1, 2, 3, 4],[5, 6, 7, 8]],
                             [[9, 10, 11, 12],[13, 14, 15, 16]],
                             [[17, 18, 19, 20],[21, 22, 23, 24]]    ]      )
print(tf.shape(t))
print(tf.shape(t).eval())


# t = tf.costant([#4차
#                     [#3차
#                         [#2차
#                             [#1차]
#                         ]
#                     ]
#                 ])


t = tf.constant([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                  [[13,14,15,16], [17,18,19,20], [21,22,23,24]]]])  # 4열 하나의 면에. 3행.
print (tf.shape(t).eval()) # [1 2 3 4]

# matmul vs multiply
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
result = tf.matmul(matrix1, matrix2).eval()
print(result) # [[12]], 3*2 + 3*2

result = (matrix1 * matrix2).eval()
print(result) # [[6 6][6 6]]

# Random values for variable initializations
print(tf.random_normal([3]).eval())
print(tf.random_normal([2, 3]).eval())

# Reduce Mean / Sum
print(tf.reduce_mean([2,4]).eval()) # 3

x = [[1.,2.],[3.,4.]]
result = tf.reduce_mean(x).eval()
print(result)  # 2.5

result = tf.reduce_mean(x, axis=0).eval()   # axis=0 은 열을 의미.
print(result)  # [2. 3.]

result = tf.reduce_mean(x, axis=1).eval()   # axis=1 은 행을 의미.
print(result)  # [1.5 3.5]

result = tf.reduce_mean(x, axis=-1).eval()   # axis=-1 은 행끼리의 연산을 수행하라는 의미.
print(result)  # [1.5 3.5]

result = tf.reduce_mean(x, axis=-2).eval()   # axis=-2 은 열끼리의 연산을 수행하라는 의미.
print(result)  # [2. 3.]

# ========================================
result = tf.reduce_sum(x).eval()
print(result) # 10.0

result = tf.reduce_sum(x, axis=0).eval()
print(result) # [4. 6.]

result = tf.reduce_sum(x, axis=1).eval()
print(result) # [3. 7.]

result = tf.reduce_sum(x, axis=-1).eval()
print(result) # [3. 7.]

result = tf.reduce_sum(x, axis=-2).eval()
print(result) # [4. 6.]

result = tf.reduce_mean(tf.reduce_sum(x, axis=1)).eval()
print(result) # 5.0

# argmax with axis
x = [[0,1,2],[2,1,0]]
print(tf.argmax(x, axis=0).eval()) # 열 [1 0 0]
print(tf.argmax(x, axis=1).eval()) # 행중심으로 최대값 찾아가라 [2 0]
print(tf.argmax(x, axis=-1).eval()) # 행
print(tf.argmax(x, axis=-2).eval()) # 열

# Reshape, squeeze, expand_dims
t = np.array([[[0,1,2],[3,4,5]],
              [[6,7,8],[9,10,11]]])
print(t.shape) # (2, 2, 3) 2면 2행 3열 '3차원'

print(tf.reshape(t, shape=[-1, 3]).eval()) # shape을 3열로 형상을 재정의해라! -1은 형태(자유롭게해줘라)  3 (3열)
#2차원으로 차원이 줄어들면서 플랫하게 출력됨 (옆으로 길게 수평 형태로 shape 바꿈)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

result = tf.reshape(t, shape=[-1, 1, 3]).eval()
# [[[ 0  1  2]]
#
#  [[ 3  4  5]]
#
#  [[ 6  7  8]]
#
#  [[ 9 10 11]]]
print(result.shape) # (4, 1, 3)

result = tf.squeeze([[0],[1],[2]]).eval()
print(result) # [0 1 2]
print(result.shape) # (3,)

result = tf.expand_dims([0,1,2], axis=1).eval() #행을 기준으로 확장해라
print(result) # [[0] [1] [2]]
print(result.shape) # (3, 1)

# One-hot Encoding
print(tf.one_hot([[0],[1],[2],[0]], depth=3).eval()) # 4행 1열 # depth에서 지정한 갯수만큼으로 3개의 출력형태 원핫인코딩
# 3차원 배열
# [[[1. 0. 0.]]
#
#  [[0. 1. 0.]]
#
#  [[0. 0. 1.]]
#
#  [[1. 0. 0.]]]

t = tf.one_hot([[0],[1],[2],[0]], depth=3).eval()
print(tf.reshape(t, shape=[-1, 3]).eval()) # 3열로 재구성
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]

# casting # 여태 주로 boolean 0이나 1로 정확도를 계산할 때 활용
print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())  #자바에서 자료형변환(타입캐스트). 입력값은 실수지만 정수값으로 뱉어라!
print(tf.cast([True , False , 1 == 1, 0 == 1], tf.int32).eval())



# Stack

x=[1, 4]
y=[2, 5]
z=[3, 6]
print(tf.stack([x, y, z]).eval())
print(tf.stack([x, y, z], axis=0).eval())  # 넣어준 x, y, z순서대로 데이터를 저장 axis=0일때와 axis=1일때가 다름. 0은 쌓는 객체를 한 행의 형태로 1은 쌓는 객체를 하나의 열의 형태로 만약 2이면 3차원에서의 면의 형태로!
print(tf.stack([x, y, z], axis=1).eval())


# Ones like and Zeros like
x=[[0, 1, 2 ],[ 2, 1, 0 ]]


z = tf.ones_like(x).eval()
print(z) #x의 크기와 똑같은 배열을 만들고 원소는 각각 다 1로 만들어주는것

z= tf.zeros_like(x).eval()
print(z) #x의 크기와 똑같은 배열을 만들고 원소는 각각 다 0로 만들어주는것



# zip
for x , y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

for x , y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)

















