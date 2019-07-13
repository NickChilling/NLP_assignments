
#%%
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Video

#%% [markdown]
# # 介绍
#%% [markdown]
# `Numpy` 提供多维数组对象以及一系列操作数组的函数, 可以说其几乎是每一个`Python`科学计算软件的基础库.

#%%
import numpy as np

#%% [markdown]
# `Numpy`的核心数据结构是`ndarray`, 它用来存储具有相同数据类型的多维数组. 除了数据, `ndarry`也包含数组的`shape, size, ndim, nbytes, dtype`.

#%%
get_ipython().run_line_magic('pinfo', 'np.ndarray')


#%%
d0 = np.array([[1, 2], [3,4]])


#%%
d0


#%%
type(d0), d0.shape, d0.size, d0.ndim, d0.dtype, d0.nbytes

#%% [markdown]
# 为什么需要`numpy`? 速度! 简单! 粗略比较一下速度.

#%%
a0 = np.arange(10000)
t0 = get_ipython().run_line_magic('timeit', '-o [i**2 for i in a0]')


#%%
a1 = np.arange(10000)
t1 = get_ipython().run_line_magic('timeit', '-o a1**2')


#%%
t0.best/t1.best

#%% [markdown]
# ## 数据类型
#%% [markdown]
# ![Numpy Dtypes](./image/numpy_types.png)
# 
# 详细参考: [numpy datatypes](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).

#%%
x = np.array([1, 2])  # Let numpy choose the datatype
y = np.array([1.0, 2.0])  # Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64)  # Force a particular datatype

print(x.dtype, y.dtype, z.dtype)


#%%
z.nbytes*8 # nbits


#%%
x1 = x + 0.3
print(x1, x1.dtype)


#%%
z[0] = 3.5
z

#%% [markdown]
# 如何使用类型: 一般指定int, float, complex 即可, 不需要细分int16, int32等

#%%
np.sqrt([-1, 2, 3])


#%%
np.sqrt([-1, 2, 3], dtype=np.complex)

#%% [markdown]
# ## 存储顺序
#%% [markdown]
# 多维数组在内存中是连续储存的, 本质上可以看成是一维, 如何将内存中数据映射到多维数组中取决于数组是按行存储的还是按列存储的. 例如有四个整数1,2,3,4, 那么:
# 
# * 按行存储就是: [[1, 2], [3, 4]]
# * 按列存储就是: [[1, 3], [2, 4]]
# 
# Fotran是按列存储的, C是按行存储的.

#%%
a = np.arange(6, dtype=np.int8)


#%%
a


#%%
a1 = a.reshape(2,3, order="F")
a1


#%%
a2 = a.reshape(2,3)
a2

#%% [markdown]
# 什么时候需要考虑存储顺序? 
# 
# 跟其他语言交互的时候, 比如调用`Fortran`(`Numpy`, `Scipy`中很多数值
# 就是调用`Fortran`的, `Anconda`现在默认使用`intel mkl`也是`Fortran`的), 但是平常使用不需要关心顺序.
# 
# `Numpy`中使用`ndarray.strides`确定映射的顺序.

#%%
a1.strides


#%%
a2.strides

#%% [markdown]
# `strides`确定对应维度移动一个元素应内存中移动的字节数, 如对应a1, 有(1x1, 2x1), 对应a2, 有(3x1, 1x1).
# 
# 某些操作, 如`transpose, reshape`, 只需要改变`strides`即可.

#%%
a = np.random.rand(10, 3)


#%%
a.strides


#%%
b = a.transpose()


#%%
b.strides


#%%
np.shares_memory(a, b)


#%%
c = a.reshape(3, 10)


#%%
np.shares_memory(a, c)

#%% [markdown]
# ## 帮助

#%%
get_ipython().run_line_magic('pinfo', 'np.array')


#%%
get_ipython().run_line_magic('psearch', 'np.con*')


#%%
np.array([[1, 2], [3, 4]])

#%% [markdown]
# google: numpy add a row
#%% [markdown]
# # 数组创建
#%% [markdown]
# ## 从列表创建

#%%
a = np.array([[1,2,3], [4, 5, 6]])
b = np.array([1, 2, 3])


#%%
a


#%%
b


#%%
print("a:", a.shape, a.size, type(a), np.ndim(a))
print("b:", b.shape, b.size, type(b), np.ndim(b))


#%%
len(a), len(b) # 返回第一个维度的长度


#%%
np.array([i for i in range(10) if i % 2 ==0])


#%%
a, a.shape[0], len(a)

#%% [markdown]
# ## 使用`Numpy`函数创建
#%% [markdown]
# ![Numpy create array functions](./image/func.png)

#%%
np.zeros(2, 3)  # Create an array of all zeros


#%%
np.zeros((2, 3))


#%%
np.ones((5, 5))   # Create an array of all ones


#%%
np.full((2,2), 7) # Create a constant array


#%%
np.eye(2)        # Create a 2x2 identity matrix


#%%
np.random.random((2,2)) # Create an array filled with random values


#%%
np.arange(9).reshape(3,-1)


#%%
np.linspace(0, 1.0, 10)


#%%
np.tril(np.arange(9).reshape(3,-1)) # np.triu


#%%
np.random.rand(3,3)

#%% [markdown]
# ## 从文件读取
#%% [markdown]
# ![Numpy create array functions](./image/numpy_func1.png)

#%%
get_ipython().system('head test.dat')
get_ipython().system('wc test.dat')


#%%
a = np.genfromtxt("test.dat", delimiter=",", comments="#")


#%%
a[:10], a.shape


#%%
# 保存数组到可读文件
np.savetxt("test.dat", np.random.random((1000, 5)), delimiter=",", header="show how to save array dat.\n a simple example")

# 保存二进制文件
np.savetxt("test.npy", np.random.random((1000, 5)))


#%%



#%%
# 读取大文件
def generate_text_file(length=1e6, ncols=20):
    data = np.random.random((int(length), int(ncols)))
    np.savetxt('large_text_file.csv', data, delimiter=',')

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


#%%
# generate_text_file() # 477M


#%%
get_ipython().system('ls -lh large_text_file.csv')


#%%
get_ipython().run_line_magic('time', 'data = np.genfromtxt(\'large_text_file.csv\', delimiter=",")')


#%%
# %time data = iter_loadtxt('large_text_file.csv')


#%%
get_ipython().run_line_magic('time', "data = pd.read_csv('large_text_file.csv')")

#%% [markdown]
# # 索引与切片
#%% [markdown]
# ![Numpy index](./image/index.png)
#%% [markdown]
# [Numpy index](https://docs.scipy.org/doc/numpy/user/basics.indexing.html)

#%%
a = np.array([[i+j for j in range(6)] for i in range(0, 60, 10)])
a


#%%
[[i+j for j in range(6)] for i in range(0, 60, 10)]


#%%
np.repeat(range(5), 5)


#%%
np.arange(6) + np.arange(0, 60, 10).reshape(6, -1)

#%% [markdown]
# ## 索引 (indexing)

#%%
print(a[0, 0], a[1, 2], a[2, 2])


#%%
a[0, 2] = 3

#%% [markdown]
# ## 切片 (Slicing)

#%%
a[:, 0], a[1, :], a[[1], :]


#%%
a[0, 3:5]


#%%
a[4:, 4:]


#%%
a[:, 2]


#%%
a[2::2, ::2]

#%% [markdown]
# ## 整数数组索引(fancing index)

#%%
a[[1, 3], :] 


#%%
a[[1, 3], :] = 5


#%%
a

#%% [markdown]
# ## 条件索引

#%%
a = np.random.random(10)*2 + -1 # random with(-1, 1)
a


#%%
a[a>0.3] = 20


#%%
a>0.3, a


#%%
a

#%% [markdown]
# ## copies and views

#%%
a = np.arange(10)
b = a[::2]
b1 = a[5:]
c = a.copy()


#%%
np.may_share_memory(a, b)


#%%
np.may_share_memory(a, c)


#%%
np.may_share_memory(b, b1)


#%%
a.base is c


#%%
b.base is a


#%%
b.base


#%%
print(c.base)


#%%
a  = np.random.rand(5, 3)


#%%
a


#%%
c = a.flatten()
c


#%%
d = a.ravel()
d


#%%
d = 0


#%%
type(d)


#%%
d[:] = 0


#%%
d


#%%
a


#%%
c


#%%
get_ipython().run_line_magic('pinfo', 'np.copy')

#%% [markdown]
# ![Copy and Views](./image/copy_views.png)
#%% [markdown]
# [Views versus copies in NumPy](https://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html)
#%% [markdown]
# 建议: 当你不想改变原数组的时候使用`np.copy`
#%% [markdown]
# # 数组操作
#%% [markdown]
# [Numpy functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs)
#%% [markdown]
# ## 数学操作

#%%
x = np.random.random((3,3))
y = np.random.random((3,3))

print(x,'\n\n', y)


#%%
x+y, x*y, x/y, x-y


#%%
np.sin(x), np.sqrt(y)


#%%
x@y, x.dot(y)  # 矩阵乘法, x.dot(y) in Py2


#%%
x.T, x.T.T # 转置


#%%
dir(x)

#%% [markdown]
# ## aggregate function
# 
# ![aggregate function](./image/aggregate_func.png)

#%%
z = np.arange(6).reshape(2, 3)
z


#%%
np.max(z), np.max(z, axis=0), np.max(z, axis=1)


#%%
z.shape


#%%
np.sum(z), np.sum(z, axis=0), np.sum(z, axis=1)

#%% [markdown]
# ![Numpy axis](./image/axis.png)

#%%
a = np.arange(21)


#%%
b = a.reshape(3, 7)
b 


#%%
np.sum(b, axis=0)


#%%
a = np.random.rand(10, 9, 7)


#%%
a.sum(axis=2).shape


#%%



#%%


#%% [markdown]
# ## 条件表达式
# 
# ![Numpy condition expression](./image/condition.png)

#%%
a = np.random.random(10)*2 + -1 # random with(-1, 1)
a


#%%
np.where(a>0.3)


#%%
np.where((a > 0.3) & (np.sin(a) > 0.6 ))


#%%
a[np.where((a > 0.3) & (np.sin(a) > 0.6 ))]


#%%



#%%
np.select([a>0, a<0], [a, a*-1])


#%%
a = np.random.rand(2,3)


#%%
a


#%%
a[np.where(a>0.5)] = 10


#%%
a


#%%
np.where(a>0.5)


#%%


#%% [markdown]
# ## Broadcasting(广播)
#%% [markdown]
# ![Numpy boradcasting](./image/broadcasting.png)

#%%
a = np.tile(np.arange(0, 40, 10), (3, 1)).T
b = np.array([0, 1, 2])
c = np.empty_like(a)   

for i in range(a.shape[0]):
    c[i, :] = a[i, :] + b

print(a, "\n\n", c)


#%%
b1 = np.tile(b, (a.shape[0], 1))
c = a + b1
print(a, "\n\n", a + b)


#%%
np.tile(b, (a.shape[0], 3))


#%%
c = a+b
print(a, "\n\n", c)


#%%
a


#%%
a = np.arange(0, 40, 10)
a = a.reshape(4, 1)
# a = a[:, np.newaxis]  # adds a new axis -> 2D array

b = np.array([0, 1, 2])
c = a+b

print(a, "\n\n", b,"\n\n", c)


#%%
a * b # universal function: https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs

#%% [markdown]
# ## 数组形状操作
#%% [markdown]
# ![Numpy shape functions](./image/shape_func.png)

#%%
a = np.array([[1, 2, 3], [4, 5, 6]])
a


#%%
print(a.ravel(), "\n\n", a.flatten())


#%%
get_ipython().run_line_magic('pinfo', 'a.flatten')


#%%
a.reshape(-1)


#%%
a.reshape(2, -1)
a


#%%
a.reshape(2, 3)
a


#%%
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])


#%%
np.concatenate((a, b), axis=0)


#%%
np.concatenate((a, b.T), axis=1)


#%%
get_ipython().run_line_magic('pinfo2', 'np.append')


#%%
a1 = np.arange(4)
b1 = a1 + 2
c1 = a1 + 3


#%%
a1, b1, c1


#%%
np.vstack((a1, b1, c1))


#%%
np.hstack((a1, b1, c1))


#%%
get_ipython().run_line_magic('pinfo2', 'np.hstack')


#%%



#%%
a = np.array((1,2,3))
b = np.array((2,3,4))
np.hstack((a,b))

#%% [markdown]
# # 例子
#%% [markdown]
# ## 康威生命游戏
# 
# 1. 当前细胞为存活状态时，当周围低于2个（不包含2个）存活细胞时， 该细胞变成死亡状态。（模拟生命数量稀少）
# 1. 当前细胞为存活状态时，当周围有2个或3个存活细胞时， 该细胞保持原样。
# 1. 当前细胞为存活状态时，当周围有3个以上的存活细胞时，该细胞变成死亡状态。（模拟生命数量过多）
# 1. 当前细胞为死亡状态时，当周围有3个存活细胞时，该细胞变成存活状态。 （模拟繁殖）
# 
# Ref:
# 
# 康威生命游戏: [中文](https://zh.wikipedia.org/wiki/%E5%BA%B7%E5%A8%81%E7%94%9F%E5%91%BD%E6%B8%B8%E6%88%8F), [英文](https://en.wikipedia.org/wiki/The_Game_of_Life)
#%% [markdown]
# ![The Game of Life](./image/glider.png)
#%% [markdown]
# ### Python 实现

#%%
def compute_neighbours(Z):
    shape = len(Z), len(Z[0])
    N  = [[0,]*(shape[0]) for i in range(shape[1])]
    for x in range(1,shape[0]-1):
        for y in range(1,shape[1]-1):
            N[x][y] = Z[x-1][y-1]+Z[x][y-1]+Z[x+1][y-1]                     + Z[x-1][y]            +Z[x+1][y]                       + Z[x-1][y+1]+Z[x][y+1]+Z[x+1][y+1]
    return N


#%%
def iterate(Z):
    N = compute_neighbours(Z)
    shape = len(Z), len(Z[0])
    for x in range(1,shape[0]-1):
        for y in range(1,shape[1]-1):
            if Z[x][y] == 1 and (N[x][y] < 2 or N[x][y] > 3):
                Z[x][y] = 0
            elif Z[x][y] == 0 and N[x][y] == 3:
                Z[x][y] = 1
    return Z


#%%
Z = [[0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0]]

Z1 = np.zeros((6, 6), dtype=int)
Z1[1:-1, 1:-1] = Z

Z = Z1
Z


#%%
fig, axes = plt.subplots(3, 4, figsize=(6, 8))

axes = axes.flatten()

axes[0].matshow(Z, cmap="Greys")
axes[0].set_xlabel("iteration %s" % 0)

for i in range(1,12):
    Z = iterate(Z)
    axes[i].matshow(np.array(Z)[1:-1, 1:-1], cmap="Greys")
    axes[i].set_xlabel("iteration %s" % i)

#%% [markdown]
# ### Numpy 实现 1

#%%
Z = [[0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0]]

Z1 = np.zeros((6, 6), dtype=int)
Z1[1:-1, 1:-1] = Z

Z = Z1


#%%
N = np.zeros(Z.shape, dtype=int)
N[1:-1,1:-1] += (Z[ :-2, :-2] + Z[ :-2,1:-1] + Z[ :-2,2:] +
                 Z[1:-1, :-2]                + Z[1:-1,2:] +
                 Z[2:  , :-2] + Z[2:  ,1:-1] + Z[2:  ,2:])


#%%
N


#%%
# Flatten arrays
N_ = N.ravel()
Z_ = Z.ravel()

# Apply rules
# 当前细胞为存活状态时，当周围低于2个（不包含2个）存活细胞时， 该细胞变成死亡状态。（模拟生命数量稀少）
R1 = np.argwhere( (Z_==1) & (N_ < 2) )
# 当前细胞为存活状态时，当周围有2个或3个存活细胞时， 该细胞保持原样。
R2 = np.argwhere( (Z_==1) & (N_ > 3) )
# 当前细胞为存活状态时，当周围有3个以上的存活细胞时，该细胞变成死亡状态。（模拟生命数量过多）
R3 = np.argwhere( (Z_==1) & ((N_==2) | (N_==3)) )
# 当前细胞为死亡状态时，当周围有3个存活细胞时，该细胞变成存活状态。 （模拟繁殖）
R4 = np.argwhere( (Z_==0) & (N_==3) )

# Set new values
Z_[R1] = 0
Z_[R2] = 0
Z_[R3] = Z_[R3]
Z_[R4] = 1

# Make sure borders stay null
Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = 0


#%%
Z

#%% [markdown]
# ### Numpy 实现 2

#%%
Z = [[0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0]]

Z1 = np.zeros((6, 6), dtype=int)
Z1[1:-1, 1:-1] = Z

Z = Z1


#%%
N = np.zeros(Z.shape, dtype=int)
N[1:-1,1:-1] += (Z[ :-2, :-2] + Z[ :-2,1:-1] + Z[ :-2,2:] +
                 Z[1:-1, :-2]                + Z[1:-1,2:] +
                 Z[2:  , :-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

# 当前细胞为存活状态时，当周围有2个或3个存活细胞时， 该细胞保持原样。
birth = (N==3)[1:-1,1:-1] & (Z[1:-1,1:-1]==0)
# 当前细胞为死亡状态时，当周围有3个存活细胞时，该细胞变成存活状态。 （模拟繁殖）
survive = ((N==2) | (N==3))[1:-1,1:-1] & (Z[1:-1,1:-1]==1)
Z[...] = 0
Z[1:-1,1:-1][birth | survive] = 1


#%%
Video("game-of-life.mp4")


#%%


#%% [markdown]
# # 参考
# 
# 1. [Numerical Python: A Practical Techniques Approach for Industry](https://www.amazon.com/Numerical-Python-Practical-Techniques-Approach/dp/1484205545)
# 1. [Guide to NumPy, by the creator of Numpy](http://web.mit.edu/dvp/Public/numpybook.pdf)
# 1. [Scipy Lecture Notes](http://www.scipy-lectures.org/intro/numpy/)
# 1. [From Python to Numpy](http://www.labri.fr/perso/nrougier/from-python-to-numpy)

#%%



#%%



