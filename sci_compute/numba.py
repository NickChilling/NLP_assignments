#%% [markdown]
# # 实例: 矩阵乘法

#%%
get_ipython().run_line_magic('pylab', 'inline')
import numpy as np


#%%
import numba
from numba import njit, vectorize


#%%
from IPython.display import Image, Video


#%%
m = 5
n = 3
p = 6
a = np.random.rand(n, m)
b = np.random.rand(m, p)


#%%
c0 = a@b


#%%
c0


#%%
t0 = get_ipython().run_line_magic('timeit', '-o  a@b')


#%%
def dot_py(a, b):
    n, m = a.shape
    m, p = b.shape
    c = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                c[i][j] += a[i][k] * b[k][j]
    return c


#%%
c1 = dot_py(a,b)


#%%
t1 = get_ipython().run_line_magic('timeit', '-o dot_py(a,b)')


#%%
t1.best/t0.best


#%%
@njit(parallel=True, fastmath=True, nogil=True)
def dot_numba(a, b):
    n, m = a.shape
    m, p = b.shape
    c = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                c[i][j] += a[i][k] * b[k][j]
    return c


#%%
dot_numba = njit(parallel=True, fastmath=True, nogil=True)(dot_py)


#%%
c2 = dot_numba(a,b)
c2 


#%%
t2 = get_ipython().run_line_magic('timeit', '-o  dot_numba(a,b)')


#%%
t1.best/t2.best


#%%
print("numba: ", t1.best/t2.best)
print("numpy: ", t1.best/t0.best)

#%% [markdown]
# # 分段函数
#%% [markdown]
# $$ 
# \Theta(x)=\left\{\begin{array}{l}{0, x<0} \\ {\frac{1}{2}, x=0} \\ {1, x>0}\end{array}\right.
#  $$

#%%
def py_Heaviside(x):
    if x == 0.0:
        return 0.5
    elif x < 0.0:
        return 0.0
    else:
        return 1.0


#%%
x = np.linspace(-5, 5, 10000)


#%%
get_ipython().run_line_magic('timeit', '[py_Heaviside(i) for i in x]')


#%%
np_vec_Heaviside = np.vectorize(py_Heaviside)


#%%
np_vec_Heaviside(x)

#%% [markdown]
# $[x_0, x_1, x_2, \dots$]
#%% [markdown]
# $[f(x_0), f(x_1), f(x_2), \dots]$

#%%
np.sin(x)


#%%
get_ipython().run_line_magic('timeit', 'np_vec_Heaviside(x)')


#%%
def np_Heaviside(x):
    return (x > 0.0) + (x == 0.0)*0.5


#%%
get_ipython().run_line_magic('timeit', 'np_Heaviside(x)')


#%%
@vectorize(fastmath=True)
def jit_Heaviside(x):
    if x == 0.0:
        return 0.5
    elif x < 0.0:
        return 0.0
    else:
        return 1.0


#%%
62.7/7.34


#%%
get_ipython().run_line_magic('timeit', 'jit_Heaviside(x)')


#%%
2.27*1000/7.34

#%% [markdown]
# # Julia fractal

#%%
Image("julia_set.gif", width=500, height=500)

#%% [markdown]
# $$ 
# f_{c}(z)=z^{2}+c,
# $$
# where $c$ is complex number. The [julia set](https://en.wikipedia.org/wiki/Julia_set) for this system is the subset of the complex plane given by
# $$ 
# J\left(f_{c}\right)=\left\{z \in \mathbb{C} : \forall n \in \mathbb{N},\left|f_{c}^{n}(z)\right| \leq 2\right\}
#  $$
# 
# more interesting site:
# 
# * http://www.relativitybook.com/CoolStuff/julia_set.html
# * http://www.relativitybook.com/CoolStuff/julia_set_4d.html

#%%
def py_julia_fractal(z_re, z_im, j, n=2, nmax=500, c=0.279):
    for m in range(len(z_re)):
        for n in range(len(z_im)):
            z = z_re[m] + 1j * z_im[n]
            for t in range(nmax):
                z = z ** 2 + c
                if np.abs(z) > 2.0:
                    j[m, n] = t 
                    break


#%%
jit_julia_fractal = njit(parallel=True, fastmath=True, nogil=True)(py_julia_fractal)


#%%
N = 1024

z_real = np.linspace(-1.5, 1.5, N)
z_imag = np.linspace(-1.5, 1.5, N)


#%%
# %timeit -n 1 -r 1 py_julia_fractal(z_real, z_imag, j)


#%%
get_ipython().run_line_magic('timeit', 'jit_julia_fractal(z_real, z_imag, j)')


#%%
22.7*1000/97


#%%
j = np.zeros((N, N), np.int64)
jit_julia_fractal(z_real, z_imag, j, c = 1-(1+5**0.5)/2, nmax=100)
imshow(j, cmap=plt.cm.hot, extent=[-1.5, 1.5, -1.5, 1.5])
axis('off')

#%% [markdown]
# ![image.png](attachment:image.png)

#%%
j = np.zeros((N, N), np.int64)
jit_julia_fractal(z_real, z_imag, j, c = -0.11+0.65569999j, nmax=100)
imshow(j, cmap=plt.cm.gist_ncar, extent=[-1.5, 1.5, -1.5, 1.5])
axis('off')


#%%
j = np.zeros((N, N), np.int64)
jit_julia_fractal(z_real, z_imag, j, c =  0.279, nmax=100)
imshow(j, cmap=plt.cm.gist_ncar, extent=[-1.5, 1.5, -1.5, 1.5])
axis('off')

#%% [markdown]
# ![image.png](attachment:image.png)

#%%



