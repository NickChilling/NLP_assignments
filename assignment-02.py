#%% [markdown]
# ## Assignments for Week-02
#%% [markdown]
# In this course, we learnt what's the search problem and what's the machine leanring. In this assignment, we need you do some more practice.
#%% [markdown]
# ### 1. Re-code the house price machine learning
# 
# ###### 1. Random Choose Method to get optimal *k* and *b*
# ###### 2.Supervised Direction to get optimal *k* and *b*
# ###### 3.Gradient Descent to get optimal *k* and *b*
# ###### 4. Try different Loss function and learning rate. 
# 
# For example, you can change the loss function: $Loss = \frac{1}{n} sum({y_i - \hat{y_i}})^2$ to $Loss = \frac{1}{n} sum(|{y_i - \hat{y_i}}|)$
# 
# And you can change the learning rate and observe the performance.

#%%
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data = load_boston()
X, y = data['data'], data['target']
import random
def price(rm,k,b):
    return k*rm+b

#%% [markdown]
# 1. random selection

#%%
def draw_rm_and_price():
    plt.scatter(X[:, 5], y)


#%%
X_rm = X[:,5]
k = random.randint(-100,100)
b = random.randint(-100,100)
price_by_random = [price(r,k,b) for r in X_rm]
draw_rm_and_price()
plt.scatter(X_rm,price_by_random)


#%%
def deviation_loss(y,y_hat):
    return mean([y[i]-y_hat[i] for i in range(len(y))])
def mse_loss(y,y_hat):
    return mean([(y[i]-y_hat[i])**2 for i in range(len(y))])
def abs_loss(y,y_hat):
    return mean([abs(y[i]-y_hat[i]) for i in range(len(y))])
def mean(a):
    return sum(a)/len(a)


#%%

try_times = 2000
min_loss = float('inf')
best_k , best_b = None, None

for i in range(try_times):
    k = random.randint(-100,100)
    b = random.randint(-100,100)
    price_hat = [price(r,k,b) for r in X_rm]
    current_loss = deviation_loss(y, price_hat)
    
    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = k, b
        print('When time is : {}, get best_k: {} best_b: {}, and the loss is: {}'.format(i, best_k, best_b, min_loss))


#%%
min_loss = float('inf')
trying_times = 1000
best_k = random.randint(-100,100)
best_b = random.randint(-100,100)

direction = [(i,j) for i in (1,-1) for j in (1,-1)]

next_direction = random.choice(direction)

scalar = 0.1
for i in range(trying_times):
    
    k_direction, b_direction = next_direction
    
    current_k, current_b = best_k + k_direction * scalar, best_b + b_direction * scalar
    
    price_by_k_and_b = [price(r, current_k, current_b) for r in X_rm]

    current_loss = deviation_loss(y, price_by_k_and_b)

    if current_loss < min_loss: # performance became better
        min_loss = current_loss
        best_k, best_b = current_k, current_b
        
        next_direction = next_direction
        print('When time is : {}, get best_k: {} best_b: {}, and the loss is: {}'.format(i, best_k, best_b, min_loss))
    else:
        next_direction = random.choice(direction)


#%%
def compute_gradient(x,y,k,b,loss_func,delta = 0.0001):
    y_hat =  [price(r, k, b) for r in x]
    delta_y_hat_b = [price(r,k,b+delta) for r in x]
    delta_y_hat_k = [price(r,k+delta,b) for r in x]
    o_loss = loss_func(y,y_hat)
    b_delta_loss = loss_func(y,delta_y_hat_b)
    k_delta_loss = loss_func(y,delta_y_hat_k)
    return (k_delta_loss-o_loss)/delta, (b_delta_loss-o_loss/delta)
trying_times = 1000
best_loss = float('inf')
best_k = random.randint(-100,100)
best_b = random.randint(-100,100)
learning_rate = 0.01
for i in range(trying_times):
    k_gradient, b_gradient = compute_gradient(X_rm,y,best_k,best_b,deviation_loss)
    current_k,current_b = best_k-learning_rate*k_gradient , best_b-learning_rate*b_gradient
    price_by_k_and_b = [price(r, current_k, current_b) for r in X_rm]
    current_loss = deviation_loss(y, price_by_k_and_b)
    if current_loss< best_loss:
        best_loss = current_loss
        print(current_loss,current_k,current_b)
    best_b = current_b
    best_k = current_k

#%% [markdown]
# ## 2. Answer following questions:
# 
# 
# ###### 1. Why do we need machine learning methods instead of creating a complicated formula?
# 
#%% [markdown]
# Ans: 特征与标签之间复杂的函数公式往往难以直接观察出来, 所以需要用机器学习方法帮助我们选择出一个最优的函数映射关系.
#%% [markdown]
# ###### 2.  Wha't's the disadvantages of `the 1st Random Choosen` methods in our course? 
#%% [markdown]
# Ans: 效果完全随机, 需要不断地迭代, 每次迭代很可能不会产生更优的参数.
#%% [markdown]
# ###### 3. Is the `2nd method supervised direction` better than 1st one?  What's the disadvantages of `the 2nd supversied directin` method? 
#%% [markdown]
# Ans:第二个有监督方向更好一些,保证每次更新都是朝着一个更优的点. 但容易陷入局部最优点
#%% [markdown]
# ###### 4. Why do we use `Derivative / Gredient` to fit a target function? 
#%% [markdown]
# Ans: 导数/梯度 决定了函数沿着方向轴变化的方向. 有了导数,可以更精确地决定参数更新的方向
#%% [markdown]
# ###### 5. In the words 'Gredient Descent', what's the `Gredient` and what's the `Descent`?
#%% [markdown]
# Ans: 梯度, 是目标函数对某个方向轴的微分.沿着梯度的反方向更新参数, 可以让目标函数的值下降.因此是梯度下降法
#%% [markdown]
# ###### 6. What's the advantages of `the 3rd gradient descent method` compared to the previous methods?
#%% [markdown]
# Ans: 确保了每次迭代都能离最优点/局部最优点 更进一步
#%% [markdown]
# ###### 7. Using the simple words to describe: What's the machine leanring.
#%% [markdown]
# Ans:机器学习即是让计算机自动寻找到一个函数.能够拟合出特征和标签的映射关系
#%% [markdown]
# ## 3. Finish the search problem
#%% [markdown]
# Please using the search policy to implement an agent. This agent receives two input, one is @param start station and the other is @param destination. Your agent should give the optimal route based on Beijing Subway system. 
#%% [markdown]
# > Deadline: 2019-July-13
# 
# >Submit: Submit the source code and result to github. 
# 
# 
#%% [markdown]
# ![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1562414356407&di=b57814aafd215bb8b8d9d8cd37c573d6&imgtype=0&src=http%3A%2F%2Fcli.clewm.net%2Ffile%2F2015%2F03%2F24%2F174ed60082b8422ac0636cfd3efb9e7f.jpg)
#%% [markdown]
# #### Dataflow: 
#%% [markdown]
# ##### 1.	Get data from web page.
# 
# > a.	Get web page source from: https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485
# 
# > b.	You may need @package **requests**[https://2.python-requests.org/en/master/] page to get the response via url
# 
# > c.	You may need save the page source to file system.
# 
# > d.	The target of this step is to get station information of all the subway lines;
# 
# > e.	You may need install @package beautiful soup[https://www.crummy.com/software/BeautifulSoup/bs4/doc/]  to get the url information, or just use > Regular Expression to get the url.  Our recommendation is that using the Regular Expression and BeautiflSoup both. 
# 
# > f.	You may need BFS to get all the related page url from one url. 
# Question: Why do we use BFS to traverse web page (or someone said, build a web spider)?  Can DFS do this job? which is better? 

#%%
import requests
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
req = requests.get('https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485',headers=headers)
req.encoding = 'utf-8'
res = req.text 


#%%
from bs4 import BeautifulSoup
bs = BeautifulSoup(res)


#%%
import re 
lines_re = r'<a target=_blank href="(/item/(%\w+)+)">(北京地铁\w+线)'
pattern = re.compile(lines_re)
re_res = pattern.findall(res)


#%%
lines_dict = {}
for i in re_res:
    lines_dict[i[2]] = i[0]


#%%
lines_dict


#%%
lines_res = []
for line,url in lines_dict.items():
    result = []
    line_response = requests.get('https://baike.baidu.com'+lines_dict[line],headers=headers)
    line_response.encoding = 'utf-8'
    line_res = line_response.text
    bs = BeautifulSoup(line_res)
    bs_res = list(bs.find_all('td'))
    re_str = r'>(\w+站)</a>'
    pat = re.compile(re_str)
    for sl in bs_res:
        station = pat.findall(str(sl))
        if station and (station[0] not in result):
            result.append(station[0])
    lines_res.append(result)
print(lines_res)


#%%
from collections import defaultdict
connection_map = defaultdict(set)
for line in lines_res:
    for i,station in enumerate(line):
        if i == 0:
            connection_map[station].add(line[i+1])
        elif i == len(line)-1:
            connection_map[station].add(line[i-1])
        else:
            connection_map[station].add(line[i-1])
            connection_map[station].add(line[i+1])


#%%
import ast
import time


#%%
city_location = {}
key = "EM7BZ-BH5WU-JAWVE-22XVP-XNCYS-KBFG4"
sk = "S1FrJEtStGYktJsa1CMiRcePqZPRaXj"
for station in connection_map.keys():
    htp = "https://apis.map.qq.com/ws/geocoder/v1/?address={}地铁站&key={}&region=北京".format(station.split('站'),key)
    positon_req = requests.get(htp)
    positon_dict = ast.literal_eval(positon_req.text)
    city_location[station] = (positon_dict['result']['location']['lng'],positon_dict['result']['location']['lat'])


#%%
get_ipython().run_line_magic('pdb', '')

#%% [markdown]
# %pdb

#%%
import networkx as nx 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
subway = nx.Graph(connection_map)
plt.figure(figsize=(20,20))
nx.draw(subway,city_location,with_labels=True,node_size = 20,font_size = 10)

#%% [markdown]
# 由于查询结果不精确和地图精度问题. 地图效果并不好

#%%
def search(start,end,connection):
    pathes = [[start]]
    while pathes:
        path = pathes.pop(0)
        frontier = path[-1]
        next_stations = connection[frontier]
        for ns in next_stations:
            if ns in path:
                continue
                
            else:
                new_path = path+[ns]
                pathes.append(new_path)
                
                if ns == end:
                    return new_path
        pathes = sorted(pathes,key=len)
        
search('车公庄西站','公益西桥站',connection_map)

#%% [markdown]
# ##### 2.	Preprocessing data from page source.
# 
# > a.	Based on the page source gotten from url. You may need some more preprocessing of the page. 
# 
# > b.	the Regular Expression you may need to process the text information.
# 
# > c.	You may need @package networkx, @package matplotlib to visualize data. 
# 
# > d.	You should build a dictionary or graph which could represent the connection information of Beijing subway routes. 
# 
# > e.	You may need the defaultdict, set data structures to implement this procedure. 
#%% [markdown]
# ##### 3. Build the search agent
# 
# > Build the search agent based on the graph we build.
# 
# for example, when you run: 
# 
# ```python
# >>> search('奥体中心', '天安门') 
# ```
# you need get the result: 
# 
# 奥体中心-> A -> B -> C -> ... -> 天安门
# 
# 
#%% [markdown]
# ## （Optional）Create different policies for transfer system.
# 
#%% [markdown]
# 以下部门为可选部分，请酌情完成。 并不要求全部同学完成。
#%% [markdown]
# As much as you can to use the already implemented search agent. You just need to define the **is_goal()**, **get_successor()**, **strategy()** three functions. 
# 
# > a.	Define different policies for transfer system. 
# 
# > b.	Such as Shortest Path Priority（路程最短优先）, Minimum Transfer Priority(最少换乘优先), Comprehensive Priority(综合优先)
# 
# > c.	Implement Continuous transfer. Based on the Agent you implemented, please add this feature: Besides the @param start and @param destination two stations, add some more stations, we called @param by_way, it means, our path should from the start and end, but also include the  @param by_way stations. 
# 
# e.g 
# ```
# 1. Input:  start=A,  destination=B, by_way=[C] 
#     Output: [A, … .., C, …. B]
# 2. Input: start=A, destination=B, by_way=[C, D, E]
#     Output: [A … C … E … D … B]  
#     # based on your policy, the E station could be reached firstly. 
# ![image.png](attachment:image.png)
# ```
#%% [markdown]
# ##### 5.	Test your result with commercial applications. 
# 
# 将你的结果和高德地图或者百度地图进行比较，如果有不同，请分析原因
# 
#%% [markdown]
# 恭喜，完成本次课程，你对常用的人工智能方法以及有一定的了解了。基于规则的，基于概率模型的，基于搜索的，基于机器学习的。 可以说，我们现在通常见到的方法都能够归属到这几类方法中。 这就是**人工智能**，并没有很难是吧？ 继续加油！
#%% [markdown]
# ![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1562415163815&di=4b29a2a863a8285212033760f288ed7a&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20180710%2F8704194a1d7f46a383ddc29d40c9bca5.jpeg)

