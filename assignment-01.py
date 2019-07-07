
#%%
# ms-python.python added
import os
try:
    os.chdir(os.path.join(
        os.getcwd(), 'jupyters_and_slides/2019-summer/assignments'))
    print(os.getcwd())
except:
    pass

#%% [markdown]
#  ## Lesson-01 Assignment
#%% [markdown]
#  >
#%% [markdown]
#  `各位同学大家好，欢迎各位开始学习我们的人工智能课程。这门课程假设大家不具备机器学习和人工智能的知识，但是希望大家具备初级的Python编程能力。根据往期同学的实际反馈，我们课程的完结之后 能力能够超过80%的计算机人工智能/深度学习方向的硕士生的能力。`
#%% [markdown]
#  ## 本次作业的内容
#%% [markdown]
#  #### 1. 复现课堂代码
# 
#  在本部分，你需要参照我们给大家的GitHub地址里边的课堂代码，结合课堂内容，复现内容。

#%%
simple_grammer = '''sentence => noun_phrase verb_phrase
noun_phrase => Article Adj* noun
verb_phrase => verb noun_phrase
Adj* => null | Adj Adj*
Article => 这 | 那
noun => 女人 |球| 桌 | 猫
verb => 观察 | 坐在 | 听 | 看
Adj => 蓝色的 | 好看的 | 小的
'''


#%%
import random

def create_grammar(grammar_str,split='=>',line_split='\n'):
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip(): continue
        exp, stmt = line.split(split)
        grammar[exp.strip()] = [s.split() for s in stmt.split('|') ]
    return grammar

choice = random.choice

def generate(grammar,target):
    if target not in grammar:return target
    expaned = [generate(grammar,t) for t in choice(grammar[target])]
    return ''.join([e  for e in expaned if e != 'null'])


#%%
example = create_grammar(simple_grammer)


#%%
generate(example,target='sentence')


#%%
# 西部世界
human = """
human = 自己 寻找 活动
自己 = 我 | 俺 | 我们 
寻找 = 找找 | 想找点 
活动 = 乐子 | 玩的
"""

host = """
host = 寒暄 报数 询问 业务相关 结尾 
报数 = 我是 数字 号 ,
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ,
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好 
询问 = 请问你要 | 您需要
业务相关 = 玩玩 具体业务
玩玩 = null
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？
"""
host_grammar = create_grammar(host,split='=')


#%%
for i in range(20):
    print(generate(host_grammar,target='host'))


#%%
programming = """
stmt => if_exp | while_exp | assignment 
assignment => var = var
if_exp => if ( var ) { /n .... stmt }
while_exp=> while ( var ) { /n .... stmt }
var => chars number
chars => char | char char
char => student | name | info  | database | course
number => 1 | 2 | 3
"""
program_gram = create_grammar(programming,split='=>')
generate(program_gram,'stmt')

#%% [markdown]
#  #### `jieba`分词

#%%
import pandas as pd 
file_name = '../datasource/sqlResult_1558435.csv'
df = pd.read_csv(file_name,encoding='gb18030')
articles = df['content'].tolist()

import re
def tocken(strs):
    return re.findall('\w+',strs)


#%%
import jieba 
from collections import Counter


#%%
jieba_cut_counter = Counter(jieba.cut(articles[0]))


#%%
articles_clean = [''.join(tocken(str(article))) for article in articles]


#%%
with open('article9k.txt','w') as f:
    for i in range(9000):
        f.write(articles_clean[i]+'\n')


#%%
def cut(strs): return list(jieba.cut(strs))
TOCKEN = [] 
for i, line in enumerate(open('article9k.txt')):
    TOCKEN +=  cut(line)


#%%
word_count = Counter(TOCKEN)


#%%
import matplotlib.pyplot as plt 
frequency = [f for k,f in word_count.most_common(100)]
plt.plot(list(range(100)),frequency)


#%%
def prob_1(word):
    return word_count[word] / len(TOCKEN)

TOCKEN_2_GRAM = [''.join(TOCKEN[i:i+2]) for i in range(len(TOCKEN)-1)]
word_count_2 = Counter(TOCKEN_2_GRAM)

def prob_2(word1,word2):
    if word1+word2 in word_count_2.keys(): return word_count_2[word1+word2]/ len(TOCKEN_2_GRAM)
    else: return 1/ len(TOCKEN_2_GRAM)
prob_2('我们','在')


#%%
def get_prob(sentence):
    words = cut(sentence)
    probs = 1
    for i in range(len(words)-1):
        word = words[i]
        n_word = words[i+1]
        probs *= prob_2(word,n_word)
    return probs
get_prob('我觉得不行')

#%% [markdown]
#  #### 2. 请回答以下问题
# 
#  回答以下问题，并将问题发送至 mqgao@kaikeba.com中：
#  ```
#      2.1. what do you want to acquire in this course？
#      2.2. what problems do you want to solve？
#      2.3. what’s the advantages you have to finish you goal?
#      2.4. what’s the disadvantages you need to overcome to finish you goal?
#      2.5. How will you plan to study in this course period?
#  ```
#%% [markdown]
#  #### 3. 如何提交
#  代码 + 此 jupyter 相关，提交至自己的 github 中(**所以请务必把GitHub按照班主任要求录入在Trello中**)；
#  第2问，请提交至mqgao@kaikeba.com邮箱。
#  #### 4. 作业截止时间
#  此次作业截止时间为 2019.7.6日
#%% [markdown]
#  #### 5. 完成以下问答和编程练习
#%% [markdown]
#  >
#%% [markdown]
#  ## 基础理论部分
#%% [markdown]
#  #### 0. Can you come up out 3 sceneraies which use AI methods?
#%% [markdown]
#  Ans: {
#  1. 自动驾驶
#  2. 人脸识别
#  3. 机器翻译}
#%% [markdown]
#  #### 1. How do we use Github; Why do we use Jupyter and Pycharm;
#%% [markdown]
#  Ans: {`github`是我们存放、获取课程资料以及提交课程作业的平台 ，`jupyter`和`pycharm`是我们运行、调试代码的IDE。`jupyter`
#  适合实验和展示我们开发思路以及实验效果。`pycharm`更适用于大型工程项目的构建。}
#%% [markdown]
#  #### 2. What's the Probability Model?
#%% [markdown]
#  Ans: 概率模型是指对词/句的出现的概率建模，通过统计词出现的频率来估计词的概率。使用概率来生成词/句。
#%% [markdown]
#  #### 3. Can you came up with some sceneraies at which we could use Probability Model?
#%% [markdown]
#  Ans: 生成句子时, 可以使用概率模型生成出现概率最高的句子. 在机器翻译问题中,也可以使用概率模型
#%% [markdown]
#  #### 4. Why do we use probability and what's the difficult points for programming based on parsing and pattern match?
#%% [markdown]
#  Ans: 由语法生成的句子一定符合语法, 但人类产生的语句不一定符合语法.
#  使用基于统计的概率模型,能让生成的句子更像是人话.
#%% [markdown]
#  #### 5. What's the Language Model;
#%% [markdown]
#  Ans: 语言模型是对生成句子的概率建模.
#%% [markdown]
#  #### 6. Can you came up with some sceneraies at which we could use Language Model?
#%% [markdown]
#  Ans: 语言模型可以用于生成概率模型. 用于生成语句.
#%% [markdown]
#  #### 7. What's the 1-gram language model;
#%% [markdown]
#  Ans:1-gram模型是将概率模型大幅简化, 某个词出现的条件概率,只和它上一个相连的词相关
#%% [markdown]
#  #### 8. What's the disadvantages and advantages of 1-gram language model;
#%% [markdown]
#  Ans: # 1-gram模型是将模型复杂度大大简化,计算速度加快. 但带来的问题是, 某个词出现的频率不仅是和上一个词相关, 也和前几个词都相关
#  这样模型的效果就不会很好
#%% [markdown]
#  #### 9. What't the 2-gram models;
#%% [markdown]
#  Ans: 2-gram模型与1-gram模型相似, 不同之处仅是在计算概率时考虑了词前两个相邻的词联合概率.
#%% [markdown]
#  ## 编程实践部分
#%% [markdown]
#  #### 1. 设计你自己的句子生成器
#%% [markdown]
#  如何生成句子是一个很经典的问题，从1940s开始，图灵提出机器智能的时候，就使用的是人类能不能流畅和计算机进行对话。和计算机对话的一个前提是，计算机能够生成语言。
# 
#  计算机如何能生成语言是一个经典但是又很复杂的问题。 我们课程上为大家介绍的是一种基于规则（Rule Based）的生成方法。该方法虽然提出的时间早，但是现在依然在很多地方能够大显身手。值得说明的是，现在很多很实用的算法，都是很久之前提出的，例如，二分查找提出与1940s, Dijstra算法提出于1960s 等等。
#%% [markdown]
#  在著名的电视剧，电影《西部世界》中，这些机器人们语言生成的方法就是使用的SyntaxTree生成语言的方法。
# 
#  >
#  >
# 
#  ![WstWorld](https://timgsa.baidu.com/timg?image&quality=80&size=b10000_10000&sec=1561818705&di=95ca9ff2ff37fcb88ae47b82c7079feb&src=http://s7.sinaimg.cn/mw690/006BKUGwzy75VK46FMi66&690)
# 
#  >
#  >
#%% [markdown]
#  在这一部分，需要各位同学首先定义自己的语言。 大家可以先想一个应用场景，然后在这个场景下，定义语法。例如：
# 
#  在西部世界里，一个”人类“的语言可以定义为：
#  ```

#%%
# human = """
# human = 自己 寻找 活动
# 自己 = 我 | 俺 | 我们
# 寻找 = 看看 | 找找 | 想找点
# 活动 = 乐子 | 玩的
# """
# ```
#
# 一个“接待员”的语言可以定义为
# ```
# host = """
# host = 寒暄 报数 询问 业务相关 结尾
# 报数 = 我是 数字 号 ,
# 数字 = 单个数字 | 数字 单个数字
# 单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
# 寒暄 = 称谓 打招呼 | 打招呼
# 称谓 = 人称 ,
# 人称 = 先生 | 女士 | 小朋友
# 打招呼 = 你好 | 您好
# 询问 = 请问你要 | 您需要
# 业务相关 = 玩玩 具体业务
# 玩玩 = 耍一耍 | 玩一玩
# 具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
# 结尾 = 吗？"""
#
# ```
#
#
#

#%% [markdown]
#  请定义你自己的语法:
#%% [markdown]
#  第一个语法：

#%%
jojo_meme = '''
sentence =>  meme1 | meme2 | meme3 
meme1 => 角色名 , 人称代词 动词* 名词 啦 !
meme2 => 人称代词 真是 形容词 副词 啦 !
meme3 => 欧拉* 木大*
角色名 => JOJO | DIO | 布加拉提
人称代词 => 你 | 我 | 他
动词* => 不 动词 | 动词
动词 => 做 | 当 | 变 | 是
名词 => 人 | 吸血鬼 | 替身
形容词 => high | low
副词 => 到不行 | 极点
欧拉* => 欧拉 | 欧拉 欧拉*
木大* => 木大 | 木大 木大*
'''

#%% [markdown]
#  第二个语法：

#%%
statement = '''
sentence => 代词 副词 是 让人 形容词 啦
代词 => 这 | 那 
副词 => 真 | 特别 | 尤其 | 非常 | 过于
形容词 => 难受 | 想哭 | 绝望 | high到不行 | 喜极而泣
'''

#%% [markdown]
# 

#%%
jojo_gram = create_grammar(jojo_meme)
print(generate(jojo_gram,'sentence'))
sentence_gram = create_grammar(statement)
print(generate(sentence_gram,'sentence'))

#%% [markdown]
#  TODO: 然后，定义一个函数，generate_n，将generate扩展，使其能够生成n个句子:

#%%
def generate_n(n=10):
    sentence = []
    if n == 0 :
        return sentence
    else:
        for i in range(n):
            sentence.append(generate(jojo_gram,'meme1'))
    return sentence 
generate_n(10)

#%% [markdown]
#  >
#%% [markdown]
#  #### 2. 使用新数据源完成语言模型的训练
#%% [markdown]
#  按照我们上文中定义的`prob_2`函数，我们更换一个文本数据源，获得新的Language Model:
# 
#  1. 下载文本数据集（你可以在以下数据集中任选一个，也可以两个都使用）
#      + 可选数据集1，保险行业问询对话集： https://github.com/Computing-Intelligence/insuranceqa-corpus-zh/raw/release/corpus/pool/train.txt.gz
#      + 可选数据集2：豆瓣评论数据集：https://github.com/Computing-Intelligence/datasource/raw/master/movie_comments.csv
#  2. 修改代码，获得新的**2-gram**语言模型
#      + 进行文本清洗，获得所有的纯文本
#      + 将这些文本进行切词
#      + 送入之前定义的语言模型中，判断文本的合理程度
#%% [markdown]
#  >

#%%
file_path = '../datasource/movie_comments.csv'
df = pd.read_csv(file_path)
comments = df['comment'].tolist()
comments_clean = [''.join(tocken(str(comment))) for comment in comments]


#%%
NEW_TOCKEN = [] 
for i in range(20000):
    NEW_TOCKEN += cut(comments_clean[i])


#%%
word_counter = Counter(NEW_TOCKEN)
def prob(word1,word2):
    if word1+word2 in word_counter.keys(): return word_counter[word1+word2] / len(NEW_TOCKEN)
    else: return 1/len(NEW_TOCKEN)
def sentence_prob(sentence):
    sentence_clean = ''.join(tocken(str(sentence)))
    words = cut(sentence_clean)
    probs = 1
    for i in range(len(words)-1):
        word = words[i]
        n_word = words[i+1]
        probs *= prob(word,n_word)
    return probs

#%% [markdown]
#  #### 3. 获得最优质的的语言
#%% [markdown]
#  当我们能够生成随机的语言并且能判断之后，我们就可以生成更加合理的语言了。请定义 generate_best 函数，该函数输入一个语法 + 语言模型，能够生成**n**个句子，并能选择一个最合理的句子:
# 
# 
#%% [markdown]
#  提示，要实现这个函数，你需要Python的sorted函数

#%%
sorted([1, 3, 5, 2])

#%% [markdown]
#  这个函数接受一个参数key，这个参数接受一个函数作为输入，例如

#%%
sorted([(2, 5), (1, 4), (5, 0), (4, 4)], key=lambda x: x[0])

#%% [markdown]
#  能够让list按照第0个元素进行排序.

#%%
sorted([(2, 5), (1, 4), (5, 0), (4, 4)], key=lambda x: x[1])

#%% [markdown]
#  能够让list按照第1个元素进行排序.

#%%
sorted([(2, 5), (1, 4), (5, 0), (4, 4)], key=lambda x: x[1], reverse=True)

#%% [markdown]
#  能够让list按照第1个元素进行排序, 但是是递减的顺序。
#%% [markdown]
#  >

#%%
def generate_best():  # you code here
    sentences = generate_n(20)
    sentence_list = [] 
    for i in sentences:
        sentence_list.append((i, sentence_prob(i)))
    return sorted(sentence_list,key=lambda x: x[1], reverse=True)
    
generate_best()

#%% [markdown]
#  好了，现在我们实现了自己的第一个AI模型，这个模型能够生成比较接近于人类的语言。
#%% [markdown]
#  >
#%% [markdown]
#  Q: 这个模型有什么问题？ 你准备如何提升？
#%% [markdown]
#  Ans:
#%% [markdown]
#  >
#%% [markdown]
#  ##### 以下内容为可选部分，对于绝大多数同学，能完成以上的项目已经很优秀了，下边的内容如果你还有精力可以试试，但不是必须的。
#%% [markdown]
#  #### 4. (Optional) 完成基于Pattern Match的语句问答
#  > 我们的GitHub仓库中，有一个assignment-01-optional-pattern-match，这个难度较大，感兴趣的同学可以挑战一下。
#%% [markdown]
# 
#  #### 5. (Optional) 完成阿兰图灵机器智能原始论文的阅读
#  1. 请阅读阿兰图灵关于机器智能的原始论文：https://github.com/Computing-Intelligence/References/blob/master/AI%20%26%20Machine%20Learning/Computer%20Machinery%20and%20Intelligence.pdf
#  2. 并按照GitHub仓库中的论文阅读模板，填写完毕后发送给我: mqgao@kaikeba.com 谢谢
#%% [markdown]
#  >
#%% [markdown]
#  各位同学，我们已经完成了自己的第一个AI模型，大家对人工智能可能已经有了一些感觉，人工智能的核心就是，我们如何设计一个模型、程序，在外部的输入变化的时候，我们的程序不变，依然能够解决问题。人工智能是一个很大的领域，目前大家所熟知的深度学习只是其中一小部分，之后也肯定会有更多的方法提出来，但是大家知道人工智能的目标，就知道了之后进步的方向。
#%% [markdown]
#  然后，希望大家对AI不要有恐惧感，这个并不难，大家加油！
#%% [markdown]
#  >
#%% [markdown]
#  ![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1561828422005&di=48d19c16afb6acc9180183a6116088ac&imgtype=0&src=http%3A%2F%2Fb-ssl.duitang.com%2Fuploads%2Fitem%2F201807%2F28%2F20180728150843_BECNF.thumb.224_0.jpeg)

