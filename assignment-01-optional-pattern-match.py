#%% [markdown]
# ### Assignment-01-附加题-极具挑战性的题目，基础一般的同学请把前边的做好就非常非常好了，这个是一个趣味题 😝
#%% [markdown]
# ## 基于模式匹配的对话机器人实现
#%% [markdown]
# ### Pattern Match
# 
# 机器能否实现对话，这个长久以来是衡量机器人是否具有智能的一个重要标志。 Alan Turing早在其文中就提出过一个测试机器智能程度的方法，该方法主要是考察人类是否能够通过对话内容区分对方是机器人还是真正的人类，如果人类无法区分，我们就称之为具有”智能“。而这个测试，后来被大家叫做”图灵测试“，之后也被翻拍成了一步著名电影，叫做《模拟游戏》。 
# 
# 
# 
# 既然图灵当年以此作为机器是否具备智能的标志，这项任务肯定是复杂的。自从 1960s 开始，诸多科学家就希望从各个方面来解决这个问题，直到如今，都只能解决一部分问题。 目前对话机器人的建立方法有很多，今天的作业中，我们为大家提供一共快速的基于模板的对话机器人配置方式。
# 
# 此次作业首先希望大家能够读懂这段程序的代码，其次，在此基于我们提供的代码，**能够把它改造成汉语版本，实现对话效果。**
#%% [markdown]
# ```
# Pattern: (我想要A)
# Response: (如果你有 A，对你意味着什么呢？)
# 
# Input: (我想要度假)
# Response: (如果你有度假，对你意味着什么呢？)
# ```
#%% [markdown]
# 为了实现模板的判断和定义，我们需要定义一个特殊的符号类型，这个符号类型就叫做"variable"， 这个"variable"用来表示是一个占位符。例如，定义一个目标: "I want X"， 我们可以表示成  "I want ?X", 意思就是?X是一个用来占位的符号。
# 
# 如果输入了"I want holiday"， 在这里 'holiday' 就是 '?X'

#%%
def is_variable(pat):
    return pat.startswith('?') and all(s.isalpha() for s in pat[1:])


#%%
def pat_match(pattern, saying):
    if is_variable(pattern[0]): return True
    else:
        if pattern[0] != saying[0]: return False
        else:
            return pat_match(pattern[1:], saying[1:])

#%% [markdown]
# ### 例如

#%%
pat_match('I want ?X'.split(), "I want holiday".split())


#%%
pat_match('I have dreamed a ?X'.split(), "I dreamed about dog".split())


#%%
pat_match('I dreamed about ?X'.split(), "I dreamed about dog".split())

#%% [markdown]
# ### 获得匹配的变量
#%% [markdown]
# 以上的函数能够判断两个 pattern 是不是相符，但是我们更加希望的是获得每个variable对应的是什么值。
# 
# 我们对程序做如下修改:

#%%
def pat_match(pattern, saying):
    if is_variable(pattern[0]):
        return pattern[0], saying[0]
    else:
        if pattern[0] != saying[0]: return False
        else:
            return pat_match(pattern[1:], saying[1:])


#%%
pattern = 'I want ?X'.split()
saying = "I want holiday".split()


#%%
pat_match(pattern, saying)


#%%
pat_match("?X equals ?X".split(), "2+2 equals 2+2".split())

#%% [markdown]
# 但是，如果我们的 Pattern 中具备两个变量，那么以上程序就不能解决了，我们可以对程序做如下修改: 
# 

#%%
def pat_match(pattern, saying):
    if not pattern or not saying: return []
    
    if is_variable(pattern[0]):
        return [(pattern[0], saying[0])] + pat_match(pattern[1:], saying[1:])
    else:
        if pattern[0] != saying[0]: return []
        else:
            return pat_match(pattern[1:], saying[1:])

#%% [markdown]
# 于是，我们可以获得： 

#%%
pat_match("?X greater than ?Y".split(), "3 greater than 2".split())

#%% [markdown]
# 如果我们知道了每个变量对应的是什么，那么我们就可以很方便的使用我们定义好的模板进行替换：
#%% [markdown]
# 为了方便接下来的替换工作，我们新建立两个函数，一个是把我们解析出来的结果变成一个 dictionary，一个是依据这个 dictionary 依照我们的定义的方式进行替换。

#%%
def pat_to_dict(patterns):
    return {k: v for k, v in patterns}


#%%
def subsitite(rule, parsed_rules):
    if not rule: return []
    
    return [parsed_rules.get(rule[0], rule[0])] + subsitite(rule[1:], parsed_rules)


#%%
got_patterns = pat_match("I want ?X".split(), "I want iPhone".split())


#%%
got_patterns


#%%
subsitite("What if you mean if you got a ?X".split(), pat_to_dict(got_patterns))

#%% [markdown]
# 为了将以上输出变成一句话，也很简单，我们使用 Python 的 join 方法即可： 

#%%
john_pat = pat_match('?P needs ?X'.split(), "John needs resting".split())


#%%
' '.join(subsitite("What if you mean if you got a ?X".split(), pat_to_dict(got_patterns)))


#%%
john_pat = pat_match('?P needs ?X'.split(), "John needs vacation".split())


#%%
subsitite("Why does ?P need ?X ?".split(), pat_to_dict(john_pat))


#%%
' '.join(subsitite("Why does ?P need ?X ?".split(), pat_to_dict(john_pat)))

#%% [markdown]
# 那么如果我们现在定义一些patterns，就可以实现基于模板的对话生成了:

#%%
defined_patterns = {
    "I need ?X": ["Image you will get ?X soon", "Why do you need ?X ?"], 
    "My ?X told me something": ["Talk about more about your ?X", "How do you think about your ?X ?"]
}


#%%
def get_response(saying, rules):
    """" please implement the code, to get the response as followings:
    
    >>> get_response('I need iPhone') 
    >>> Image you will get iPhone soon
    >>> get_response("My mother told me something")
    >>> Talk about more about your monther.
    """
    pass
    

#%% [markdown]
# ### Segment Match
#%% [markdown]
# 我们上边的这种形式，能够进行一些初级的对话了，但是我们的模式逐字逐句匹配的， "I need iPhone" 和 "I need ?X" 可以匹配，但是"I need an iPhone" 和 "I need ?X" 就不匹配了，那怎么办？ 
# 
# 为了解决这个问题，我们可以新建一个变量类型 "?\*X", 这种类型多了一个星号(\*),表示匹配多个
#%% [markdown]
# 首先，和前文类似，我们需要定义一个判断是不是匹配多个的variable

#%%
def is_pattern_segment(pattern):
    return pattern.startswith('?*') and all(a.isalpha() for a in pattern[2:])


#%%
is_pattern_segment('?*P')


#%%
from collections import defaultdict

#%% [markdown]
# 然后我们把之前的 ```pat_match```程序改写成如下， 主要是增加了 ``` is_pattern_segment ```的部分. 

#%%
fail = [True, None]

def pat_match_with_seg(pattern, saying):
    if not pattern or not saying: return []
    
    pat = pattern[0]
    
    if is_variable(pat):
        return [(pat, saying[0])] + pat_match_with_seg(pattern[1:], saying[1:])
    elif is_pattern_segment(pat):
        match, index = segment_match(pattern, saying)
        return [match] + pat_match_with_seg(pattern[1:], saying[index:])
    elif pat == saying[0]:
        return pat_match_with_seg(pattern[1:], saying[1:])
    else:
        return fail

#%% [markdown]
# 这段程序里比较重要的一个新函数是 ```segment_match```，这个函数输入是一个以 ```segment_pattern```开头的模式，尽最大可能进行，匹配到这个*边长*的变量对于的部分。

#%%
def segment_match(pattern, saying):
    seg_pat, rest = pattern[0], pattern[1:]
    seg_pat = seg_pat.replace('?*', '?')

    if not rest: return (seg_pat, saying), len(saying)    
    
    for i, token in enumerate(saying):
        if rest[0] == token and is_match(rest[1:], saying[(i + 1):]):
            return (seg_pat, saying[:i]), i
    
    return (seg_pat, saying), len(saying)

def is_match(rest, saying):
    if not rest and not saying:
        return True
    if not all(a.isalpha() for a in rest[0]):
        return True
    if rest[0] != saying[0]:
        return False
    return is_match(rest[1:], saying[1:])


#%%
segment_match('?*P is very good'.split(), "My dog and my cat is very good".split())

#%% [markdown]
# 现在，我们就可以做到以下的匹配模式了: 

#%%
pat_match_with_seg('?*P is very good and ?*X'.split(), "My dog is very good and my cat is very cute".split())


#%%


#%% [markdown]
# 如果我们继续定义一些模板，我们进行匹配，就能够进行更加复杂的问题了: 

#%%
response_pair = {
    'I need ?X': [
        "Why do you neeed ?X"
    ],
    "I dont like my ?X": ["What bad things did ?X do for you?"]
}


#%%
pat_match_with_seg('I need ?*X'.split(), 
                  "I need an iPhone".split())


#%%
subsitite("Why do you neeed ?X".split(), pat_to_dict(pat_match_with_seg('I need ?*X'.split(), 
                  "I need an iPhone".split())))

#%% [markdown]
#  我们会发现，pat_to_dict在这个场景下会有有一点小问题，没关系，修正一些: 

#%%
def pat_to_dict(patterns):
    return {k: ' '.join(v) if isinstance(v, list) else v for k, v in patterns}


#%%
subsitite("Why do you neeed ?X".split(), pat_to_dict(pat_match_with_seg('I need ?*X'.split(), 
                  "I need an iPhone".split())))

#%% [markdown]
# 如果我们定义这样的一个模板:

#%%
("?*X hello ?*Y", "Hi, how do you do")


#%%
subsitite("Hi, how do you do?".split(), pat_to_dict(pat_match_with_seg('?*X hello ?*Y'.split(), 
                  "I am mike, hello ".split())))

#%% [markdown]
# ### 现在是你的时间了

#%%
#我们给大家一些例子: 
    
rules = {
    "?*X hello ?*Y": ["Hi, how do you do?"],
    "I was ?*X": ["Were you really ?X ?", "I already knew you were ?X ."]
}

#%% [markdown]
# ### 问题1
#%% [markdown]
# 编写一个程序, ```get_response(saying, response_rules)```输入是一个字符串 + 我们定义的 rules，例如上边我们所写的 pattern， 输出是一个回答。
#%% [markdown]
# ### 问题2
#%% [markdown]
# 改写以上程序，将程序变成能够支持中文输入的模式。
# *提示*: 你可以需用用到 jieba 分词
#%% [markdown]
# ### 问题3
#%% [markdown]
# 多设计一些模式，让这个程序变得更好玩，多和大家交流，看看大家有什么好玩的模式
#%% [markdown]
# ### 问题4
#%% [markdown]
# 1. 这样的程序有什么优点？有什么缺点？你有什么可以改进的方法吗？ 
# 2. 什么是数据驱动？数据驱动在这个程序里如何体现？
# 3. 数据驱动与 AI 的关系是什么？ 
# 
#%% [markdown]
# 一些参考 pattern

#%%
rule_responses = {
    '?*x hello ?*y': ['How do you do', 'Please state your problem'],
    '?*x I want ?*y': ['what would it mean if you got ?y', 'Why do you want ?y', 'Suppose you got ?y soon'],
    '?*x if ?*y': ['Do you really think its likely that ?y', 'Do you wish that ?y', 'What do you think about ?y', 'Really-- if ?y'],
    '?*x no ?*y': ['why not?', 'You are being a negative', 'Are you saying \'No\' just to be negative?'],
    '?*x I was ?*y': ['Were you really', 'Perhaps I already knew you were ?y', 'Why do you tell me you were ?y now?'],
    '?*x I feel ?*y': ['Do you often feel ?y ?', 'What other feelings do you have?'],
    '?*x你好?*y': ['你好呀', '请告诉我你的问题'],
    '?*x我想?*y': ['你觉得?y有什么意义呢？', '为什么你想?y', '你可以想想你很快就可以?y了'],
    '?*x我想要?*y': ['?x想问你，你觉得?y有什么意义呢?', '为什么你想?y', '?x觉得... 你可以想想你很快就可以有?y了', '你看?x像?y不', '我看你就像?y'],
    '?*x喜欢?*y': ['喜欢?y的哪里？', '?y有什么好的呢？', '你想要?y吗？'],
    '?*x讨厌?*y': ['?y怎么会那么讨厌呢?', '讨厌?y的哪里？', '?y有什么不好呢？', '你不想要?y吗？'],
    '?*xAI?*y': ['你为什么要提AI的事情？', '你为什么觉得AI要解决你的问题？'],
    '?*x机器人?*y': ['你为什么要提机器人的事情？', '你为什么觉得机器人要解决你的问题？'],
    '?*x对不起?*y': ['不用道歉', '你为什么觉得你需要道歉呢?'],
    '?*x我记得?*y': ['你经常会想起这个吗？', '除了?y你还会想起什么吗？', '你为什么和我提起?y'],
    '?*x如果?*y': ['你真的觉得?y会发生吗？', '你希望?y吗?', '真的吗？如果?y的话', '关于?y你怎么想？'],
    '?*x我?*z梦见?*y':['真的吗? --- ?y', '你在醒着的时候，以前想象过?y吗？', '你以前梦见过?y吗'],
    '?*x妈妈?*y': ['你家里除了?y还有谁?', '嗯嗯，多说一点和你家里有关系的', '她对你影响很大吗？'],
    '?*x爸爸?*y': ['你家里除了?y还有谁?', '嗯嗯，多说一点和你家里有关系的', '他对你影响很大吗？', '每当你想起你爸爸的时候， 你还会想起其他的吗?'],
    '?*x我愿意?*y': ['我可以帮你?y吗？', '你可以解释一下，为什么想?y'],
    '?*x我很难过，因为?*y': ['我听到你这么说， 也很难过', '?y不应该让你这么难过的'],
    '?*x难过?*y': ['我听到你这么说， 也很难过',
                 '不应该让你这么难过的，你觉得你拥有什么，就会不难过?',
                 '你觉得事情变成什么样，你就不难过了?'],
    '?*x就像?*y': ['你觉得?x和?y有什么相似性？', '?x和?y真的有关系吗？', '怎么说？'],
    '?*x和?*y都?*z': ['你觉得?z有什么问题吗?', '?z会对你有什么影响呢?'],
    '?*x和?*y一样?*z': ['你觉得?z有什么问题吗?', '?z会对你有什么影响呢?'],
    '?*x我是?*y': ['真的吗？', '?x想告诉你，或许我早就知道你是?y', '你为什么现在才告诉我你是?y'],
    '?*x我是?*y吗': ['如果你是?y会怎么样呢？', '你觉得你是?y吗', '如果你是?y，那一位着什么?'],
    '?*x你是?*y吗':  ['你为什么会对我是不是?y感兴趣?', '那你希望我是?y吗', '你要是喜欢， 我就会是?y'],
    '?*x你是?*y' : ['为什么你觉得我是?y'],
    '?*x因为?*y' : ['?y是真正的原因吗？', '你觉得会有其他原因吗?'],
    '?*x我不能?*y': ['你或许现在就能?*y', '如果你能?*y,会怎样呢？'],
    '?*x我觉得?*y': ['你经常这样感觉吗？', '除了到这个，你还有什么其他的感觉吗？'],
    '?*x我?*y你?*z': ['其实很有可能我们互相?y'],
    '?*x你为什么不?*y': ['你自己为什么不?y', '你觉得我不会?y', '等我心情好了，我就?y'],
    '?*x好的?*y': ['好的', '你是一个很正能量的人'],
    '?*x嗯嗯?*y': ['好的', '你是一个很正能量的人'],
    '?*x不嘛?*y': ['为什么不？', '你有一点负能量', '你说 不，是想表达不想的意思吗？'],
    '?*x不要?*y': ['为什么不？', '你有一点负能量', '你说 不，是想表达不想的意思吗？'],
    '?*x有些人?*y': ['具体是哪些人呢?'],
    '?*x有的人?*y': ['具体是哪些人呢?'],
    '?*x某些人?*y': ['具体是哪些人呢?'],
    '?*x每个人?*y': ['我确定不是人人都是', '你能想到一点特殊情况吗？', '例如谁？', '你看到的其实只是一小部分人'],
    '?*x所有人?*y': ['我确定不是人人都是', '你能想到一点特殊情况吗？', '例如谁？', '你看到的其实只是一小部分人'],
    '?*x总是?*y': ['你能想到一些其他情况吗?', '例如什么时候?', '你具体是说哪一次？', '真的---总是吗？'],
    '?*x一直?*y': ['你能想到一些其他情况吗?', '例如什么时候?', '你具体是说哪一次？', '真的---总是吗？'],
    '?*x或许?*y': ['你看起来不太确定'],
    '?*x可能?*y': ['你看起来不太确定'],
    '?*x他们是?*y吗？': ['你觉得他们可能不是?y？'],
    '?*x': ['很有趣', '请继续', '我不太确定我很理解你说的, 能稍微详细解释一下吗?']
}


