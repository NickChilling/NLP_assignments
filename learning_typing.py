#%% [markdown]
# now , it's seen used in function 
# `def function(para1:type)->return_type:`

#%%
import typing
#%% [markdown]
# you can used alias to avoid redundant typing 
#%%
Vector = typing.List[float]
ConnectionOptions =typing.Dict[str,str]
Address = typing.Tuple[str,str]



#%% [markdown]
# you can use `NewType()` to create diff types


#%%
UserId = typing.NewType("UserId",int)
