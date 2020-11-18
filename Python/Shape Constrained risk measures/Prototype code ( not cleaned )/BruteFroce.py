import numpy as np
import pandas as pd
import scipy as sp
import sklearn as skl
from sklearn.model_selection import cross_val_score
from math import *
import random as rd
import cvxpy as cp
import multiprocessing as mp
import matplotlib.pyplot as plt
import gc
import statsmodels.api as sm
from sklearn.model_selection import KFold


df=pd.read_csv('C:/Users/malex/Desktop/scrm/code/SyntheticData.csv')
df.columns=["x","y"]
y=df['y']
X=np.arange(min(df['x']),max(df['x']),5)
temp=[]
Y=[]
Xn=[]
for i in range(len(X)-1):
    for j in range(len(y)):
        if X[i]<=df['x'][j]<=X[i+1]:
            temp.append(y[j])
    Xn.append((X[i]+X[i+1])/2)
    Y.append(temp)
    temp=[]
for i in range(len(Y)):
        if len(Y[i])==0:
            Xn.pop(i)

tau=[0.1,0.3,0.5,0.7,0.9]
q=[]
quantiles=[]
for t in tau:
    for i in range(len(Y)):
        if len(Y[i])!=0:
            Y[i].sort()
            idx=int(t*len(Y[i]))
            q.append(Y[i][idx])
    quantiles.append(q)
    q=[]

colors=["r+","g+","y+","m+","c+"]
df=pd.read_csv("C:/Users/malex/Desktop/scrm/code/SyntheticData.csv")
df.columns=["stock0","loss"]
plt.plot(df["stock0"],df["loss"],"k+")
for q in range(5):
    plt.plot(Xn,quantiles[q],label='$\\tau={}$'.format(tau[q]))
#plt.legend(loc=[1.01, 0.4])
plt.ylabel('$\\Psi(S_{t_1},S_{t_2},\\Theta)$')
plt.xlabel('$S_{t_1}$')
plt.show()

