import gc
gc.collect()
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
from sklearn.model_selection import train_test_split
import time
###function
def maternkernel(x,y,gamma):
    x=np.array(x)
    y=np.array(y)
    return (1+sqrt(3)*sp.linalg.norm(x-y)/gamma)*exp(-sqrt(3)*sp.linalg.norm(x-y)/gamma)
def minmaxkernel(x,y,gamma):
    aux=x
    auy=y
    x=np.array(x)
    y=np.array(y)
    if len(x.shape)==0:
        x=[aux]
        y=[auy]
    d=len(x)
    res=0
    for i in range(d):
        res=res+min(x[i],y[i])
    return res
def pinball(z,t):
    if t>1 or t<0:
        print("tau must be in [0,1] \n")
        t=float(input("try an other tau"))
    return(0.5*cp.abs(z)+(t-0.5)*z)
#testing the pinball loss function output
out=[]
for i in np.arange(-5,5,0.1):
    out.append(pinball(i,0.5))
#linear kernel
def linearkernel(x,y,gamma):
    x=np.array(x)
    y=np.array(y)
    return x.T*y+gamma
#laplacian kernel
def LaplaceKernel(x,y,gamma):
    x=np.array(x)
    y=np.array(y)
    return exp(-sp.linalg.norm(x-y)/gamma)
def SigmoidKernel(x,y,gamma):
    x=np.array(x)
    y=np.array(y)
    return np.tanh((1/N2)*x.T*y+gamma)
#gaussian kernel
def gausskernel(x,y,gamma):
    x=np.array(x)
    y=np.array(y)
    return np.exp((-gamma**(-0.5))*sp.linalg.norm(x-y)**2)
#gram matrix
def computeG(X,gamma):
    N2=len(X)
    G=np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            G[i,j]=LaplaceKernel(X[i],X[j],gamma)
    return G
#computing estimated functions
def get_fq(x,q,N,M,A,points,gamma):
    value1=0
    for n in range(N):
        value1+= A[n,q]*LaplaceKernel(data[n],x,gamma)
    value2=0
    for m in range(N,M+N):
        value2+= A[m,q]*LaplaceKernel(points[m-N],x,gamma)
    return(value1+value2)
#getting the cv score
def getperformance(X,Z,Y,An,Q,N,M,tau):
    res=0
    for q in range(Q):
        for n in range(len(Y)):
            res+=pinball(Y[n]-(get_fq(X[n],q,N,M,An,Z,gamma)+(b[q]).value),tau[q])
    return((1/N)*res.value)
def create_folds(X,k):
    return(KFold(n_splits=k).split(X))

#function to extract a sub matrix
def extractSubMatrix(
    matrix,
    rowStartIdx, rowEndIdx,
    colStartIdx, colEndIdx):

    result = [
        x[ colStartIdx : colEndIdx ]
        for x in matrix[ rowStartIdx : rowEndIdx ]
    ]


##Cross validation
#reading the data
df = pd.read_csv("C:/Users/malex/Desktop/scrm/code/SyntheticData.csv")
df.columns=["stock1","stock0","loss between time 1&2"]

y=df["loss between time 1&2"]
data=[]
for i in range(len(df["stock0"])):
        data.append([df["stock0"][i],df["stock1"][i]])

#creating folds and splitting test and train sets
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
data=X_train
y=y_train
foldX=[]
foldY=[]
y_train=np.array(y_train)
X_train=np.array(X_train)
y=[]
data=[]
for i in range(len(X_train)):
        data.append(X_train[i])
        y.append(y_train[i])
data=np.array(data)
y=np.array(y)
for i,j in create_folds(X_train,2):
    foldX.append(data[i].tolist())
    foldY.append(y[i].tolist())
distance=[]
for i in range(len(data)):
    for j in range(len(data)):
        distance.append(abs(data[i]-data[j]))

#go for 2 folds
data=[]
y=[]
data1=np.array(foldX[0])
y1=np.array(foldY[0])
datatest1=np.array(foldX[1])
ytest1=np.array(foldY[1])
data2=np.array(foldX[1])
y2=np.array(foldY[1])
datatest2=np.array(foldX[0])
ytest2=np.array(foldY[0])

DataX=[data1,data2]
DataY=[y1,y2]
TestX=[datatest1,datatest2]
TestY=[ytest1,ytest2]
lmdg_v=[91.125, 410.0625, 1845.28125,5000, 8303.765625]
gamma_v=[np.median(distance)]
b_v=[(10**log(i))*max(np.abs(df['loss between time 1&2'])) for i in [exp(1),3,4.2,6,exp(2)]]
perf=[]
performance=[]
lmdf=cp.Parameter()
values=[]
perf2=[]
X_test=np.array(X_test)
y_test=np.array(y_test)
#running the CV
start_time = time.time()
for gamma in gamma_v:
    print("s=",gamma)
    for lmdg in lmdg_v:
        for lmdb in b_v:
            lmd=lmdg

            for i in range(2):

                data=DataX[i]
                y=DataY[i]
                start_time2 = time.time()
                minX0=min(df["stock0"])
                maxX0=max(df["stock0"])
                minX1=min(df["stock1"])
                maxX1=max(df["stock1"])
                #delta net
                delta= 6
                points=[]
                #points=(np.arange(minX0,maxX0,delta)).tolist()
                for k in np.arange(minX0,maxX0+1,delta):
                    for j in np.arange(minX1,maxX1+1,delta):
                        points.append([k,j])


                data2=data
                data=[]
                for k in range(len(data2)):
                        data.append(data2[k])

                X=data+points
                #pinball loss function


                #computing the gram matrix
                G=computeG(X,gamma)
                Geps=G+(10**(-4)*np.ones((len(X),len(X))))
                #computing the eta coefficient

                eta=sqrt(2)*(1-exp(-sqrt(2*delta**2)/(gamma**2)))**(0.5)
                #computing the W and U matrices
                Q=5
                I=Q-1
                W=np.zeros((I,Q))
                j=-1
                for l in range(Q-1):
                    j=j+1
                    while j>=l:
                        W[l,j]=-1
                        W[l,j+1]=1
                        break
                U=W
                e=np.zeros((Q-1,Q-1))
                l,j=0,-1
                for l in range(Q-1):
                    j=j+1
                    while j>=l:
                        e[l,j]=1
                        break
                eq=np.zeros((Q,Q))
                l,j=0,-1
                for l in range(Q):
                    j=j+1
                    while j>=l:
                        eq[l,j]=1
                        break
                N=len(data)


                #optimization problem
                tau=[0.1,0.3,0.5,0.7,0.9]
                l=0
                q=0
                M=len(points)
                A=cp.Variable((M+N,Q))
                b=cp.Variable(Q)
                Gsqrt=sp.linalg.sqrtm(Geps)

                hi=((Geps@(A@W.T))[N:N+M])
                hj=(Geps@(A@W.T))
                soc_constraint=[(1/eta)*(U@b)[l]+(1/(eta))*cp.min((hi@e[l]))>=cp.norm((Gsqrt@hj)@e[l],2) for l in range(Q-1)]
                obj=0
                Gn=np.array(extractSubMatrix(G,0,N,0,N+M))
                y=np.array(y)
                for q in range(Q):
                    for n in range(N):
                        obj+=pinball(y[n]-((Gn@A)[n,q]+b[q]),tau[q])


                hl=(Gsqrt@A)
                f1=0
                for q in range(Q):
                    f1=f1+cp.norm(hl@eq[q],2)**2
                bn=cp.norm(b,2)
                prob = cp.Problem(cp.Minimize((1/N)*obj),soc_constraint+[bn<=lmdb]+[f1<=lmd])
                prob.solve(solver="MOSEK")
                end_time2=time.time()
                #print("prob value =",obj.value)
                perf.append(getperformance(TestX[i].tolist(),points,TestY[i],A.value,Q,N,M,tau))
            values.append((lmd/1000,lmdb))
        # print("prf value",np.mean(perf))
            performance.append(np.mean(perf))

            perf=[]

##for getting optimal hyperparemeters
val2=[]
for i in range(len(performance)):
    val2.append((values[i][0],values[i][1]/1000,performance[i]))
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
x, y, z = zip(*val2)
z = list(map(float, z))
grid_x, grid_y = np.mgrid[min(x):max(x):50j, min(y):max(y):50j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
np.where(grid_z.min()==grid_z.min())
#when running line 282 we get two index values say x and y, to get the optimal hyperparemeters we associate the following values: lmd=grid_x[x][0], lmdb=grid_y[0][y] and we re-run the optimization problem from line 178->263 after re-reading the data




#plotting the quantile function curves over a scatter plot of the data
plt.rc('legend',**{'fontsize':35})
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   :40}

plt.rc('font', **font)
colors=["r+","g+","y+","m+","c+"]
plt.plot(data,y,"k+")
seq0=np.arange(min(df["stock0"]),max(df["stock0"]),1)
seq1=np.arange(min(df["stock1"]),max(df["stock1"]),1)
seq=[]
for i in range(len(seq0)):
    for j in range(len(seq1)):
        seq.append((seq0[i],seq1[j],get_fq([seq0[i],seq1[j]],q,N,M,A.value,points,gamma)+b.value[q]))
q=0
seq2=[]
for i in range(len(seq0)):
    for j in range(len(seq1)):
        seq2.append((seq0[i],seq1[j],get_fq([seq0[i],seq1[j]],q,N,M,A.value,points,gamma)+b.value[q]))

q=3
seq3=[]
for i in range(len(seq0)):
    for j in range(len(seq1)):
        seq3.append((seq0[i],seq1[j],get_fq([seq0[i],seq1[j]],q,N,M,A.value,points,gamma)+b.value[q]))

for q in range(Q):
    fq=[]
    for i in seq:
        fq.append(get_fq(i,q,N,M,A.value,points,gamma)+b.value[q])
    plt.plot(seq,fq,label='$\\tau={}$'.format(tau[q]))
# plt.legend(loc=[1.01, 0.4])
plt.ylabel('$\\Psi(S_{t_1},S_{t_2},\\Theta)$')
plt.xlabel('$S_{t_1}$')
plt.show()



val2=[]
for i in range(len(performance)):
    val2.append((values[i][0],values[i][1]/1000,performance[i]))
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
x, y, z = zip(*seq)
z = list(map(float, z))
grid_x, grid_y = np.mgrid[min(x):max(x):50j, min(y):max(y):50j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

fig = plt.figure()
ax = fig.gca(projection='3d')
im=ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral,alpha=0.6)
ax.contour3D(grid_x, grid_y, grid_z, 50, cmap='binary')
fig.colorbar(im)
x2, y2, z2 = zip(*seq2)
z2 = list(map(float, z2))
grid_x2, grid_y2 = np.mgrid[min(x2):max(x2):50j, min(y2):max(y2):50j]
grid_z2 = griddata((x2, y2), z2, (grid_x2, grid_y2), method='cubic')
az = fig.gca(projection='3d')
im2=az.plot_surface(grid_x2, grid_y2, grid_z2, cmap=plt.cm.coolwarm,alpha=0.6)
az.contour3D(grid_x2, grid_y2, grid_z2, 50, cmap='binary')


fig.colorbar(im2)
x3, y3, z3 = zip(*seq3)
z3 = list(map(float, z3))
grid_x3, grid_y3 = np.mgrid[min(x3):max(x3):50j, min(y3):max(y3):50j]
grid_z3 = griddata((x3, y3), z3, (grid_x3, grid_y3), method='cubic')
aw = fig.gca(projection='3d')
im3=aw.plot_surface(grid_x3, grid_y3, grid_z3, cmap=plt.cm.binary,alpha=0.6)
az.contour3D(grid_x3, grid_y3, grid_z3, 50, cmap='binary')


fig.colorbar(im3)

y=df["loss between time 1&2"]
ay = fig.gca(projection='3d')
ay.scatter(df["stock0"], df["stock1"], y, c='r', marker='o')
ay.set_zlabel("$\\Psi(S_{t_1},S_{t_2},\\Theta)$")
ay.zaxis.labelpad = 30
ay.set_xlabel("$S_{t_1}^1$")
ay.xaxis.labelpad = 50
ay.set_ylabel("$S_{t_1}^2$")
ay.yaxis.labelpad = 50
ay.view_init(60, 35)
plt.show()









