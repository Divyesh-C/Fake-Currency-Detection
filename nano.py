import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import math
import matplotlib.ticker as tick
from matplotlib.ticker import FuncFormatter
import pandas as pd

# Identify chemical compositions bearing thermal stability in the nanocrystalline state.
def CalcIE(Xgb, ieo, Sig, Ga, Gb, DelH, DelE):
    j= float(Sig*(Ga-Gb)/6)
    k= float(DelH*(17*Xgb+0.5)/3)
    l= float(2*Xgb*(j-k-DelE)/Sig)
    ie= float(ieo + l)
    print(f"Interfacial energy={ie} when concentration= {Xgb} %")
    return ie



# Getting arrays of Interfacial energy and concentration
def IE(N):
    ienergy=[]
    t=[]
    for i in range(1,N):
        input(t(i))
        t.append(t(i))
        k = CalcIE(t(i),)
        ienergy.append(k)
    return t, ienergy

# Getting arrays of Gibbs Free energy and grain boundary fraction 
def DelG(M):
    DeltaGmix=[]
    q=[]
    for i in range(1,M):
        input(q(i))
        q.append(q(i))
        p = CalcFree(q(i),)
        DeltaGmix.append(p)
    return q, DeltaGmix

# Plotting graphs
def Plots2d(x, y):
    plt.plot(x, y, marker = 'o')
    plt.title("Plots")
    plt.xlabel("Concentration/ Grain Boundary Fraction")
    plt.ylabel("Interfacial Energy/ Gibbs Free energy mix")
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.3)
    #plt.show()

def Plots3d(x, y, z):
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax=plt.axes(projection='3d')
    #Axes3D.plot_surface(x, y, z)
    surf=ax.scatter(x, y, z)
    #surf=ax.plot_wireframe(x, y, z,  rstride=10, cstride=10)
    #surf=ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False, rstride=10, cstride=10)
    #ax.set_zlim(-1500, 100)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(tick.FuncFormatter(lambda x, _: '{:.02f}'.format(x)))
    #ax.zaxis.set_major_formatter('{x:.02f}')
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title("Energy Plot")
    ax.set_xlabel("Grain Size")
    ax.set_ylabel("Concentration")
    ax.set_zlabel("Gibbs Free Energy")
    ax.grid(color = 'green', linestyle = '--', linewidth = 0.3)
    plt.show()

# Function to identify Gibbs free energy of mixture varying with grain size.   
def CalcFree(z=12, v=0.5, delHmix=-5993, delHseg=17819, Oa=0.00000711, Ga=0.7, Ob=0.000014, Gb=0.28, t=0.5, k=8.314, T=300):
    X=[]
    D=[]
    delGM=[]
    for d in np.arange(2,101,1):
        for Xgb in np.arange(0.1,1,0.05):
                Fgb=1-(((d-t)/d)**3)
                X1=0.1
                Xc=(X1-Fgb*Xgb)/(1-Fgb)
                if Xc>0:
                    try:
                        Oc=delHmix/(z*X1*(1-X1))
                        Ogb=2*(Oc-delHseg/z)
                        DelGc=z*Oc*Xc*(1-Xc) + k*T*(Xc*(math.log(Xc))+(1-Xc)*(math.log(1-Xc)))
                        DelGgb=z*Ogb*Xgb*(1-Xgb) + Oa*Ga*(1-Xgb)/t + Ob*Gb*Xgb/t + k*T*(Xgb*(math.log(Xgb))+(1-Xgb)*(math.log(1-Xgb)))
                        DelGm= (1-Fgb)*DelGc + Fgb*DelGgb + z*v*Fgb*(Xgb-Xc) * ((2*Xgb-1)*Ogb - (Oa*Ga - Ob*Gb)/(z*t))
                        X.append(Xgb)
                        D.append(d)
                        delGM.append(DelGm)  
                    except ValueError:  # Log of a very small number.
                        Xc=0
                #Plots2d(delGM,X)
    #print(f"Gibbs Free energy mix={DelGm} when grain boundary fraction= {Fgb} %")
    x1=np.array(X)
    d1=np.array(D)
    delgm1=np.array(delGM)
    return delgm1, d1, x1

# Inputs
#N=input("Number of concentration values:")
#M=input("Number of grain boundary fraction values")
#x1, y1= IE(N)
#x2, y2= DelG(M)
#z=np.linspace(0,100,M)
#Plots2d(x1, y1)
U,T,V=CalcFree()
S=U.shape
s=S[0]
#print(s)
u=U.reshape(s,1)
t,v=np.meshgrid(T, V)
#print(T.shape)
#print(U.shape)
mymin = np.min(T)
min_positions = [i for i, x in enumerate(T) if x == mymin]
print("DelGmin =", U[min_positions])
print("D =", T[min_positions])
print("Xgb =", V[min_positions])
#print(T[min_positions], U[min_positions], V[min_positions]) 
#print(V.shape)
#Plots2d(T,V)
#Plots2d(U,V)
#Plots2d(T,U)
Plots3d(T,V,U)
