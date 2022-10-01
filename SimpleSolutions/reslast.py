import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def resu(netw):
    os.chdir(netw+"_results")
    Q={}
    P={}
    A={}
    u={}
    c={}
    names=[]
    t=[]

    for i in range(len(os.listdir())):
        sol = os.listdir()
        if "Q.last" in sol[i].split("_"):
            Q[sol[i].split("_")[0]] = np.loadtxt(sol[i])
        if "P.last" in sol[i].split("_"):
            P[sol[i].split("_")[0]] = np.loadtxt(sol[i])/133.33 
        if "A.last" in sol[i].split("_"):
            A[sol[i].split("_")[0]] = np.loadtxt(sol[i])
        if "u.last" in sol[i].split("_"):
            u[sol[i].split("_")[0]] = np.loadtxt(sol[i])
        if "c.last" in sol[i].split("_"):
            c[sol[i].split("_")[0]] = np.loadtxt(sol[i])

    os.chdir("..")
    for i in Q:
        names.append(i)

    t=Q[names[0]][:,0]

    return Q,A,P,u,c,names,t


def resuout(netw):
    os.chdir(netw+"_results")
    Q={}
    P={}
    A={}
    u={}

    for i in range(len(os.listdir())):
        sol = os.listdir()
        if "Q.out" in sol[i].split("_"):
            Q[sol[i].split("_")[0]] = np.loadtxt(sol[i])
        if "P.out" in sol[i].split("_"):
            P[sol[i].split("_")[0]] = np.loadtxt(sol[i])/133.33 
        if "A.out" in sol[i].split("_"):
            A[sol[i].split("_")[0]] = np.loadtxt(sol[i])
        if "u.out" in sol[i].split("_"):
            u[sol[i].split("_")[0]] = np.loadtxt(sol[i])

    os.chdir("..")

    return Q,A,P,u



# IS0 keys
key0= ["11-L-int-carotidI", "18-L-int-carotidII", "23-L-MCA", "25-L-ACA-A1",  "29-L-ACA-A2", "31-ACoA"]

keyB= ["11-L-int-carotidI", "18-L-int-carotidII", "23-L-MCA", "25-L-ACA-A1",  "291-L-ACA-A2i", "31-ACoA"]

def pl(x,y,kx,ky,var):
    for i in range(len(kx)):
        if kx[i] in ky:
            plt.figure()
            plt.plot(x[kx[i]][:,3],label=kx[i])
            plt.plot(y[ky[i]][:,3],label=ky[i])
            plt.legend()
            plt.title(var)
























