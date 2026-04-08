'''
Distributive Integer Linear Programming for Mininmum Dominating Set Problem
Copyright (C) 2025  Rabea Mazen Saleh

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see
<https://www.gnu.org/licenses/>.
'''
#main code for DILP
import os
import networkx as nx
import numpy as np
from scipy.io import mmread
import pulp
from collections import deque
import time

files=[
#put the files names of the graphs that you want to run the algorithm on them here...
#supported graph file extensions: ".mtx" and ".edges"
#example:'tech-routers-rf.mtx',
#or:'rec-libimseti-dir.edges'
]

rootdir = #put the address of the folder containing the graphs, example:'/kaggle/input/network-repository-dilp/'
for file in files:
    path=''
    if(file[-3:]=='mtx'):
        path=rootdir+file[:-4]+'/'+file
        #G=nx.Graph(mmread(path))
    elif(file[-5:]=='edges'):
        path=rootdir+file[:-6]+'/'+file
        #G=nx.read_edgelist(path,comments='%',nodetype=np.int64)
    P=0.01 #choose the value of P (P is described in the paper)
    listoflenls=[]
    listofsizeMDS=[]
    listofcompare=[]
    listoftimes=[]
    listofmaxls=[]
    listofavgls=[]
    for seed in [1,2,3,4,5]: #set the random seeds
        print('seed',seed)
        np.random.seed(seed=seed)
        q=deque()
        color=np.zeros([len(G.nodes)],dtype=np.int64)
        visited=np.zeros([len(G.nodes)],dtype=np.int64)
        li=list(range(len(G.nodes)))
        for i in range(len(G.nodes)):
            li[i]=[]
        for i in G.nodes:
            if(np.random.uniform()<P):#uniform when not given arguments, will generate anumber from 0 to 1 
                visited[i]=1
                color[i]=i
                q.appendleft(i)
                li[i].append(i)
        while(len(q)>0):
            x=q.pop()
            for u in G[x]:
                if(visited[u]):
                    continue
                visited[u]=1
                color[u]=color[x]
                q.appendleft(u)
                li[color[x]].append(u)
        isMDS=np.zeros([len(G.nodes)],dtype=np.int64)
        MDSs=[]
        mxtime=0
        for l in li:
            if(len(l)==0):
                continue
            start_time=time.time()
            N=[]
            for i in l:
                N.append(i)
                for j in G[i]:
                    N.append(j)
            N=list(set(N))
            model=pulp.LpProblem("part", pulp.LpMinimize)
            x=pulp.LpVariable.dicts("x",N,cat=pulp.LpBinary)
            model+=(pulp.lpSum([x[i] for i in N]))
            for v in l:
                model+=(x[v]+pulp.lpSum([x[u] for u in list(G[v])]) >= 1)
            #model.solve(pulp.apis.PULP_CBC_CMD(msg=0))
            
            model.solve(pulp.apis.PULP_CBC_CMD(msg=0))
            #print("Status:", pulp.LpStatus[model.status])
            # Each of the variables is printed with it's resolved optimum value
            #for v in model.variables():
            #    print(v.name, "=", v.varValue)
            MDS=[i for i in N if x[i].value() == 1.0]
            for i in MDS:
                isMDS[i]=1
            MDSs.append(len(MDS))
            current=time.time()-start_time
            mxtime=max(mxtime,current)
    
        listoftimes.append(mxtime)
        ls=[len(l) for l in li if(len(l)!=0)]
        listofmaxls.append(max(ls))
        listofavgls.append(sum(ls)/len(ls))
        listoflenls.append(len(ls))
        coveredG=np.zeros([len(G.nodes)],dtype=np.int32)
        listofsizeMDS.append(np.sum(isMDS))
        MDS=[]
        for i in range(len(G.nodes)):
            if isMDS[i]:
                MDS.append(i)
        for i in MDS:
            coveredG[i]=1
            for j in G[i]:
                coveredG[j]=1
        listofcompare.append((len(G.nodes),np.sum(coveredG)))
    print(file)
    print('len(G.nodes)',len(G.nodes))
    print('len(G.edges)',len(G.edges))
    print('P=',P)
    print('average |MDS|= ',sum(listofsizeMDS)/len(listofsizeMDS))
    print('listofsizeMDS',listofsizeMDS)
    print('number of groups  ',listoflenls)
    print('times taken by every seed   ',listoftimes)
    print('average time=  ',sum(listoftimes)/len(listoftimes),"  seconds")
    print('listofcompare',listofcompare)
    print('maximum group size over 10 seeds=   ',max(listofmaxls))
    print('average group size over 10 seeds=   ',sum(listofavgls)/len(listofavgls))
    print('-------------------------------------------------------------------------------')
