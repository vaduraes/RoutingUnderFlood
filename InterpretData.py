import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import plotly_express as px

import networkx as nx
import osmnx as ox

File="AnalyzeIntegrality_P"+str(0)+".npz"
File_npz=np.load(File,allow_pickle=True)

#DataResults=File_npz['arr_0']
A_DataStart=File_npz['arr_1']
A_DataEnd=File_npz['arr_2']

File_npz.close()

HasIntegrality=[]
NumUnique=[]
CVAR_A=[]
ETT_A=[]
for i in range(1,35,1):
    try:
        File="AnalyzeIntegrality_P"+str(i)+".npz"
        File_npz=np.load(File,allow_pickle=True)
        
        DataResults=File_npz['arr_0']

        for j in range(DataResults.shape[0]):
            CVaR=[]
            ETT=[]
            for k in range(DataResults.shape[1]):
                HasIntegrality.append(DataResults[j,k,4])
                if DataResults[j,k,4]!=0 and DataResults[j,k,3][1]!=0:
                    print(File)
            
                ETT.append(np.unique(DataResults[j,k,1]))    
                CVaR.append(np.unique(DataResults[j,k,2]))
            
            CVaR=np.array(CVaR)
            ETT=np.array(ETT)
            CVaR=np.unique(CVaR)
            ETT=np.unique(ETT)
            CVAR_A.append(CVaR)
            ETT_A.append(ETT)
            NumUnique.append(len(CVaR))
        
        DataStart=File_npz['arr_1']
        DataEnd=File_npz['arr_2']
        A_DataStart=np.concatenate((A_DataStart,DataStart),axis=0)
        A_DataEnd=np.concatenate((A_DataEnd,DataEnd),axis=0)
        DataResults
        
        File_npz.close()
    except:
        print("Error")

Class=(A_DataStart.shape[0]*["Start"])+(A_DataEnd.shape[0]*["Destination"])
px.set_mapbox_access_token("pk.eyJ1IjoidmFkdXJhZXMiLCJhIjoiY2tpc2lodTQ4MGJoYjJ4cm85bHJnMGhldyJ9.eIUyVKXZDbenyuF4qhGUOQ")

X=np.concatenate((A_DataStart[:,1],A_DataEnd[:,1]))
Y=np.concatenate((A_DataStart[:,0],A_DataEnd[:,0]))

df = pd.DataFrame(list(zip(X,Y,Class)), 
               columns =["X_from", "Y_from", "Class"]) 

fig=px.scatter_mapbox(df, lon= "X_from", lat="Y_from", zoom=12,color="Class")

fig.write_html("LocationsInvestigated.htm")

#Percentage of integrality
HasIntegrality=np.array(HasIntegrality)
Percentage_I=sum(HasIntegrality<=0)/len(HasIntegrality)

#
C=[]
E=[]
for i in range(len(CVAR_A)):
    
    Idx=(CVAR_A[i]>0.1)*(ETT_A[i]>0.1)
    Idx=np.where(Idx)[0]
    for j in Idx:
        C.append(CVAR_A[i][j])
        E.append(ETT_A[i][j] )
    
NumUnique=np.array(NumUnique)

plt.boxplot([E,C],labels=["ETT","CVaR"])
plt.ylabel("Time [s]")




