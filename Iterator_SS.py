import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
import timeit
import math
from functools import partial
from pyomo.environ import *

def Stochastic_Time_Dependent_Dijkstra(StartNode_Idx, EndNode_Idx, DJKs_Road_V0, Max_ETT=10**9, alpha=0.9):

  Node_ID          =DJKs_Road_V0["Node_ID"]

  StartNode_ID     =Node_ID[StartNode_Idx]
  EndNode_ID       =Node_ID[EndNode_Idx]

  Arc_ID           =DJKs_Road_V0["Arc_ID"]
  Arc_Idx          =DJKs_Road_V0["Arc_Idx"]

  #Using each hour of the time series as a possible scenario
  Arc_Travel_Time  =DJKs_Road_V0["Arc_Travel_Time"]
  TT_T0=np.sum(Arc_Travel_Time,axis=1)

  NScenarios=Arc_Travel_Time.shape[1]
  TT_T0=(np.sum(Arc_Travel_Time,axis=1)!=np.inf)
  Arc_Travel_Time=Arc_Travel_Time[TT_T0,:]
  Arc_ID=Arc_ID[TT_T0]
  Arc_Idx_Idx=Arc_Idx[TT_T0]
  Node_Idx=list(np.unique(Arc_Idx_Idx))

  Arc_Length       =DJKs_Road_V0["Arc_Length"]
  Idx_ArcsWith_i   =DJKs_Road_V0["Idx_ArcsWith_i"]

  ##
  ArcInForEachNode=[] #ArcInForEachNode
  ArcOutForEachNode=[] #ArcOutForEachNode

  count=0
  for i in Node_Idx:
      ArcInForEachNode.append(np.where(Arc_Idx_Idx[:,0]==i)[0])
      ArcOutForEachNode.append(np.where(Arc_Idx_Idx[:,1]==i)[0])

      if i==StartNode_Idx:
          StartNode_Idx_Idx=count
      if i==EndNode_Idx:
          EndNode_Idx_Idx=count

      count=count+1

  ArcZero=list(np.arange(len(Node_Idx)))
  ArcZero.remove(StartNode_Idx_Idx)
  ArcZero.remove(EndNode_Idx_Idx)

  ###### Set
  Model = ConcreteModel()
  Model.NumNodes= Set(initialize=Node_Idx)# Set of Nodes
  Model.ArcIdx = RangeSet(0,Arc_Idx_Idx.shape[0]-1)# Set of all arcs

  #### Variables
  Model.x =  Var(range(Arc_Idx_Idx.shape[0]), domain=Binary)#
  Model.c = Var(domain=NonNegativeReals)
  Model.z = Var(range(NScenarios),domain=NonNegativeReals)

  MeanETT=np.mean(Arc_Travel_Time,axis=1) # For each arc

  def objective_rule(Model):   

      ETT=summation(MeanETT,Model.x)
      CVaR=Model.c+1/((1-alpha)*NScenarios)*sum(Model.z[i] for i in range(NScenarios))
      Output=CVaR 

      return Output

  def CVaRZ(Model,i):
      return Model.z[i]>=summation(Arc_Travel_Time[:,i], Model.x)-Model.c

  Model.CVaRZ = Constraint(range(NScenarios), rule=CVaRZ)

  def FlowZero(Model,i):
      return (sum(-Model.x[arc] for arc in ArcOutForEachNode[i])+sum(Model.x[arc] for arc in ArcInForEachNode[i]))==0

  Model.FlowZero = Constraint(ArcZero, rule=FlowZero)

  Model.PositiveFlow = Constraint(expr=-sum(Model.x[arc] for arc in ArcOutForEachNode[StartNode_Idx_Idx])+sum(Model.x[arc] for arc in ArcInForEachNode[StartNode_Idx_Idx])==1)
  Model.NegativeFlow = Constraint(expr=-sum(Model.x[arc] for arc in ArcOutForEachNode[EndNode_Idx_Idx])+sum(Model.x[arc] for arc in ArcInForEachNode[EndNode_Idx_Idx])==-1)

  Model.MaxETT = Constraint(expr=summation(MeanETT,Model.x)<=Max_ETT)

  Model.OBJ = Objective(rule = objective_rule, sense=minimize)
  opt = SolverFactory("gurobi", solver_io="python")
  opt.options['max_iter'] = 5000
  results=opt.solve(Model, tee=True)
  Optimal_X=Model.x.get_values()
  Optimal_X=np.array([Optimal_X[i] for i in range(len(Arc_Idx_Idx))],dtype=int)
  IdxArcIn=np.where(Optimal_X)[0]
  ArcIdxUsed=Arc_Idx_Idx[IdxArcIn]
  Model.del_component(Model.MaxETT)  

  return ArcIdxUsed, Model, Arc_Idx_Idx, MeanETT

def PyomoChangeETT (Model, MeanETT, Max_ETT, Arc_Idx_Idx):

    Model.MaxETT = Constraint(expr=summation(MeanETT, Model.x)<=Max_ETT)
    opt = SolverFactory("gurobi", solver_io="python")
    opt.options['max_iter'] = 5000
    results=opt.solve(Model, tee=True)
    Optimal_X=Model.x.get_values()
    Optimal_X=np.array([Optimal_X[i] for i in range(len(Arc_Idx_Idx))],dtype=int)
    IdxArcIn=np.where(Optimal_X)[0]
    ArcIdxUsed=Arc_Idx_Idx[IdxArcIn]
    Model.del_component(Model.MaxETT)  
    return ArcIdxUsed, Model, Arc_Idx_Idx


def Compute_TravelTimes(ArcIdxUsed, DJKs_Road_V0, alpha):

  Arc_Travel_Time  =DJKs_Road_V0["Arc_Travel_Time"]
  TT_s=np.sum(Arc_Travel_Time[ArcIdxUsed,:],axis=0)

  NumScenarios=TT_s.shape[0]


  CutInt=np.maximum(int(math.ceil(NumScenarios*(1-alpha))),1)

  Expected_TT=np.mean(TT_s)
  CVaR=np.mean(np.sort(TT_s)[-CutInt:]) 

  return Expected_TT, CVaR


def GetGlobalOptimaRoute_Pyomo(ArcIdxUsed, StartNode_Idx, EndNode_Idx,  DJKs_Road_V0, alpha):
  Node_ID          =DJKs_Road_V0["Node_ID"]
  NodesIdx_In=ArcIdxUsed
  Route_idx=[]
  Route_idx.append(StartNode_Idx)
  CurrentNode=StartNode_Idx
  for i in range(len(NodesIdx_In)):
    
    Route_idx.append(NodesIdx_In[NodesIdx_In[:,0]==CurrentNode,1][0])
    CurrentNode=NodesIdx_In[NodesIdx_In[:,0]==CurrentNode,1][0]
    
  ArcIdxUsed=[np.where((ArcIdxUsed[i,0]==DJKs_Road_V0["Arc_Idx"][:,0]) * (ArcIdxUsed[i,1]==DJKs_Road_V0["Arc_Idx"][:,1]))[0][0] for i in range(len(ArcIdxUsed))]

  Route=Node_ID[Route_idx] #Convert of OpenSmap IDs
  ETT,CVaR=Compute_TravelTimes(ArcIdxUsed, DJKs_Road_V0, alpha)

  return Route, Route_idx,ETT,CVaR


def MinETT_Dijkstra(StartNode_Idx, EndNode_Idx, DJKs_Road_V0):

  Node_ID          =DJKs_Road_V0["Node_ID"]

  StartNode_ID     =Node_ID[StartNode_Idx]
  EndNode_ID       =Node_ID[EndNode_Idx]

    
  Arc_ID           =DJKs_Road_V0["Arc_ID"]
  Arc_Idx          =DJKs_Road_V0["Arc_Idx"]

  #Using each hour of the time series as a possible scenario
  Arc_Travel_Time  =DJKs_Road_V0["Arc_Travel_Time"]
  TT_T0=np.sum(Arc_Travel_Time,axis=1)

  NScenarios=Arc_Travel_Time.shape[1]
  TT_T0=(np.sum(Arc_Travel_Time,axis=1)!=np.inf)
  Arc_Travel_Time=Arc_Travel_Time[TT_T0,:]
  Arc_ID=Arc_ID[TT_T0]
  Arc_Idx_Idx=Arc_Idx[TT_T0]
  Node_Idx=list(np.unique(Arc_Idx_Idx))

  Arc_Length       =DJKs_Road_V0["Arc_Length"]
  Idx_ArcsWith_i   =DJKs_Road_V0["Idx_ArcsWith_i"]

  ##
  ArcInForEachNode=[] #ArcInForEachNode
  ArcOutForEachNode=[] #ArcOutForEachNode

  count=0
  for i in Node_Idx:
      ArcInForEachNode.append(np.where(Arc_Idx_Idx[:,0]==i)[0])
      ArcOutForEachNode.append(np.where(Arc_Idx_Idx[:,1]==i)[0])

      if i==StartNode_Idx:
          StartNode_Idx_Idx=count
      if i==EndNode_Idx:
          EndNode_Idx_Idx=count

      count=count+1

  ArcZero=list(np.arange(len(Node_Idx)))
  ArcZero.remove(StartNode_Idx_Idx)
  ArcZero.remove(EndNode_Idx_Idx)

  ###### Set
  Model = ConcreteModel()
  Model.NumNodes= Set(initialize=Node_Idx)# Set of Nodes
  Model.ArcIdx = RangeSet(0,Arc_Idx_Idx.shape[0]-1)# Set of all arcs

  #### Variables
  Model.x =  Var(range(Arc_Idx_Idx.shape[0]), domain=Binary)#
  Model.c = Var(domain=NonNegativeReals)
  Model.z = Var(range(NScenarios),domain=NonNegativeReals)

  MeanETT=np.mean(Arc_Travel_Time,axis=1) # For each arc

  def objective_rule(Model):   

      ETT=summation(MeanETT,Model.x)
      #CVaR=Model.c+1/((1-alpha)*NScenarios)*sum(Model.z[i] for i in range(NScenarios))
      Output=ETT 

      return Output


  def FlowZero(Model,i):
      return (sum(-Model.x[arc] for arc in ArcOutForEachNode[i])+sum(Model.x[arc] for arc in ArcInForEachNode[i]))==0

  Model.FlowZero = Constraint(ArcZero, rule=FlowZero)

  Model.PositiveFlow = Constraint(expr=-sum(Model.x[arc] for arc in ArcOutForEachNode[StartNode_Idx_Idx])+sum(Model.x[arc] for arc in ArcInForEachNode[StartNode_Idx_Idx])==1)
  Model.NegativeFlow = Constraint(expr=-sum(Model.x[arc] for arc in ArcOutForEachNode[EndNode_Idx_Idx])+sum(Model.x[arc] for arc in ArcInForEachNode[EndNode_Idx_Idx])==-1)

  Model.OBJ = Objective(rule = objective_rule, sense=minimize)
  opt = SolverFactory("gurobi", solver_io="python")
  opt.options['max_iter'] = 5000
  opt.options['threads'] = 4
  results=opt.solve(Model, tee=True)
  Optimal_X=Model.x.get_values()
  Optimal_X=np.array([Optimal_X[i] for i in range(len(Arc_Idx_Idx))],dtype=int)
  IdxArcIn=np.where(Optimal_X)[0]
  ArcIdxUsed=Arc_Idx_Idx[IdxArcIn]

  return ArcIdxUsed, Model, Arc_Idx_Idx, MeanETT

def RunSingleRouteTDD(StartNode_Idx, EndNode_Idx, DJKs_Road_V0, alpha):

  RouteSolutionsID, RouteSolutionsIdx, ETTSolutions, CVaRSolutions=[], [], [], []

  #Min ETT
  ArcIdxUsed, _, _, _=MinETT_Dijkstra(StartNode_Idx, EndNode_Idx, DJKs_Road_V0)
  RouteL, Route_idxL, ETTL, CVaRL=GetGlobalOptimaRoute_Pyomo(ArcIdxUsed, StartNode_Idx, EndNode_Idx,  DJKs_Road_V0, alpha)
  RouteSolutionsID.append(RouteL)
  RouteSolutionsIdx.append(Route_idxL)
  ETTSolutions.append(ETTL)
  CVaRSolutions.append(CVaRL)

  #Min CVaR
  ArcIdxUsed, Model, Arc_Idx_Idx, MeanETT=Stochastic_Time_Dependent_Dijkstra(StartNode_Idx, EndNode_Idx, DJKs_Road_V0, Max_ETT=10**9, alpha=alpha)
  Route, Route_idx, ETT_U, CVaR_U=GetGlobalOptimaRoute_Pyomo(ArcIdxUsed, StartNode_Idx, EndNode_Idx,  DJKs_Road_V0, alpha)

  if np.abs(ETTL-ETT_U)<1:
    return RouteSolutionsID, RouteSolutionsIdx, ETTSolutions, CVaRSolutions
  
  else:
    
    RouteSolutionsID.append(Route)
    RouteSolutionsIdx.append(Route_idx)
    ETTSolutions.append(ETT_U)
    CVaRSolutions.append(CVaR_U)    
    print ("Current CVaR: %f"%CVaRSolutions[-1])

    Delta=(ETT_U-ETTL)/5
    Max_ETT=ETT_U-Delta #New ETT
    for i in range(10):
      
      ArcIdxUsed, Model, Arc_Idx_Idx=PyomoChangeETT(Model, MeanETT, Max_ETT, Arc_Idx_Idx)
      RouteF, Route_idxF, ETTF, CVaRF=GetGlobalOptimaRoute_Pyomo(ArcIdxUsed, StartNode_Idx, EndNode_Idx,  DJKs_Road_V0, alpha)

      if ETTF-ETTL>=(ETT_U-ETTL)/10:
        RouteSolutionsID.append(RouteF)
        RouteSolutionsIdx.append(Route_idxF)
        ETTSolutions.append(ETTF)
        CVaRSolutions.append(CVaRF)

        Delta=(ETTF-ETTL)/5
        Max_ETT=ETTF-Delta
        
      else:
        SortOrder=np.argsort(ETTSolutions)
        ETTSolutions=np.array(ETTSolutions)
        CVaRSolutions=np.array(CVaRSolutions)
        ETTSolutions=ETTSolutions[SortOrder]
        CVaRSolutions=CVaRSolutions[SortOrder]

        RouteSolutionsID = [RouteSolutionsID[i] for i in SortOrder]
        RouteSolutionsIdx = [RouteSolutionsIdx[i] for i in SortOrder]
        return RouteSolutionsID, RouteSolutionsIdx, ETTSolutions, CVaRSolutions

    SortOrder=np.argsort(ETTSolutions)
    ETTSolutions=np.array(ETTSolutions)
    CVaRSolutions=np.array(CVaRSolutions)
    ETTSolutions=ETTSolutions[SortOrder]
    CVaRSolutions=CVaRSolutions[SortOrder]

    RouteSolutionsID = [RouteSolutionsID[i] for i in SortOrder]
    RouteSolutionsIdx = [RouteSolutionsIdx[i] for i in SortOrder]
    return RouteSolutionsID, RouteSolutionsIdx, ETTSolutions, CVaRSolutions
  
  #RouteSolutionsID, RouteSolutionsIdx, ETTSolutions, CVaRSolutions=RunSingleRouteTDD(StartNode_Idx, EndNode_Idx, DJKs_Road_V0, alpha)
  
def OneRun(DJKs_Road_V0, alpha,OD, Ith_run):
    NumAttempts=0
    
    np.random.seed(Ith_run)
    while 1:    
        
        #Randomly select source and destination

        #Randomly select source and destination
        Node_ID=DJKs_Road_V0["Node_ID"]
        NumberOfNodes=len(Node_ID)

        
        I=np.random.randint(len(OD),size=2)
    
        StartNode_Idx=OD[I[0]]
        EndNode_Idx=OD[I[1]]

        try:
            RouteSolutionsID, RouteSolutionsIdx, ETTSolutions, CVaRSolutions=RunSingleRouteTDD(StartNode_Idx, EndNode_Idx, DJKs_Road_V0, alpha)
                
            
            
            Results_FRoute={"RouteSolutionsID":RouteSolutionsID,
                            "RouteSolutionsIdx":RouteSolutionsIdx,
                            "ETTSolutions":ETTSolutions,
                            "CVaRSolutions":CVaRSolutions}
            
            return Results_FRoute
        
        except:
            return -1
            
        NumAttempts+=1
        Ith_run=Ith_run+1
        print("%d"%NumAttempts)
        if NumAttempts==1000:
            print("Something went wrong tried 1000 o>d with no sucess")
            break


####
PathData="./Data/"
#Using time series of flood as as synthetic flood scenario
Data=np.load(PathData+"DJKMatthew.npz",allow_pickle=True)
TimeDiscretizations=Data["TimeDiscretizations"]
D=np.load(PathData+"OD.npz")
OD=D["OD"]
DJKs_Road_V0=Data["DJKs_Road_Flood"].item()

alpha=0.9

Data=np.load(PathData+"DJKMatthew.npz",allow_pickle=True)
DJKs_Road_V0=Data["DJKs_Road_Flood"].item()
Arc_Travel_Time  =DJKs_Road_V0["Arc_Travel_Time"]
Arc_Idx          =DJKs_Road_V0["Arc_Idx"]

TT_T0=np.sum(Arc_Travel_Time,axis=1)
TT_T0=(np.sum(Arc_Travel_Time,axis=1)!=np.inf)
Arc_Idx_Idx=Arc_Idx[TT_T0]
Node_Idx=list(np.unique(Arc_Idx_Idx))

Function=partial(OneRun,DJKs_Road_V0,alpha,Node_Idx)