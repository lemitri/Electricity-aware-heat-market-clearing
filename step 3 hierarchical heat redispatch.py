#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:43:46 2016

@author: lemitri
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:31:11 2016

@author: lemitri
"""

import os
import pandas as pd
import scipy.stats as sp
#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('ticks')

import gurobipy as gb
import itertools as it

import numpy as np


#
#alpha_up_kmeans = {}
#alpha_down_kmeans = {}
#
#for s in range(S_kmeans): 
#    for t in range(T):
#        for h in CHP:
#            alpha_up_kmeans[scenario_supply_kmeans[s],h,time[t]]=alpha_scenario_kmeans[scenario_supply_kmeans[s],h,time[t]]*rho_heat[h]*1.1
#            alpha_down_kmeans[scenario_supply_kmeans[s],h,time[t]]=alpha_scenario_kmeans[scenario_supply_kmeans[s],h,time[t]]*rho_heat[h]*0.9
#
#        for h in heat_only:
#            alpha_up_kmeans[scenario_supply_kmeans[s],h,time[t]]=alpha_scenario_kmeans[scenario_supply_kmeans[s],h,time[t]]*1.1
#            alpha_down_kmeans[scenario_supply_kmeans[s],h,time[t]]=alpha_scenario_kmeans[scenario_supply_kmeans[s],h,time[t]]*0.9
            
            
#%%
           
#alpha_R_hierarchical = {}
#    
#for W_max in [450]:    
#    for s in scenario_kmeans:
#      
#        for h in CHP_sorted['bp']:
#            for t in time:
#                alpha_R_hierarchical[W_max,s,h,t] = alpha_scenario_kmeans[scenario_dic_kmeans[s][2],h,t]*(rho_heat[h] + rho_elec[h]*r_min[h])-spot_price_hierarchical_scenario[W_max,s,t]*r_min[h]
#        
#        
#        for h in CHP_sorted['ex']:
#            for t in time:
#                if spot_price_hierarchical_scenario[W_max,s,t] < alpha_scenario_kmeans[scenario_dic_kmeans[s][2],h,t]*rho_elec[h]:
#                    alpha_R_hierarchical[W_max,s,h,t] = alpha_scenario_kmeans[scenario_dic_kmeans[s][2],h,t]*(rho_heat[h] + rho_elec[h]*r_min[h])-spot_price_hierarchical_scenario[W_max,s,t]*r_min[h]
#        
#                else:
#                    alpha_R_hierarchical[W_max,s,h,t] =  spot_price_hierarchical_scenario[W_max,s,t]*rho_heat[h]/rho_elec[h]
#        
#        for h in heat_pump:  
#            for t in time:
#                alpha_R_hierarchical[W_max,s,h,t] =  spot_price_hierarchical_scenario[W_max,s,t]/COP[h]
#                
#        for h in heat_only:  
#            for t in time:
#                alpha_R_hierarchical[W_max,s,h,t] = alpha_scenario_kmeans[scenario_dic_kmeans[s][2],h,t]


#%% building the STOCHSTICA MPEC optimization problem (GUROBI) = heat market clearing

class expando(object):
    '''
    
    
        A small class which can have attributes set
    '''
    pass

class hierarchical_heat_redispatch_scenario:
    def __init__(self,W_max,s0):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self._load_data(W_max,s0)
        self._build_model()

    
    def optimize(self):
        self.model.optimize()
    
    def _load_data(self,W_max,s0):



        #indexes
        self.data.time = time
        self.data.time_list=time_list
        #self.data.scenario = scenario
        #self.data.S = S
#        self.data.node=node
#        self.data.line=line
#        self.data.pipe=pipe
        self.data.gen=gen
        self.data.heat_storage=heat_storage
        #self.data.elec_storage=elec_storage
        self.data.heat_pump =heat_pump
        #self.data.wind=wind
        self.data.heat_only = heat_only
        self.data.CHP_sorted = CHP_sorted
        self.data.CHP = CHP
        #self.data.producers=producers
        #self.data.heat_station=heat_station
        #self.data.elec_station=elec_station
        #self.data.heat_exchanger_station = heat_exchanger_station        
        
#        #producers sorted per node
#        self.data.producers_node = producers_node
#        self.data.heat_station_node=heat_station_node
#        self.data.elec_station_node=elec_station_node
#        self.data.CHP_node = CHP_node
#        self.data.CHP_sorted_node= CHP_sorted_node
#        self.data.heat_pump_node = heat_pump_node
#        self.data.heat_only_node = heat_only_node
#        self.data.heat_storage_node= heat_storage_node
#        self.data.gen_node = gen_node
#        self.data.wind_node= wind_node
#        self.data.elec_storage_node = elec_storage_node
#        #self.data.heat_exchanger_station_node = heat_exchanger_station_node
        
#        # connexions between nodes
#        self.data.pipe_connexion= pipe_connexion
#        self.data.pipe_start = pipe_start
#        self.data.pipe_end = pipe_end
#        self.data.line_connexion= line_connexion
#        self.data.line_start = line_start
#        self.data.line_end = line_end
        
        # LOADS
        self.data.heat_load = heat_load
        #self.data.elec_load = {(s,t):elec_load_scenario[self.data.scenario_dic[s][0],t] for s in scenario for t in time}  
        
        # Heat station parameters
        self.data.CHP_maxprod = CHP_maxprod
        self.data.heat_maxprod = heat_maxprod
        self.data.rho_elec = rho_elec
        self.data.rho_heat = rho_heat
        self.data.r_min = r_min
        self.data.storage_discharge_eff= storage_rho_plus
        self.data.storage_charge_eff= storage_rho_moins
        self.data.storage_maxcapacity= storage_maxcapacity
        self.data.storage_maxprod= storage_maxprod
        self.data.storage_loss= storage_loss
        self.data.storage_energy_init= storage_init
        self.data.COP = COP 
        
        # Heat and electricity initial dispatch (DA)
        self.data.Q = {(h,t): Q_hierarchical[W_max,h,t] for h in CHP+heat_only for t in time}
        self.data.P = {(h,t):P_hierarchical_scenario[W_max,s0,h,t] for h in CHP+heat_pump for t in time}
        #self.data.spot_price = {t:spot_price_hierarchical_scenario[s0,t] for t in time}
        
        
        # DHN parameters
#        self.data.pipe_maxflow = pipe_maxflow 
        
        # Cost parameters
#        self.data.alpha_heat = {(g,t):alpha_R_hierarchical[W_max,s0,g,t] for g in CHP+heat_only for t in time}       

        self.data.alpha_up = {(g,t):alpha_up_kmeans[scenario_dic_kmeans[s0][2],g,t] for g in CHP+heat_only for t in time}       
        self.data.alpha_down = {(g,t):alpha_down_kmeans[scenario_dic_kmeans[s0][2],g,t] for g in CHP+heat_only for t in time}       

        ##elec transmission system
        #self.data.B = B
        #self.data.line_maxflow = line_maxflow

       
#        #default on initial DA heat dispatch : pnalties and premiums for redispatch
#        self.data.alpha_plus = alpha_plus
#        self.data.alpha_moins = alpha_moins

        #self.data.big_M = 1000
   
    def _build_model(self):
        
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
    
    def _build_variables(self):
        
        #indexes shortcuts 
        time = self.data.time
#        node=self.data.node
#        line=self.data.line
#        pipe=self.data.pipe
        #scenario = self.data.scenario
        #S = self.data.S
        #gen=self.data.gen
        heat_storage=self.data.heat_storage
        #elec_storage=self.data.elec_storage
        heat_pump=self.data.heat_pump
        #wind=self.data.wind
        heat_only=self.data.heat_only
        CHP_sorted=self.data.CHP_sorted
        CHP=self.data.CHP
        #producers=self.data.producers
        #heat_station=self.data.heat_station
        #elec_station=self.data.elec_station
        #heat_exchanger_station=self.data.heat_exchanger_station       
        m = self.model
        #big_M = self.data.big_M
        
        # heat market optimization variables


#        self.variables.flow_pipe = {}
#        for t in time:
#            for p in pipe:
#                self.variables.flow_pipe[p,t] = m.addVar(lb=0,ub=self.data.pipe_maxflow[p],name='flow pipe({0},{1})'.format(p,t))                    
 
        self.variables.storage_discharge = {} #heat storage: heat discharged (first stage)
        for t in time:
            for h in heat_storage:
                self.variables.storage_discharge[h,t] = m.addVar(lb=0,ub=self.data.storage_maxprod[h],name='storage discharge({0},{1})'.format(h,t))
                    
        self.variables.storage_charge = {} #heat storage: heat charged (first stage)
        for t in time:
            for h in heat_storage:
                self.variables.storage_charge[h,t] = m.addVar(lb=0,ub=self.data.storage_maxprod[h],name='storage charge({0},{1})'.format(h,t))
                    

        self.variables.storage_energy = {} #heat stored in heat storage h at end of time period t
        for t in time:
            for h in heat_storage:
                self.variables.storage_energy[h,t] = m.addVar(lb=0,ub=self.data.storage_maxcapacity[h],name='storage energy({0},{1})'.format(h,t))                

        self.variables.Q_up = {} #heat production from CHPs and HO units (first satge)
        for t in time:
            for h in CHP+heat_only:
                self.variables.Q_up[h,t] = m.addVar(lb=0,ub=self.data.heat_maxprod[h],name='Q down({0},{1})'.format(h,t))

        self.variables.Q_down = {} #heat production from CHPs and HO units (first satge)
        for t in time:
            for h in CHP+heat_only:
                self.variables.Q_down[h,t] = m.addVar(lb=0,ub=self.data.heat_maxprod[h],name='Q down({0},{1})'.format(h,t))

        m.update()
    
    def _build_objective(self): # building the objective function for the heat maret clearing

        #indexes shortcuts 
        time = self.data.time
#        node=self.data.node
#        line=self.data.line
#        pipe=self.data.pipe
        #scenario = self.data.scenario
        #S = self.data.S
        #gen=self.data.gen
        heat_storage=self.data.heat_storage
        #elec_storage=self.data.elec_storage
        heat_pump=self.data.heat_pump
        #wind=self.data.wind
        heat_only=self.data.heat_only
        CHP_sorted=self.data.CHP_sorted
        CHP=self.data.CHP
        #producers=self.data.producers
        #heat_station=self.data.heat_station
        #elec_station=self.data.elec_station
        #heat_exchanger_station=self.data.heat_exchanger_station       
        m = self.model      

        m.setObjective(gb.quicksum(self.data.alpha_up[h,t]*self.variables.Q_up[h,t] -self.data.alpha_down[h,t]*self.variables.Q_down[h,t] for h in CHP+heat_only for t in time),   
            gb.GRB.MINIMIZE)
            
#        m.setObjective(gb.quicksum(self.data.alpha_heat[h,t]*self.variables.Q_up[h,t] + 0.0001*self.variables.Q_down[h,t] for h in CHP+heat_only for t in time),   
#            gb.GRB.MINIMIZE) 
       
    def _build_constraints(self):

        #indexes shortcuts 
        time = self.data.time
#        node=self.data.node
#        line=self.data.line
#        pipe=self.data.pipe
        #scenario = self.data.scenario
        #S = self.data.S
        #gen=self.data.gen
        heat_storage=self.data.heat_storage
        #elec_storage=self.data.elec_storage
        heat_pump=self.data.heat_pump
        #wind=self.data.wind
        heat_only=self.data.heat_only
        CHP_sorted=self.data.CHP_sorted
        CHP=self.data.CHP
        #producers=self.data.producers
        #heat_station=self.data.heat_station
        #elec_station=self.data.elec_station
        #heat_exchanger_station=self.data.heat_exchanger_station       
        m = self.model
        #big_M = self.data.big_M


        # heat balance
 
        self.constraints.heat_balance = {} 
        
        for t in time:
            self.constraints.heat_balance[t] = m.addConstr(
                    gb.quicksum(self.data.Q[h,t] + self.variables.Q_up[h,t] - self.variables.Q_down[h,t] for h in CHP+heat_only)+gb.quicksum(self.data.P[h,t]*self.data.COP[h] for h in heat_pump)+gb.quicksum(self.variables.storage_discharge[h,t]-self.variables.storage_charge[h,t] for h in heat_storage),
                    gb.GRB.EQUAL,
                    self.data.heat_load[t],name='heat balance ({0})'.format(t))


        # heat storage: storage states update

        self.constraints.storage_update={}
        self.constraints.storage_init={}
        self.constraints.storage_final={}
        
        for h in heat_storage:
            
            for (t1,t2) in zip(time[:-1],time[1:]):
                self.constraints.storage_update[h,t2]=m.addConstr(
                    self.variables.storage_energy[h,t2],
                    gb.GRB.EQUAL,
                    self.variables.storage_energy[h,t1]-self.data.storage_discharge_eff[h]*self.variables.storage_discharge[h,t2]+self.data.storage_charge_eff[h]*self.variables.storage_charge[h,t2]-self.data.storage_loss[h])
        

            self.constraints.storage_init[h]=m.addConstr(
                self.variables.storage_energy[h,time[0]],
                gb.GRB.EQUAL,
                self.data.storage_energy_init[h]-self.data.storage_discharge_eff[h]*self.variables.storage_discharge[h,time[0]]+self.data.storage_charge_eff[h]*self.variables.storage_charge[h,time[0]]-self.data.storage_loss[h])


            self.constraints.storage_final[h]=m.addConstr(
                self.variables.storage_energy[h,time[-1]],
                gb.GRB.GREATER_EQUAL,
                self.data.storage_energy_init[h])


        #CHP's joint FOR in each scenario 
        
        self.constraints.heat_minprod = {} 

        for t in time:
            for h in CHP+heat_only:
                
                self.constraints.heat_minprod[h,t] = m.addConstr(
                    self.data.Q[h,t]+self.variables.Q_up[h,t]-self.variables.Q_down[h,t],
                    gb.GRB.GREATER_EQUAL,
                    0)

        self.constraints.heat_maxprod = {} 

        for t in time:
            for h in CHP+heat_only:
                
                self.constraints.heat_maxprod[h,t] = m.addConstr(
                    self.data.Q[h,t]+self.variables.Q_up[h,t]-self.variables.Q_down[h,t],
                    gb.GRB.LESS_EQUAL,
                    self.data.heat_maxprod[h])
                        
        self.constraints.CHP_maxprod = {} 
        self.constraints.CHP_ratio = {}
        
        for t in time:
            for h in CHP_sorted['ex']:
                
                self.constraints.CHP_maxprod[h,t] = m.addConstr(
                    self.data.rho_heat[h]*(self.data.Q[h,t]+self.variables.Q_up[h,t]-self.variables.Q_down[h,t])+self.data.rho_elec[h]*self.data.P[h,t],
                    gb.GRB.LESS_EQUAL,
                    self.data.CHP_maxprod[h])
                
                self.constraints.CHP_ratio[h,t] = m.addConstr(
                    self.data.P[h,t],
                    gb.GRB.GREATER_EQUAL,
                    self.data.r_min[h]*(self.data.Q[h,t]+self.variables.Q_up[h,t]-self.variables.Q_down[h,t]))

            for h in CHP_sorted['bp']:
                
                self.constraints.CHP_ratio[h,t] = m.addConstr(
                    self.data.P[h,t],
                    gb.GRB.EQUAL,
                    self.data.r_min[h]*(self.data.Q[h,t]+self.variables.Q_up[h,t]-self.variables.Q_down[h,t]))
                   

#%%

#syst_cost_hierarchical_scenario = {}
#heat_cost_hierarchical_scenario = {}
#heat_price_hierarchical_scenario = {}
#
#Q_hierarchical_scenario = {}
#Q_up_hierarchical_scenario = {}
#Q_down_hierarchical_scenario = {}
#E_hierarchical_scenario = {}
#
#Q_hierarchical_average = {}
#
#syst_cost_hierarchical_average = {}
#heat_cost_hierarchical_average = {}
#heat_price_hierarchical_average = {}

#heat_redispatch_cost_hierarchical_scenario = {}
#heat_redispatch_cost_hierarchical_average = {}

W_range = [200,250]

for W_max in W_range:   
    
    for s0 in scenario_kmeans:
        dispatch = hierarchical_heat_redispatch_scenario(W_max,s0)
        dispatch.model.params.OutputFlag = 0
        dispatch.optimize()
#        dispatch.model.computeIIS()
           
        for g in heat_only+CHP:
            for t in time:
                Q_hierarchical_scenario[W_max,s0,g,t] = Q_hierarchical[W_max,g,t] + dispatch.variables.Q_up[g,t].x -  dispatch.variables.Q_down[g,t].x
                Q_up_hierarchical_scenario[W_max,s0,g,t] = dispatch.variables.Q_up[g,t].x 
                Q_down_hierarchical_scenario[W_max,s0,g,t] = dispatch.variables.Q_down[g,t].x
                
        for g in heat_pump:
            for t in time:
                Q_hierarchical_scenario[W_max,s0,g,t] = P_hierarchical_scenario[W_max,s0,g,t]*COP[g]
                
        for g in heat_storage:
            for t in time:
                Q_hierarchical_scenario[W_max,s0,g,t]=dispatch.variables.storage_discharge[g,t].x - dispatch.variables.storage_charge[g,t].x     
                
            E_hierarchical_scenario[W_max,s0,g,time[0]] = storage_init[g]
            
            for (t1,t2) in zip(time[:-1],time[1:]):
                E_hierarchical_scenario[W_max,s0,g,t2]=  dispatch.variables.storage_energy[g,t1].x  
    
        for t in time:
            heat_price_hierarchical_scenario[W_max,s0,t] = dispatch.constraints.heat_balance[t].Pi

#%%  
W_range=[50,75,100,125,150,175,200,225,250,275,300]

for W_max in W_range:   
    
    for s0 in scenario_kmeans:
         
#        heat_cost_hierarchical_scenario[W_max,s0] = sum(Q_hierarchical_scenario[W_max,s0,g,t]*alpha_scenario_kmeans[scenario_dic_kmeans[s0][2],g,t] for g in heat_only for t in time) + sum((P_hierarchical_scenario[W_max,s0,g,t]*rho_elec[g]+Q_hierarchical_scenario[W_max,s0,g,t]*rho_heat[g])*alpha_scenario_kmeans[scenario_dic_kmeans[s0][2],g,t] - P_hierarchical_scenario[W_max,s0,g,t]*spot_price_hierarchical_scenario[W_max,s0,t] for g in CHP for t in time) + sum(Q_hierarchical_scenario[W_max,s0,g,t]/COP[g]*spot_price_hierarchical_scenario[W_max,s0,t] for g in heat_pump for t in time)
        heat_cost_hierarchical_scenario[W_max,s0] = sum(Q_hierarchical[W_max,g,t]*alpha_scenario_kmeans[scenario_dic_kmeans[s0][2],g,t] for g in heat_only for t in time) + sum((P_hierarchical_scenario[W_max,s0,g,t]*rho_elec[g]+Q_hierarchical[W_max,g,t]*rho_heat[g])*alpha_scenario_kmeans[scenario_dic_kmeans[s0][2],g,t] - P_hierarchical_scenario[W_max,s0,g,t]*spot_price_hierarchical_scenario[W_max,s0,t] for g in CHP for t in time) + sum(Q_hierarchical_scenario[W_max,s0,g,t]/COP[g]*spot_price_hierarchical_scenario[W_max,s0,t] for g in heat_pump for t in time)
        heat_redispatch_cost_hierarchical_scenario[W_max,s0] = sum(Q_up_hierarchical_scenario[W_max,s0,g,t]*alpha_up_kmeans[scenario_dic_kmeans[s0][2],g,t] - Q_down_hierarchical_scenario[W_max,s0,g,t]*alpha_down_kmeans[scenario_dic_kmeans[s0][2],g,t] for g in CHP+heat_only for t in time)
        syst_cost_hierarchical_scenario[W_max,s0] = sum(P_hierarchical_scenario[W_max,s0,g,t]*alpha_scenario_kmeans[scenario_dic_kmeans[s0][2],g,t] for g in gen for t in time) + sum(Q_hierarchical_scenario[W_max,s0,g,t]*alpha_scenario_kmeans[scenario_dic_kmeans[s0][2],g,t] for g in heat_only for t in time) + sum((P_hierarchical_scenario[W_max,s0,g,t]*rho_elec[g]+Q_hierarchical_scenario[W_max,s0,g,t]*rho_heat[g])*alpha_scenario_kmeans[scenario_dic_kmeans[s0][2],g,t] for g in CHP for t in time)
       
    for t in time:
        heat_price_hierarchical_average[W_max,t] = sum(heat_price_hierarchical_scenario[W_max,s,t] for s in scenario_kmeans)/S_all_kmeans
        for h in CHP+heat_only+heat_pump+heat_storage:
            Q_hierarchical_average[W_max,h,t] = sum(Q_hierarchical_scenario[W_max,s,h,t] for s in scenario_kmeans)/S_all_kmeans
    syst_cost_hierarchical_average[W_max] = sum(syst_cost_hierarchical_scenario[W_max,s] for s in scenario_kmeans)/S_all_kmeans
    heat_cost_hierarchical_average[W_max] = sum(heat_cost_hierarchical_scenario[W_max,s] for s in scenario_kmeans)/S_all_kmeans
    heat_redispatch_cost_hierarchical_average[W_max] = sum(heat_redispatch_cost_hierarchical_scenario[W_max,s] for s in scenario_kmeans)/S_all_kmeans
