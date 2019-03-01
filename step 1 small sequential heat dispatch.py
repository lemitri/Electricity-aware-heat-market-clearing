# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:26:59 2016

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




#%% heat marginal costs!!!!!!!!!!

spot_price_hierarchical_average_0=spot_price_hierarchical_average

alpha_heat_sequential = {}    

for W_max in W_range:
      
    for h in CHP_sorted['bp']:
        for t in time:
            alpha_heat_sequential[W_max,h,t] = alpha_scenario_kmeans['S_supply1',h,t]*(rho_heat[h] + rho_elec[h]*r_min[h])-spot_price_hierarchical_average_0[W_max,t]*r_min[h]
    
    
    for h in CHP_sorted['ex']:
        for t in time:
            if spot_price_hierarchical_average_0[W_max,t] < alpha_scenario_kmeans['S_supply1',h,t]*rho_elec[h]:
                alpha_heat_sequential[W_max,h,t] = alpha_scenario_kmeans['S_supply1',h,t]*(rho_heat[h] + rho_elec[h]*r_min[h])-spot_price_hierarchical_average_0[W_max,t]*r_min[h]
    
            else:
                alpha_heat_sequential[W_max,h,t] =  spot_price_hierarchical_average_0[W_max,t]*rho_heat[h]/rho_elec[h]
    
    for h in heat_pump:  
        for t in time:
            alpha_heat_sequential[W_max,h,t] = spot_price_hierarchical_average_0[W_max,t]/COP[h]
            
    for h in heat_only:  
        for t in time:
            alpha_heat_sequential[W_max,h,t] = alpha_scenario_kmeans['S_supply1',h,t]



#
##alpha_heat_sequential = {}
##    
##for W_max in W_range:
##      
##    for h in CHP_sorted['bp']:
##        for t in time:
##            alpha_heat_sequential[W_max,h,t] = alpha_scenario_kmeans['S_supply1',h,t]*(rho_heat[h] + rho_elec[h]*r_min[h])-np.random.normal(spot_price_CED_average[W_max,t],spot_price_CED_average[W_max,t]/10)*r_min[h]
##    
##    
##    for h in CHP_sorted['ex']:
##        for t in time:
##            p_estimate=np.random.normal(spot_price_CED_average[W_max,t],spot_price_CED_average[W_max,t]/10)
##            if p_estimate < alpha_scenario_kmeans['S_supply1',h,t]*rho_elec[h]:
##                alpha_heat_sequential[W_max,h,t] = alpha_scenario_kmeans['S_supply1',h,t]*(rho_heat[h] + rho_elec[h]*r_min[h])-p_estimate*r_min[h]
##    
##            else:
##                alpha_heat_sequential[W_max,h,t] =  p_estimate*rho_heat[h]/rho_elec[h]
##    
##    for h in heat_pump:  
##        for t in time:
##            alpha_heat_sequential[W_max,h,t] =  np.random.normal(spot_price_CED_average[W_max,t],spot_price_CED_average[W_max,t]/10)/COP[h]
##            
##    for h in heat_only:  
##        for t in time:
##            alpha_heat_sequential[W_max,h,t] = alpha_scenario_kmeans['S_supply1',h,t]
#



#for W_max in W_range: 
#
#    f, ax=plt.subplots()
#    
#    sb.despine(offset=1)
#           
#    for h in CHP:
#        ax.plot([x for x in range(24)],[alpha_heat_sequential[W_max,h,t] for t in time],c='red')
#    for h in heat_pump:
#        ax.plot([x for x in range(24)],[alpha_heat_sequential[W_max,h,t] for t in time],c='blue')   
#
#    ax.legend(bbox_to_anchor=(0.8,1.02),
#               bbox_transform=plt.gcf().transFigure,ncol=2,fontsize=11)

#%% heat dispatch

class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class sequential_heat:
    def __init__(self,W_max):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self._load_data(W_max)
        self._build_model()
    
    def optimize(self):
        self.model.optimize()
    
    def _load_data(self,W_max):
        
        #indexes
        self.data.time = time
        self.data.time_list=time_list
#        self.data.node=node
#        self.data.line=line
#        self.data.pipe=pipe
        #self.data.gen=gen
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
        #self.data.elec_load = {t:elec_load_mean[t] for t in time}  
        # Heat station parameters
        #self.data.CHP_maxprod = CHP_maxprod
        self.data.heat_maxprod = {(g,t):heat_maxprod[g] for g in CHP+heat_only+heat_pump for t in time} #BIDS
        #self.data.heat_maxprod = {(g,t):heat_maxprod[g] for g in CHP+heat_only+heat_pump for t in time}
        #self.data.rho_elec = rho_elec
        #self.data.rho_heat = rho_heat
        #self.data.r_min = r_min
        self.data.storage_discharge_eff= storage_rho_plus
        self.data.storage_charge_eff= storage_rho_moins
        self.data.storage_maxcapacity= storage_maxcapacity
        self.data.storage_maxprod= storage_maxprod
        self.data.storage_loss= storage_loss
        self.data.storage_energy_init= storage_init
        #self.data.COP = COP 
        
        # DHN parameters
#        self.data.pipe_maxflow = pipe_maxflow 
        
        # Elec station parameters
        # self.data.elec_maxprod = elec_maxprod
        # self.data.wind_scenario = {(w,t):wind_scenario_mean[w,t] for w in wind for t in time}
        
        # Cost parameters
        self.data.alpha = {(g,t):alpha_heat_sequential[W_max,g,t] for g in CHP+heat_only+heat_pump for t in time} #BIDS
        

        ##elec transmission system
        #self.data.B = B
        #self.data.line_maxflow = line_maxflow
   
    def _build_model(self):
        
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()
    
    def _build_variables(self):
        
        #indexes shortcuts 
        time = self.data.time
#        node=self.data.node
#        line=self.data.line
#        pipe=self.data.pipe
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
        
        #heat market optimization variables

        self.variables.Q = {} #heat production from CHPs and HO units (first satge)
        for t in time:
            for h in CHP+heat_only+heat_pump:
                self.variables.Q[h,t] = m.addVar(lb=0,ub=self.data.heat_maxprod[h,t],name='Q({0},{1})'.format(h,t))

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

#        # electricity transmission system variables 
#
#        self.variables.node_angle = {}
#        for t in time:
#            for n in node:
#                    self.variables.node_angle[n,t] = m.addVar(lb=-gb.GRB.INFINITY,name='node angle({0},{1})'.format(n,t))
#
#        self.variables.flow_line = {}
#        for t in time:
#            for l in line:
#                self.variables.flow_line[l,t] = m.addVar(lb=-self.data.line_maxflow[l],ub=self.data.line_maxflow[l],name='flow line({0},{1})'.format(l,t))                    
 
        m.update()
        
    def _build_objective(self): # building the objective function for the heat maret clearing
        
        #indexes shortcuts 
        time = self.data.time
#        node=self.data.node
#        line=self.data.line
#        pipe=self.data.pipe
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
             
        m.setObjective(
            gb.quicksum(self.variables.Q[h,t]*self.data.alpha[h,t] for h in CHP+heat_only+heat_pump for t in time),
            gb.GRB.MINIMIZE)
            
        
    def _build_constraints(self):
        

        #indexes shortcuts 
        time = self.data.time
#        node=self.data.node
#        line=self.data.line
#        pipe=self.data.pipe
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
        
        ##1) heat balance equation
        
        self.constraints.heat_balance = {}
        
        for t in time:
                
                self.constraints.heat_balance[t] = m.addConstr(
                    gb.quicksum(self.variables.storage_discharge[i,t]-self.variables.storage_charge[i,t] for i in heat_storage) + gb.quicksum(self.variables.Q[i,t] for i in CHP+heat_only+heat_pump),
                    gb.GRB.EQUAL,
                    self.data.heat_load[t],name='heat balance({0})'.format(t))    
                  
        # storage (1st stage)

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
                  
#%%

Q_sequential = {}
elec_maxprod_sequential = {}
elec_minprod_sequential = {}
        
for W_max in W_range:

    heat_dispatch = sequential_heat(W_max)                  
    heat_dispatch.model.params.OutputFlag = 0
    heat_dispatch.optimize()
    
    for t in time:
        
        for g in CHP+heat_only+heat_pump:
            Q_sequential[W_max,g,t]=heat_dispatch.variables.Q[g,t].x

        for g in heat_storage:
            Q_sequential[W_max,g,t]=heat_dispatch.variables.storage_discharge[g,t].x - heat_dispatch.variables.storage_charge[g,t].x   

    for t in time:
        
        for h in CHP_sorted['bp']:
            elec_maxprod_sequential[W_max,h,t] = 0
    
        for h in CHP_sorted['ex']:
            elec_maxprod_sequential[W_max,h,t] = (CHP_maxprod[h]-rho_heat[h]*Q_sequential[W_max,h,t])/rho_elec[h] - r_min[h]*Q_sequential[W_max,h,t]    

        for g in heat_pump:
            elec_maxprod_sequential[W_max,g,t] = Q_sequential[W_max,g,t]/COP[g]
            
        for g in gen:
            elec_maxprod_sequential[W_max,g,t] = elec_maxprod[g]   

        for g in wind:
            elec_maxprod_sequential[W_max,g,t] = W_max 
            
        for h in CHP:
            elec_minprod_sequential[W_max,h,t] = r_min[h]*Q_sequential[W_max,h,t]