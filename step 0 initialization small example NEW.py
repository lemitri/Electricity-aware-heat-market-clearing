# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 08:57:30 2016

@author: lemitri
"""

#%%
            
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

os.chdir("C:/Users/lemitri/Documents/phd/Sequential and Hierarchical Heat and Electricity Markets/DATA")

def produit(list): #takes a list as argument
    p=1
    for x in list:
        p=p*x
    return(p)
 
#%% indexes
    
T=24
G=2 #elec generator
W=1 #wind
H=1 #chp
HO=1 #heat only
HS=1 #storage
ES=0
HP=1 #heat pump
#HES=2
#P=2
#L=7
#LL=500

S_load=1
S_wind=1
S_supply=1
S_all=S_load*S_wind*S_supply

time = ['t{0:02d}'.format(t+0) for t in range(T)]
#time_day =  ['t{0:02d}'.format(t) for t in range(24)] # optimization periods indexes (24 hours)
time_list=np.arange(T)
#node=['N{0}'.format(n+1)for n in range(N)]
#line=['L{0}'.format(l+1)for l in range(L)]
#pipe_supply=['PS{0}'.format(p+1)for p in range(P)]
#pipe_return=['PR{0}'.format(p+1)for p in range(P)]
gen=['G{0}'.format(g+1) for g in range(G)]
heat_storage=['HS{0}'.format(s+1) for s in range(HS)]
elec_storage=['ES{0}'.format(s+1) for s in range(ES)]
heat_pump = ['HP{0}'.format(h+1) for h in range(HP)]
wind=['W{0}'.format(w+1) for w in range(W)]
heat_only = ['HO{0}'.format(ho+1) for ho in range(HO)]
#heat_exchanger_station = ['HES{0}'.format(h+1) for h in range(HES)]
CHP_sorted = {'ex':['CHP1'],'bp':[]} # CHPs indexes sorted by type: extraction or backpressure
CHP = list(it.chain.from_iterable(CHP_sorted.values()))

# heat market data
heat_station=CHP+heat_only+heat_pump
elec_station=CHP+gen+wind+heat_pump
#producers=CHP+heat_only+heat_storage+heat_pump+gen+wind+elec_storage

scenario_load=['S_load{0}'.format(s+1) for s in range(S_load)]
scenario_wind=['S_wind{0}'.format(s+1) for s in range(S_wind)]
scenario_supply=['S_supply{0}'.format(s+1) for s in range(S_supply)]
scenario=['S{0}'.format(s+1) for s in range(S_all)]
scenario_list=[[scenario_load[s1],scenario_wind[s2],scenario_supply[s3]] for s1 in range(S_load) for s2 in range(S_wind) for s3 in range(S_supply)]
scenario_dic={s:x for (s,x) in zip(scenario,scenario_list)}

           
#%% loads
           
heat_load0 = {}

heat_load0['t23'] = 82.422360248447205
heat_load0['t00'] = 78.012422360248436
heat_load0['t01'] = 72.298136645962728
heat_load0['t02'] = 72.298136645962728
heat_load0['t03'] = 72.385093167701854
heat_load0['t04'] = 82.608695652173907+10
heat_load0['t05' ] =  83.354037267080734+10 
heat_load0['t06'] = 85.590062111801231+10
heat_load0['t07']=  86.086956521739125
heat_load0['t08']= 92.11180124223602
heat_load0['t09']= 71.987577639751549
heat_load0['t10']= 69.440993788819867
heat_load0['t11']= 66.459627329192543
heat_load0['t12']= 64.720496894409933
heat_load0['t13']= 63.354037267080734 
heat_load0['t14']= 62.11180124223602 
heat_load0['t15']= 61.987577639751549 
heat_load0['t16']= 64.968944099378874
heat_load0['t17']= 64.472049689440993
heat_load0['t18']= 66.583850931677006
heat_load0['t19']= 110.987577639751549
heat_load0['t20']= 97.440993788819867
heat_load0['t21']= 95.459627329192543 
heat_load0['t22']= 85.590062111801231

heat_load={t:heat_load0[t]*3 for t in time}

elec_load_m0 = {'t00':820,'t01':820,'t02':815,'t03':815,'t04':810,'t05':850,'t06':1000,'t07':1150+100*1400/350 ,'t08':1250+100*1400/350 ,'t09':1250+100*1400/350 ,'t10':1100+100*1400/350,'t11':1000+100*1400/350,'t12':1000+50*1400/350,'t13':955+50*1400/350,'t14':950+50*1400/350,'t15':950+36*1400/350,'t16':900+25*1400/350,'t17':950,'t18':1010,'t19':1100,'t20':1125,'t21':1025,'t22':950,'t23':850}

elec_load_m = {t:elec_load_m0[t]*350/1400 for t in time}

elec_load_v={t:0 for t in time}

elec_load_scenario = {}
for s in scenario_load:
    for t in time:
        elec_load_scenario[s,t]= sp.norm(elec_load_m[t],elec_load_v[t]).rvs()

#sp.norm(mean,variance).rvs()


elec_load_scenario_mean = {t:sum(elec_load_scenario[s,t] for s in scenario_load)/S_load for t in time}



#%% cost

alpha = {}
alpha['CHP1']=12.5
alpha['HO1']=30
alpha['HP1']=100
alpha['G1']= 12.5 #11    
alpha['G2']= 35 #33 
alpha['W1']=0.0000001
    
alpha_m = {}
alpha_v = {}

for t in time:
    alpha_m['CHP1',t]=alpha['CHP1']
    alpha_m['HO1',t]=alpha['HO1']
    alpha_m['HP1',t]=alpha['HP1']
    alpha_m['G1',t]= 12.5 #11
    alpha_m['G2',t]= 35 #33
    alpha_m['W1',t]=0.0000001

    alpha_v['CHP1',t]=0
    alpha_v['HO1',t]=0
    alpha_v['HP1',t]=0
    alpha_v['G1',t]=0
    alpha_v['G2',t]=0
    alpha_v['W1',t]=0
        
alpha_scenario = {}

for s in scenario_supply:
    for t in time:
        for g in CHP+heat_only+gen+wind+heat_pump:
            alpha_scenario[s,g,t]=sp.norm(alpha_m[g,t],alpha_v[g,t]).rvs()


alpha_scenario_mean = {(g,t):sum(alpha_scenario[s,g,t] for s in scenario_supply)/S_supply for g in CHP+heat_only+gen+wind+heat_pump for t in time}

alpha_up = {}
alpha_down = {}

for t in time:
    for s in scenario_supply:
        for g in CHP:
            alpha_up[s,g,t] = 1.05*alpha_scenario[s,g,t]
            alpha_down[s,g,t] = 0.95*alpha_scenario[s,g,t]
#            alpha_up[s,g,t] = 1.05*alpha_scenario[s,g,t]
#            alpha_down[s,g,t] = 0.95*alpha_scenario[s,g,t]
for t in time:
    for s in scenario_supply:
        for g in heat_only:
            alpha_up[s,g,t] = alpha_scenario[s,g,t]
            alpha_down[s,g,t] = alpha_scenario[s,g,t]
#            alpha_up[s,g,t] = 1.05*alpha_scenario[s,g,t]
#            alpha_down[s,g,t] = 0.95*alpha_scenario[s,g,t]
            
#alpha_up_mean = {(g,t):sum(alpha_up[s,g,t] for s in scenario_supply)/S_supply for g in CHP+heat_only for t in time}
#alpha_down_mean = {(g,t):sum(alpha_down[s,g,t] for s in scenario_supply)/S_supply for g in CHP+heat_only for t in time}
            
#%% technical characteristics
          
heat_maxprod = {'CHP1': 300,'HP1':200,'HO1':200} #HP 150
rho_elec = {'CHP1': 2.4} # efficiency of the CHP for electricity production
rho_heat = {'CHP1': 0.25,'HO1':1} # efficiency of the CHP for heat production
r_min = {'CHP1' : 0.6} # elec/heat ratio (flexible in the case of extraction units) 
CHP_maxprod = {'CHP1':600}

COP={'HP1':3}
COP_heat_pump = COP

storage_loss={h:1.5 for h in heat_storage+elec_storage}
storage_init={h:100 for h in heat_storage+elec_storage}
storage_rho_plus={h:1.1 for h in heat_storage+elec_storage} # >=1
storage_rho_moins={h:0.9 for h in heat_storage+elec_storage} # <=1
storage_maxcapacity={h:150 for h in heat_storage+elec_storage}
storage_maxprod = {h:50 for h in heat_storage+elec_storage}


elec_maxprod = {'CHP1':1000,'G1':150,'G2':200,'W1':500} # known

W_range=[500]


#%% wind scenarios


wind_scenario = {}

#for w in wind:
#    for t in range(T):
#        wind_scenario[w,time[t]] = pp[w,t+1,'V2'] 
                
wind_scenario['S_wind1','W1','t00'] = 0.4994795107848
wind_scenario['S_wind1','W1','t01'] = 0.494795107848
wind_scenario['S_wind1','W1','t02'] = 0.494795107848
wind_scenario['S_wind1','W1','t03'] = 0.505243011484
wind_scenario['S_wind1','W1','t04'] = 0.53537368424
wind_scenario['S_wind1','W1','t05'] = 0.555562455471
wind_scenario['S_wind1','W1','t06'] = 0.628348636916
wind_scenario['S_wind1','W1','t07'] = 0.6461954549
wind_scenario['S_wind1','W1','t08'] = 0.622400860956
wind_scenario['S_wind1','W1','t09'] = 0.580111023006
wind_scenario['S_wind1','W1','t10'] = 0.714935503018
wind_scenario['S_wind1','W1','t11'] = 0.824880140759
wind_scenario['S_wind1','W1','t12'] = 0.416551027874
wind_scenario['S_wind1','W1','t13'] = 0.418463919582
wind_scenario['S_wind1','W1','t14'] = 0.39525842857
wind_scenario['S_wind1','W1','t15'] = 0.523097379857
wind_scenario['S_wind1','W1','t16'] = 0.476699300008
wind_scenario['S_wind1','W1','t17'] = 0.626077589123
wind_scenario['S_wind1','W1','t18'] = 0.684294396661
wind_scenario['S_wind1','W1','t19'] = 0.0598119722706 
wind_scenario['S_wind1','W1','t20'] = 0.0446453658917 
wind_scenario['S_wind1','W1','t21'] = 0.485237701755
wind_scenario['S_wind1','W1','t22'] = 0.49466503395
wind_scenario['S_wind1','W1','t23'] = 0.4993958131342

wind_scenario_mean = {(w,t):sum(wind_scenario[s,w,t] for s in scenario_wind)/S_wind for w in wind for t in time}

elec_net_load_scenario={(s,t):elec_load_scenario[scenario_dic[s][0],t]-sum(wind_scenario[scenario_dic[s][1],w,t]*elec_maxprod[w]for w in wind) for s in scenario for t in time}


#%%

S_kmeans_0 = 1
S_all_kmeans_0 = 1

scenario_load_kmeans_0=scenario_load
scenario_wind_kmeans_0=scenario_wind
scenario_supply_kmeans_0=scenario_supply

scenario_kmeans_0=scenario
scenario_list_kmeans_0=scenario_list
scenario_dic_kmeans_0=scenario_dic   

elec_load_scenario_kmeans_0 = elec_load_scenario
wind_scenario_kmeans_0 = wind_scenario
alpha_scenario_kmeans_0 = alpha_scenario
alpha_up_kmeans_0 = {}
alpha_down_kmeans_0 = {}
for s in range(S_kmeans_0): 
    for t in range(T):
        for h in CHP:
            alpha_up_kmeans_0[scenario_supply_kmeans_0[s],h,time[t]]=alpha_scenario_kmeans_0[scenario_supply_kmeans_0[s],h,time[t]]*rho_heat[h]*1.1
            alpha_down_kmeans_0[scenario_supply_kmeans_0[s],h,time[t]]=alpha_scenario_kmeans_0[scenario_supply_kmeans_0[s],h,time[t]]*rho_heat[h]*0.9

        for h in heat_only:
            alpha_up_kmeans_0[scenario_supply_kmeans_0[s],h,time[t]]=alpha_scenario_kmeans_0[scenario_supply_kmeans_0[s],h,time[t]]*1.1
            alpha_down_kmeans_0[scenario_supply_kmeans_0[s],h,time[t]]=alpha_scenario_kmeans_0[scenario_supply_kmeans_0[s],h,time[t]]*0.9
            
 
S_kmeans = 1
S_all_kmeans = 1

          
scenario_load_kmeans=scenario_load
scenario_wind_kmeans=scenario_wind
scenario_supply_kmeans=scenario_supply

scenario_kmeans=scenario
scenario_list_kmeans=scenario_list
scenario_dic_kmeans=scenario_dic   

elec_load_scenario_kmeans = elec_load_scenario
wind_scenario_kmeans = wind_scenario
alpha_scenario_kmeans = alpha_scenario                    
alpha_up_kmeans = alpha_up_kmeans_0
alpha_down_kmeans = alpha_down_kmeans_0