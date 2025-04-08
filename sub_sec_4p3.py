#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np

from src import adaptive_coupling_study_funcs as study
from src.lib.model.ocp_graphite import Ueq as U0_ocp
from src.lib.reader import  reader as params
from src.lib.model import setup_LIB as setup 


# Input basic simulation and physical parameters from input file.

# In[15]:


inputFile = 'src/input/'           
phy_options, sim_options = params.readInputs(inputFile)

L_e = phy_options["Electrolyte"]["L_e"]["value"] # m
L_am = phy_options["Electrode"]["L_am"]["value"] # m
L_cc = phy_options["CurrentCollector"]["L_cc"]["value"] # m

D_e_c = phy_options['Electrolyte']['D_e_c']['value'] # m^2/s

L_c = L_e + L_am + L_cc # m
tc = (L_c ** 2) / D_e_c # s --> Timescale for dimensionless times


# Setup of the performance study

# In[16]:


cvType = ['constant', 'sinewave'][0] # Voltage operating condition

order_vec= np.array(range(2, 4 + 1)) # Vector of orders of coupling 

tini_dim = 0.0 # s Simulation start time
t_transient_dim = 1e-10 # The transient time in s for which an initial monolithic simulation is run with first-order backward Euler (BE) method to obtain consistent initial algebraic variables
nt_transient = 2 # steps taken by transient BE simulation
tend_dim = 501.0 # Time in s at which CV mode simulation ends

# Dimensionless times
tini = tini_dim/tc # Dimensionless time for CC mode start
t_transient = t_transient_dim/tc # Dimensionless time for CC mode end or CV mode simulation start
tend = tend_dim/tc # Dimensionless time for CV mode simulation end 

t_md_start = t_transient # Dimensionless time for the start of multi-domain coupled simulations
t_md_end = tend # Dimensionless time for the start of multi-domain coupled simulations

# Tolerances for md simulation with adaptive coupling
dt_rtol_vec = np.array([1e-7, 1e-6, 1e-5])

# Uncomment the following lines to reproduce study in paper 
# [<!> ATTENTION: This study might take few hours to run for each of the constant and sinewave voltage cases]
# order_vec= np.array(range(1, 4 + 1)) 
# rtolStart = 1e-6
# rtolEnd = 1e-10
# nSample = 5
# dt_rtol_vec = np.logspace(np.log10(rtolStart), np.log10(rtolEnd), nSample)
# dt_rtol_vec = np.unique(dt_rtol_vec)


# Setting up LIB simulation parameters based on studied operating conditions

# In[17]:


socI = 0.5 # start from 50% state of charge
c_s_max = phy_options["Electrode"]["c_s_max"]["value"] # mol/m^3

if cvType == 'constant':
    sim_options["ChargeRate"]["type"]["value"] = "CC"   
    sim_options["ChargeRate"]["xi"]["value"] = 1.0
    sim_options["ChargeRate"]["c_s_i"]["value"] = socI * c_s_max
    
    options_electrolyte, options_cathode = setup.getSetup(phy_options, sim_options, invertBV=False)
    Ui = np.r_[options_electrolyte['y0'], options_cathode['y0']] 
    tCC_end = 11./tc 
    from src import coupling_error_convergence_funcs as studyCC
    out_CC = studyCC.initial_CC_sim(
                        tini=tini,
                        t_CC_end=tCC_end,
                        y0_cc=Ui,
                        rtol=1e-9,
                        options_electrolyte=options_electrolyte,
                        options_cathode=options_cathode
    )
    tini = tCC_end
    t_md_start = tini + t_transient
    Umean = out_CC.y[-1, -1]*options_electrolyte['parameters']['phi_c']
    Ucell_t = lambda t: Umean
    Ui = out_CC.y[:, -1] # Initial condition
else:
    sim_options["ChargeRate"]["type"]["value"] = "CV"   
    sim_options["ChargeRate"]["xi"]["value"] = 1.0
    sim_options["ChargeRate"]["c_s_i"]["value"] = socI * c_s_max
    sim_options["ChargeRate"]["phi_s_L"]["value"] = lambda t : U0_ocp(socI)
    
    options_electrolyte, options_cathode = setup.getSetup(phy_options, sim_options, invertBV=False)
    
    Umean = U0_ocp(socI) # mean from which the voltage oscillates
    dev = 0.05 
    tspan = abs(t_md_end - t_md_start)
    osc = 3 # keep odd  
    tperiod = 2*np.pi
    omega = tperiod * osc
    
    signalU = lambda t: Umean*(1. - dev*np.sin(omega * t))
    # signalU = lambda t: Umean + dev * U0 * np.sin(omega * (t - t_md_start/tspan) )
    # assert Umean == Ucell_t(t_md_start)
    Ucell_t  = lambda t: signalU(t/tspan)
        
    Ui = np.r_[options_electrolyte['y0'], options_cathode['y0']] # Initial condition


# Initial monolithic simulation   

# In[18]:


options_cathode['sim_type']="monolithic"
options_electrolyte['sim_type']="monolithic"

options_cathode['parameters']['ChargeType'] = "CV"
options_cathode['parameters']['phi_s_L'] = Ucell_t
out_transient = study.initial_transient_sim(
                        tini=tini,
                        tend=tini + t_transient,
                        y0=Ui,
                        nt=nt_transient,
                        options_electrolyte=options_electrolyte,
                        options_cathode=options_cathode
)


# Quasi-exact reference monolithic solution

# In[19]:


options_cathode['sim_type']="monolithic"
options_electrolyte['sim_type']="monolithic"

options_cathode['parameters']['ChargeType'] = "CV"
options_cathode['parameters']['phi_s_L'] = Ucell_t

ref_sol_monolithic = study.get_ref_sol(
                      y0=out_transient.y[:,-1],
                      tstart=t_md_start,
                      tend=t_md_end,
                      rtol=1e-12,
                      options_electrolyte=options_electrolyte,
                      options_cathode=options_cathode
)


# Voltage evolution of the setup

# In[20]:


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' 
rcParams['text.usetex'] = True

time_vec = ref_sol_monolithic.t[:] * tc
Ucell_vec = ref_sol_monolithic.y[-1, :] * options_electrolyte['parameters']['phi_c']

plt.figure()

if (cvType == 'constant'):    
    plt.plot(out_CC.t[:-1]*tc, out_CC.y[-1, :-1] * options_electrolyte['parameters']['phi_c'],
    color='k', lw=2, ls='--', label=r"$\mathrm{Initial\ monolithic\ CC\ simulation}$")

plt.plot(time_vec, Ucell_vec,
         color='C0', lw=2, ls='-',
         label=r"$\mathrm{Coupled\ multi\text{-}domain\ simulation}$")
         
plt.xlabel(r"$t\ \mathrm{(s)}$", fontsize=15)
plt.ylabel(r"$U_{cell}\ \mathrm{(V)}$", fontsize=15)

if (cvType != 'constant'):
    plt.ylim(.93*Umean, 1.07*Umean)

plt.grid(ls=':')
plt.legend(framealpha=.75, fancybox=True,
           loc=0, numpoints=1, fontsize=10)
plt.tight_layout()


# Performance study of multi-domain LIB simulation with adaptive coupling

# In[21]:


options_cathode['parameters']['ChargeType'] = "CV"
options_cathode['parameters']['phi_s_L'] = Ucell_t

options_cathode['sim_type']="md_coupling_vars"
options_electrolyte['sim_type']="md_coupling_vars"

nparallel=8

# explicit coupling
sols_md_explicit = study.work_precision_loop(
                                        dt_rtol_vec,
                                        order_vec,
                                        t_md_start,
                                        t_md_end,
                                        y0_global=out_transient.y[:,-1],
                                        bExplicitCoupling=True,
                                        options_electrolyte=options_electrolyte,
                                        options_cathode=options_cathode,
                                        nparallel=nparallel
)
    
# implicit coupling
sols_md_implicit = study.work_precision_loop(
                                        dt_rtol_vec,
                                        order_vec,
                                        t_md_start,
                                        t_md_end,
                                        y0_global=out_transient.y[:,-1],
                                        bExplicitCoupling=False,
                                        options_electrolyte=options_electrolyte,
                                        options_cathode=options_cathode,
                                        nparallel=nparallel
)


# Get relative errors and CPUtimes for each of the simulations

# In[22]:


errors_vec_exp = study.get_errors(dt_rtol_vec, order_vec, sols_md_explicit, ref_sol_monolithic)
errors_vec_imp = study.get_errors(dt_rtol_vec, order_vec, sols_md_implicit, ref_sol_monolithic)

cpuTime_vec_exp = study.get_CPUtimes(dt_rtol_vec, order_vec, sols_md_explicit)
cpuTime_vec_imp = study.get_CPUtimes(dt_rtol_vec, order_vec, sols_md_implicit)


# Get non-NaN curves to be plotted

# In[23]:


error_plt_exp, cpuTime_plt_exp = study.get_work_precision_curves(order_vec, cpuTime_vec_exp, errors_vec_exp)
error_plt_imp, cpuTime_plt_imp = study.get_work_precision_curves(order_vec, cpuTime_vec_imp, errors_vec_imp)


# Plot parameters

# In[24]:


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' 
rcParams['text.usetex'] = True
plt.style.use = 'science'

clrmap = plt.get_cmap("tab10") 
clr =  [clrmap(i) for i in range(10)][order_vec[0]-1:]
malpha = 0.75
mfc = [(i[0], i[1], i[2], malpha) for i in clr]
mkr = ['o', 'v', 's', 'd'][order_vec[0]-1:]
fzA=15


# Work-precision diagrams ($\mathtt{CPUtime}$ vs. $\mathrm{error}$ plots)

# In[25]:


fig1 = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharey=ax1)

ax1.set_title(r"$\mathit{explicit{\text{-}}coupling}$", fontsize=fzA)
ax2.set_title(r"$\mathit{implicit{\text{-}}coupling}$", fontsize=fzA)

ax1.set_ylabel(r"$\mathtt{CPU} \mathrm{time\ (s)}$", fontsize=fzA)
ax1.set_xlabel(r"$\mathrm{error}$", fontsize=fzA)
ax2.set_xlabel(r"$\mathrm{error}$", fontsize=fzA)

for k, current_order in enumerate(order_vec):
    
    ax1.loglog(error_plt_exp[k], cpuTime_plt_exp[k], 
                   color=clr[k], marker=mkr[k], linestyle='-', mfc=mfc[k],
                   markersize=7, markeredgewidth=1,
                   label=f"$p = {current_order-1}$")
   
    ax2.loglog(error_plt_imp[k], cpuTime_plt_imp[k], label=None,
                    color=clr[k], marker=mkr[k], linestyle='-', mfc=mfc[k],
                    markersize=7, markeredgewidth=1)
    
ax1.grid(ls=':')
ax2.grid(ls=':')
ax1.legend(loc=0, framealpha=0.75, ncol=1, numpoints=1, fontsize=fzA-5)
plt.tight_layout()   


# Save results 

# In[ ]:


bSavefig = True        
if bSavefig:
    savepath = 'output/'    
    fig1.set_dpi(600)
    fig1.savefig(savepath+f"fig_WorkPrecision_{cvType}CV_tfin{int(tend_dim)}_pmax{order_vec[-1]}_rtolmin{min(dt_rtol_vec):.0E}.pdf", dpi=600)   

