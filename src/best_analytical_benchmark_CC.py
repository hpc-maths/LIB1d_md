# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:24:38 2022


@author: ali.asad
"""

"""
The code simulates the mass and charge conservation equations
in the electrolyte and electrode domain with current collector. 

The boundary conditions here are satisfied by
the BV current density flux evaluated using
the BV model with auxiliray variables present
at the interfaces (anode and cathode).
"""
##############################################IMPORTING MODULES#####################################################################################
import matplotlib.pyplot as plt
import numpy as np

# customs modules
from src.lib.reader import  reader as params
from src.lib.model import setup_LIB as setup 

from src.lib.model import _1d_lib_dfv_model as coupled_model

#%%
# Simulation Setup
#%%
bSavefigs = True
# BEsolver = "no-simulation"
# BEsolver = "implicit-euler"
BEsolver = "Radau"
inputFile = 'input/'           
phy_options, sim_options = params.readInputs(inputFile)

phy_options['Electrolyte']['del_e']['value'] = 0.

sim_options["ChargeRate"]["type"]["value"] = "CC"
sim_options["ChargeRate"]["xi"]["value"] = 0.5
sim_options["ChargeRate"]["c_s_i"]["value"] = 13000.0
# sim_options["ChargeRate"]["phi_s_L"]["value"] = 0.3

#dx altering
sim_options["Mesh"]["ne"]["value"]=100
sim_options["Mesh"]["nam"]["value"]=50
sim_options["Mesh"]["ncc"]["value"]=50

options_electrolyte, options_cathode = setup.getSetup(phy_options, sim_options, invertBV=False)
Ui = np.r_[options_electrolyte['y0'], options_cathode['y0']]

tini = 0. 
# t_cc_cv = 2458./options_electrolyte['parameters']['t_c']
tend = 2000./options_electrolyte['parameters']['t_c']

nt_cc = 51
nt_cv = 100

# simulation solution
#%%
ne=100
ns=100
Ui_CC = Ui 
# Ui[3:2*ne+2:2] = 0.
# Ui[2*ne+7::2] = 0. 
Mi = np.diag(np.r_[options_electrolyte['mass'], options_cathode['mass']])

tini_CC = tini
tend_CC = tend

options_cathode['sim_type']="monolithic"
options_electrolyte['sim_type']="monolithic"

if (BEsolver=="Radau"):
    # RADAU solution setup
    from scipy.integrate import solve_ivp
    from src.lib.solver.radau_dae import RadauDAE
    import scipy.sparse as sparse
    
    method=RadauDAE
    atol = 1e-10
    rtol = 1e-9
    bPrint=False
    
    # sparsity pattern
    uband=6; lband=-uband
    offsets = [i for i in range(lband,uband)]
    sparsity_pattern = sparse.diags(diagonals=[np.ones((Ui.size - abs(i))) for i in offsets], offsets=offsets)
    
    from src.lib.solver import be_dae_solver as be_solver
    sol_CC = solve_ivp(fun = lambda t, u:coupled_model.func_U(u, t, options_electrolyte, options_cathode),
                        t_span=(tini_CC, tend_CC),
                        y0=Ui_CC,
                        max_step=np.inf,
                        rtol=rtol, atol=atol,
                        jac=None, jac_sparsity=sparsity_pattern,
                        method=method, vectorized=False, first_step=1e-8, dense_output=True,
                        mass=Mi, bPrint=bPrint)  
    print('CC mode solution computed')      
    print('  --> "{}"'.format(sol_CC.message))
    if not sol_CC.success:
        raise Exception('CC mode solution failed')
        
    # Ui_CV = sol_CC.y[:,-1]
    # options_cathode['parameters']['ChargeType'] = "CV"
    # options_cathode['parameters']['phi_s_L'] = Ui_CV[-1]*options_electrolyte['parameters']['phi_c'] # should be in Volts
    
    # sol_CV = solve_ivp(fun = lambda t, u:coupled_model.func_U(u, t, options_electrolyte, options_cathode),
    #                     t_span=(tini_CV, tend_CV),
    #                     y0=Ui_CV,
    #                     max_step=np.inf,
    #                     rtol=rtol, atol=atol,
    #                     jac=None, jac_sparsity=sparsity_pattern,
    #                     method=method, vectorized=False, first_step=1e-8, dense_output=True,
    #                     mass=Mi, bPrint=bPrint)  
    # print('CV mode solution computed')      
    # print('  --> "{}"'.format(sol_CV.message))
    # if not sol_CV.success:
    #     raise Exception('CV mode solution failed')
elif (BEsolver=="implicit-euler"):
    
    from src.lib.solver import be_dae_solver as be_solver
    sol_CC = be_solver.integrate_dae(mass = Mi,
                                    tini = tini_CC,
                                    tend = tend_CC,
                                    nt = nt_cc,
                                    yini = Ui_CC,
                                    fcn = lambda x,t: coupled_model.func_U(x, t, options_electrolyte, options_cathode),
                                    options_electrolyte = options_electrolyte,
                                    options_cathode = options_cathode,
                                    atol=1e-6, rtol=1e-5,
                                    verbose = False,
                                    verbose_freq = 1)
    
    
    # Ui_CV = sol_CC.y[:,-1]
    # options_cathode['parameters']['ChargeType'] = "CV"
    # options_cathode['parameters']['phi_s_L'] = Ui_CV[-1]*options_electrolyte['parameters']['phi_c'] # should be in Volts
    # sol_CV = be_solver.integrate_dae(mass = Mi,
    #                                 tini = tini_CV,
    #                                 tend = tend_CV,
    #                                 nt = nt_cv,
    #                                 yini = Ui_CV,
    #                                 fcn = lambda x,t: coupled_model.func_U(x, t, options_electrolyte, options_cathode),
    #                                 options_electrolyte = options_electrolyte,
    #                                 options_cathode = options_cathode,
    #                                 atol=1e-6, rtol=1e-5,
    #                                 verbose = False,
    #                                 verbose_freq = 1) 
else :
        raise Exception('Not performing any simulations')
    
#%%

# compile results
#%%
tc = options_electrolyte['parameters']['t_c']
phic = options_electrolyte['parameters']['phi_c']
sol_t = sol_CC.t*tc #np.r_[sol_CC.t, sol_CV.t]*tc
# soly = sol_CC.y #np.c_[sol_CC.y, sol_CV.y]
# delV = soly[-1,:]*phic

#%%

# V vs t
#%%
# plt.figure()
# plt.plot(sol_t, delV, 'k.--')
# plt.xlabel(r"$t \ (s)$")
# plt.ylabel(r"$\Delta V \ (volts)$")
# plt.grid()
# plt.tight_layout()
#%%

#Validation with BESTmicro and Analytical
#%%
best_loc = "/Users/aliasad/Documents/Data/best_sim/"
# "W:/Group/Data-Unix/appli_PITSI/users/estelle/ali/PhD/1D-modelling/best-micro/simulation/simulation/Lifoil_electrolyte_graphite/Output/"

# time_name = "simulation_Lifoil_electrolyte_graphite_20220607_11_24_00"
time_name = "simulation_Lifoil_electrolyte_graphite_20230914_17_53_44"


f_name = best_loc + time_name + "/Uoft.log"
data = np.loadtxt(f_name)
sol_t_best = data[:,0][:251]
delV_best = data[:,2][:251]

soly_code = sol_CC.sol(sol_t_best/tc)
delV_code = soly_code[-1, :]*phic
sol_t_code = sol_t_best

# corrV = np.mean([abs(delV_code[i]-delV_best[i]) for i in [0,-1]])
# delV_best += corrV 

assert (sol_t_code[-1] == sol_t_best[-1])

#%%

# U cell vs t
#%%
from src.lib.analytical.HalfCell import lib1DCCSol 
sol_th = lib1DCCSol(phy_options, sim_options)
delV_analytical = sol_th.get_analytical_cell_voltage(sol_t_best)[0]

from matplotlib import rcParams, rcParamsDefault
# plt.style.use = 'default'
# rcParams.update(rcParamsDefault)
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
# plt.style.use = 'science'
rcParams['text.usetex'] = True
# rcParams['font.family'] = "cm"
# rcParams['mathtext.fontset'] = "cm"
fntA=15
plt.style.use = 'classic'

skp=11
plt.figure(dpi=600)
plt.plot(sol_t_code[::skp], delV_code[::skp],
         ls=' ', marker='s', ms=8, mfc='none',
         label=r"$\mathrm{1D\ LIB\ model\ (Radau5)}$")
plt.plot(sol_t_best[::skp], delV_best[::skp],
         ls=' ', marker='d', ms=6, mfc=None,
         label=r"$\mathrm{BEST\mathit{micro}}$")
plt.plot(sol_t_best[::skp], delV_analytical[::skp],
         color='k', ls='--', lw=1.5, alpha=.75,
         label=r"$\mathrm{Analytical\ solution}$")

plt.xlabel(r"$t\ \mathrm{(s)}$", fontsize=fntA)
plt.ylabel(r"$U_{cell}\ \mathrm{(V)}$", fontsize=fntA)
plt.grid(ls=':')
plt.legend(framealpha=.75, fancybox=True,
           loc=0, numpoints=1, fontsize=fntA-4)
plt.tight_layout()

if bSavefigs:
    plt.savefig("../tests/validation/0p5CC_analytical_BEST_Radau5.png", dpi=600)
    plt.savefig("../tests/validation/0p5CC_analytical_BEST_Radau5.pdf", dpi=600)

#%%

# Overpotential, OCP as f(c_s(t) ) vs . t
#%%
# from src.lib.model import ocp_graphite as ocp
# csx = np.linspace(0, 1., num=100)
# csx = sol_cv[2,:]/options_cathode['activematerial']['parameters']['c_s_max']
# Uoc = ocp.Ueq(csx)

# plt.figure()
# plt.plot(csx, Uoc)
# plt.xlabel(r"$\frac{c_{s}}{c_{s,max}}$")
# plt.ylabel(r"$U_{oc} \ (volts)$")
# plt.grid()
# plt.tight_layout()

# eta_c = sol_cv[3,:] - sol_cv[1,:] - Uoc[:]
# plt.figure()
# plt.plot(sol_t*tc, np.sinh(eta_c/(2*phic)))
# plt.xlabel(r"$t \ (s)$")
# plt.ylabel(r"$\eta_{am-e}^{(C)} \ (volts)$")
# plt.grid()
# plt.tight_layout()
#%%

# Rading BEST profiles
#%%
# time_name = "simulation_Lifoil_electrolyte_graphite_20220607_11_24_00"
time_name = "simulation_Lifoil_electrolyte_graphite_20230914_17_53_44"
f_name = best_loc + time_name + "/output/output_matlab_10.txt"
data = np.loadtxt(f_name, comments='%')

xx = data[:,0]
ce = data[:,3]
cs = data[:,4]
phie = data[:,5]
phis = data[:,6]

xx = np.sort(np.unique(xx))

xxe = xx[253:753]
xxs = xx[753:1003]
cex = ce[4*1006+253:4*1006+753]*1e6
csx = cs[4*1006+753:4*1006+1003]*1e6
phiex = phie[4*1006+253:4*1006+753]
phisx = phis[4*1006+753:4*1006+1003] 


#%%

# Space profiles
#%%
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
# plt.style.use = 'default'
# rcParams.update(rcParamsDefault)
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
# plt.style.use = 'science'
rcParams['text.usetex'] = True
# rcParams['font.family'] = "cm"
# rcParams['mathtext.fontset'] = "cm"
fzA=15
plt.style.use = 'classic'

cec= options_electrolyte['parameters']['c_e_c']
csc = options_cathode['activematerial']['parameters']['c_s_c']
phic = options_electrolyte['parameters']['phi_c']
tc = options_electrolyte['parameters']['t_c']
Lc= 40.0e-6
nam=options_cathode['activematerial']['mesh']['cellX'].size
# times for comparison
tcomp = np.array([0, 500])
tcompND = tcomp/tc

# assert all(tcompND==[sol_CC.t[0], sol_CC.t[-1]])

# Best micro profiles params
# grid 1006x3x3
# xe 253:753
# xam 753:1003
# xcc 1003:

# simulation
from src.lib.jac_reorder import  jac_reorder as tf   
kk=5
xeC = np.r_[options_electrolyte['mesh']['faceX'][0],
            options_electrolyte['mesh']['cellX'][:-1:kk],
            options_electrolyte['mesh']['cellX'][-1],
            options_electrolyte['mesh']['faceX'][-1]]
xsC = np.r_[options_cathode['activematerial']['mesh']['faceX'][0],
            options_cathode['activematerial']['mesh']['cellX'][:nam:kk]]

# analytical
from src.lib.analytical.HalfCell import lib1DCCSol 
sol_th = lib1DCCSol(phy_options, sim_options)


ceB=[]; ceC=[]; ceA=[]
csB=[]; csC=[]; csA=[]
peB=[]; peC=[]; peA=[]
psB=[]; psC=[]; psA=[]

fig = plt.figure(figsize=(10,7), dpi=600)
# fig.suptitle("Solution profiles for CC mode at C-rate=1.0")
ax1 = fig.add_subplot(2,2,1)  
ax2 = fig.add_subplot(2,2,3)  
ax3 = fig.add_subplot(2,2,2)  
ax4 = fig.add_subplot(2,2,4)

for i in range(len(tcomp)):
    
    if i==0:
        # ky = 0
        lblC = r"$\mathrm{1D\ LIB\ model\ (Radau5)}$"
        lblB = r"$\mathrm{BEST\mathit{micro}}$"
        lblA = r"$\mathrm{Analytical\ solution}$"
        mfc='none'
    else:
        # ky= -1
        lblA = None
        lblB = None
        lblC = None
        mfc=None
        
    # solyC = sol_CC.y[:, ky]# sol(tcomp[i]/tc)
    solyC = sol_CC.sol(tcomp[i]/tc)
    
    iauxa, ice, iphie, iauxc, ics, iphis = tf.rev_X(solyC, options_electrolyte['nCells'],
                                                    options_cathode['nCells'])
    ceCx = (np.r_[iauxa[0], ice[:-1:kk], ice[-1], iauxc[0]]*cec)
    peCx = (np.r_[iauxa[1], iphie[:-1:kk], iphie[-1], iauxc[1]]*phic)
    csCx = (np.r_[iauxc[2], ics[:nam:kk]]*csc)
    psCx = (np.r_[iauxc[3], iphis[:nam:kk]]*phic)
    ceC.append(ceCx); csC.append(csCx); peC.append(peCx); psC.append(psCx);
    ax1.plot(xeC*Lc/1e-6, ceCx,
             color=None, ls=' ', marker='s', ms=8, mfc=mfc,
             label=lblC)
    ax3.plot(xsC*Lc/1e-6, csCx,
             color=None, ls=' ', marker='s', ms=8, mfc=mfc,
             label=lblC)
    ax2.plot(xeC*Lc/1e-6, peCx,
             color=None, ls=' ', marker='s', ms=8, mfc=mfc,
             label=lblC)
    ax4.plot(xsC*Lc/1e-6, psCx,
             color=None, ls=' ', marker='s', ms=8, mfc=mfc,
             label=lblC)
    
    f_name = best_loc + time_name + f"/output/output_matlab_{int(tcomp[i]/100)}.txt"
    # f_name = best_loc + time_name + f"/output/output_matlab_{25}.txt"
    data = np.loadtxt(f_name, comments='%')
    xx = data[:,0]
    ce = data[:,3]
    cs = data[:,4]
    phie = data[:,5]
    phis = data[:,6]
    
    if i==0:
        xx = np.sort(np.unique(xx))*0.01/Lc
        xxe = xx[253:753:5*kk]-xx[252]
        xxs = xx[753:1003:5*kk]-xx[252]
    cex = ce[4*1006+253:4*1006+753:5*kk]*1e6
    csx = cs[4*1006+753:4*1006+1003:5*kk]*1e6
    phiex = phie[4*1006+253:4*1006+753:5*kk]
    phisx = phis[4*1006+753:4*1006+1003:5*kk]
    ceB.append(cex); csB.append(csx); peB.append(phiex);  psB.append(phisx); 
    ax1.plot(xxe*Lc/1e-6, cex,
             color=None, ls=' ', marker='d', ms=6, mfc=mfc,
             label=lblB)
    ax3.plot(xxs*Lc/1e-6, csx,
             color=None, ls=' ', marker='d', ms=6, mfc=mfc,
             label=lblB)
    ax2.plot(xxe*Lc/1e-6, phiex,
             color=None, ls=' ', marker='d', ms=6, mfc=mfc,
             label=lblB)
    ax4.plot(xxs*Lc/1e-6, phisx,
             color=None, ls=' ', marker='d', ms=6, mfc=mfc,
             label=lblB)
    
    ceAx, csAx, peAx, psAx = sol_th.get_spatial_profiles_c_phi(tcomp[i], xeC*Lc, 
                                                               (xsC-0.5)*Lc, get_phis=True)
    ax1.plot(xeC*Lc/1e-6, ceAx,
             color='k', ls='--', lw=1.5, alpha=.75,
             label=lblA)
    ax3.plot(xsC*Lc/1e-6, csAx,
             color='k', ls='--', lw=1.5, alpha=.75,
             label=lblA)
    ax2.plot(xeC*Lc/1e-6, peAx,
             color='k', ls='--', lw=1.5, alpha=.75,
             label=lblA)
    ax4.plot(xsC*Lc/1e-6, psAx,
             color='k', ls='--', lw=1.5, alpha=.75,
             label=lblA)

ax1.set_ylabel(r"$c_e\ \mathrm{(mol/m^3)}$", fontsize=fzA)
ax1.set_xlabel(r"$x\ \mathrm{(\mu m)}$", fontsize=fzA)
ax1.legend(framealpha=0.75, ncol=1, numpoints=1, loc=0, fontsize=fntA-4)
ax1.grid(ls=':')
plt.tight_layout()
   
ax2.set_ylabel(r"$\varphi_e\ \mathrm{(V)}$", fontsize=fzA)
ax2.set_xlabel(r"$x\ \mathrm{(\mu m)}$", fontsize=fzA)
ax2.grid(ls=':')
plt.tight_layout()

ax3.set_ylabel(r"$c_s\ \mathrm{(mol/m^3)}$", fontsize=fzA)
ax3.set_xlabel(r"$x\ \mathrm{(\mu m)}$", fontsize=fzA)
ax3.grid(ls=':')
plt.tight_layout()

ax4.set_ylim(0.9*min(psAx), 1.1*max(psAx))
ax4.set_ylabel(r"$\phi_s\ \mathrm{(V)}$", fontsize=fzA)
ax4.set_xlabel(r"$x\ \mathrm{(\mu m)}$", fontsize=fzA)
ax4.grid(ls=':')

ax3.set_xticks([20.0, 22.5, 25.0, 27.5, 30.0])
ax4.set_xticks([20.0, 22.5, 25.0, 27.5, 30.0])

plt.tight_layout()
if bSavefigs:
    plt.savefig("../tests/validation/0p5CC_analytical_BEST_Radau5_profiles.png", dpi=600)
    plt.savefig("../tests/validation/0p5CC_analytical_BEST_Radau5_profiles.pdf", dpi=600)   
plt.show()

#%%
# errPe = []
# errPs = []
# dxV = []
# Deviation from BEST
#%%
from scipy.interpolate import interp1d

peCode = peCx
peBest = phiex
peAnalytical = peAx
xeCode = xeC
xeBest = xxe

fpeCodeonBest = interp1d(xeCode, peCode)
peCode2Best = fpeCodeonBest(xeBest)

diffe = np.mean(abs(peCode2Best-peBest)/peCode2Best)
print(f"Mean deviation in phie with BESTmicro {diffe:.8E}")

diffeA = np.mean(abs(peCode-peAx)/peCode)
print(f"Mean deviation in phie with Analytical is {diffeA:.8E}")

psCode = psCx
psBest = phisx
psAnalytical = psAx
xsCode = xsC
xsBest = xxs


fpsCodeonBest = interp1d(xsCode, psCode)
psCode2Best = fpsCodeonBest(xsBest)

diffs = np.mean(abs(psCode2Best-psBest)/psCode2Best)
print(f"Mean deviation in phis with BESTmicro {diffs:.8E}")

diffsA = np.mean(abs(psCode-psAx)/psCode)
print(f"Mean deviation in phis with Analytical is {diffsA:.8E} ")

#%%

