# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:20:12 2022

@author: ali.asad
"""

"""
The code performs space convergence study on 1D finite volume mesh
for LIB half cell problem. The error for convergence is evaluated 
against either the solution on finest grid or the analytical 
solution (only possible with CC mode).
"""
##############################################IMPORTING MODULES#####################################################################################
#%%
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from src.lib.solver.radau_dae import RadauDAE
import scipy.sparse as sparse
from tqdm import tqdm

# customs modules
from src.lib.reader import  reader as params
from src.lib.model import setup_LIB as setup 
from src.lib.model import _1d_lib_dfv_model as model
from src.lib.post_process import time_series as y_ts
####################################################################################################################################################
#%%
refSolTtypes = ["Finest", "Analytical"]

refSol = refSolTtypes[1]
bIncludeAux = False
bSaveplots = False

nCellsStart = 32
nCellsEnd = 1600
nSamples = 8

tiniDim = 0. # s
tendDim = 5. # s

phy_options, sim_options = params.readInputs('input/')

nx_vec = np.logspace(np.log10(nCellsStart//4), np.log10(nCellsEnd//4), nSamples) 
nx_vec = nx_vec.astype(int)
nx_vec = np.unique(nx_vec)

for i, current_nx in enumerate(nx_vec):
    current_dx = 1./(4*current_nx)
    first_step = (current_dx**2)/2.
    print(current_dx, first_step)

print(f"Convergence study on nCells = {4*np.array(nx_vec)}")

#%%

# Study loops
#%%
xe_all = []
xs_all = []
ce_all = []
cs_all = []
phie_all = []
phis_all = []
sols = []
if bIncludeAux:
    auxA_all = []
    auxC_all = []

method=RadauDAE
atol = 1e-13
rtol = 1e-12

sim_options["ChargeRate"]["type"]["value"] = "CC"
sim_options["ChargeRate"]["xi"]["value"] = 1.0
sim_options["ChargeRate"]["c_s_i"]["value"] = 13000.0
# sim_options["ChargeRate"]["phi_s_L"]["value"] = lambda t: 0.1036 # volts

for i, current_nx in enumerate(tqdm(nx_vec)):    
    # dx altering
    sim_options["Mesh"]["ne"]["value"]= 2*current_nx
    sim_options["Mesh"]["nam"]["value"]= current_nx
    sim_options["Mesh"]["ncc"]["value"]= current_nx
    
    options_electrolyte, options_cathode = setup.getSetup(phy_options, sim_options, invertBV=True)
    options_cathode['sim_type']="monolithic"
    options_electrolyte['sim_type']="monolithic"

    tc = options_electrolyte['parameters']['t_c']
    
    tini = tiniDim/tc 
    tend = tendDim/tc 

    Ui = np.r_[options_electrolyte['y0'], options_cathode['y0']]
    # xe = options_electrolyte['mesh']['cellX']
    # iScale = options_electrolyte['parameters']['F']/((1.-options_electrolyte['parameters']['t_0_plus'])*options_electrolyte['parameters']['inv_N_e_c'])
    # iext = options_cathode['parameters']['i_1C']
    # # iext += 1e-8*iext
    # ceFit = (iext/iScale)*(xe) + Ui[0] + 1e-2
    # Ui[2:4*current_nx+2:2] = ceFit 
    massM = np.diag(np.r_[options_electrolyte['mass'], options_cathode['mass']])
    
    var_idx = abs( (np.r_[options_electrolyte['mass'], options_cathode['mass']]==0 )*1 - 1)
    
    # sparsity pattern
    uband=6; lband=-uband
    offsets = [i for i in range(lband,uband)]
    sparsity_pattern = sparse.diags(diagonals=[np.ones((Ui.size - abs(i))) for i in offsets], offsets=offsets)
    
    current_dx = 1./(4*current_nx)
    first_step = 1e-8#(current_dx**2)/2.
    
    outSol = solve_ivp(fun = lambda t, u:model.func_U(u, t, options_electrolyte, options_cathode),
                    t_span = (tini, tend),
                    y0 = Ui,
                    max_step=np.inf,
                    rtol=rtol, atol=atol,
                    var_index = var_idx,
                    jac=None, jac_sparsity=sparsity_pattern,
                    method=method, vectorized=False, first_step=first_step, dense_output=True,
                    mass=massM, bPrint=False)
    print(f"For simulation at nCells={4*current_nx}: {outSol.message}")
    if not outSol.success:
        sol=None
        print(f"solution failed for nCells={4*current_nx}")
        continue
        # raise Exception(f"solution failed for nCells={4*current_nx}")
    
    # print(tend, tendDim, outSol.t[-1], outSol.t[-1]*tc)
    
    titp = outSol.t
    yitp = outSol.y
    
    # Interpret solutions
    x, t, aux_a, c_e, phi_e, aux_c, c_s, phi_s = y_ts.get_y_t(titp, yitp,
                                                              options_electrolyte,
                                                              options_cathode)
    # print(tend, tendDim, outSol.t[-1], outSol.t[-1]*tc, t[-1])
    
    xe_all.append(x[:2*current_nx])
    xs_all.append(x[2*current_nx:3*current_nx])
    ce_all.append(c_e)
    cs_all.append(c_s[:current_nx, :])
    phie_all.append(phi_e)
    sols.append(outSol)
    # phis_all.append(phi_s)
    if bIncludeAux:
        auxA_all.append(aux_a)
        auxC_all.append(aux_c)
#%%

# Reference solution
#%%
ce_ref = []
cs_ref = []
phie_ref = []
phis_ref = []
if bIncludeAux:
    auxA_all = []
    auxC_all = []
    
if refSol == refSolTtypes[0]:
    from scipy.interpolate import interp1d
    f_inter_ce = interp1d(xe_all[-1], ce_all[-1][:,-1])
    f_inter_phie = interp1d(xe_all[-1], phie_all[-1][:,-1])       
    f_inter_cs = interp1d(xs_all[-1], cs_all[-1][:,-1])
    # f_inter_phis = interp1d(xs_all[-1], phis_all[-1][:,-1])
    
    for i in range(nx_vec.size-1):
        ce_ref.append(f_inter_ce(xe_all[i]))
        phie_ref.append(f_inter_phie(xe_all[i]))
        cs_ref.append(f_inter_cs(xs_all[i]))
        
elif refSol == refSolTtypes[1]:
    from src.lib.analytical.HalfCell import lib1DCCSol 
    solTh = lib1DCCSol(phy_options, sim_options)
    
    for i in range(nx_vec.size):
        ceTh, csTh, phieTh = solTh.get_spatial_profiles_c_phi(tendDim, xe_all[i], xs_all[i]-solTh.Le) 
        ce_ref.append(ceTh)
        phie_ref.append(phieTh)
        cs_ref.append(csTh)
else:
    raise Exception("No reference solution selected")
#%%
    
# Error evaluation
#%%
Lc = 40e-6 # m
Le = 20e-6 # m
Lam = 10e-6 # m
phic = options_electrolyte['parameters']['phi_c']
cec = options_electrolyte['parameters']['c_e_c']
csc = options_cathode['activematerial']['parameters']['c_s_c']

err_l1_ce = []
err_l2_ce = []
err_l1_phie = []
err_l2_phie = []
err_l1_cs = []
err_l2_cs = []

dxe = []
dxs = []

if refSol == refSolTtypes[0]:
    nErr = nx_vec.size-1
else:
    nErr = nx_vec.size
    
for i in range(nErr):    
    dx = xe_all[i][1] - xe_all[i][0]
    dxe.append(dx/Lc)

    err = (ce_ref[i] - ce_all[i][:,-1]) #/ cec
    err_l1 = np.linalg.norm(err, ord=1)
    err_l2 = np.linalg.norm(err, ord=2)
    # err_l1 = np.sum( abs(err) * dx) / Le
    # err_l2 = np.sqrt(np.sum( err**2 * dx) / Le)
    # # err_l1_ce.append(err_l1)
    # err_l2_ce.append(err_l2)
    nrmlz_l1 = np.sum(ce_ref[i])
    nrmlz_l2 = np.sqrt( np.sum(ce_ref[i]**2))
    err_l1_ce.append(err_l1/nrmlz_l1)
    err_l2_ce.append(err_l2/np.linalg.norm(ce_ref[i], ord=2)) 
    
    err = (phie_ref[i] - phie_all[i][:,-1]) #/ phic
    err_l1 = np.linalg.norm(err, ord=1)
    err_l2 = np.linalg.norm(err, ord=2)
    # err_l1 = np.sum( abs(err) * dx) / Le
    # err_l2 = np.sqrt(np.sum( err**2 * dx) / Le)
    # # err_l1_phie.append(err_l1)
    # err_l2_phie.append(err_l2) 
    nrmlz_l1 = np.sum(phie_ref[i])
    nrmlz_l2 = np.sqrt( np.sum(phie_ref[i]**2))
    err_l1_phie.append(err_l1/nrmlz_l1)
    err_l2_phie.append(err_l2/np.linalg.norm(phie_ref[i], ord=2)) 
    
    
    dx = xs_all[i][1] - xs_all[i][0]
    dxs.append(dx/Lc)

    err = (cs_ref[i] - cs_all[i][:,-1]) #/ csc
    err_l1 = np.linalg.norm(err, ord=1)
    err_l2 = np.linalg.norm(err, ord=2)
    # err_l1 = np.sum( abs(err) * dx) / Lam
    # err_l2 = np.sqrt(np.sum( err**2 * dx) / Lam)
    # err_l1_cs.append(err_l1)
    # err_l2_cs.append(err_l2) 
    nrmlz_l1 = np.sum(cs_ref[i])
    nrmlz_l2 = np.sqrt(np.sum(cs_ref[i]**2))
    err_l1_cs.append(err_l1/nrmlz_l1)
    err_l2_cs.append(err_l2/np.linalg.norm(cs_ref[i], ord=2)) 
    
    # err = (phis_ref[i] - phis_all[i][:,-1]) * dx
    # err_l1 = np.sum( abs(err) * dx) / Lam
    # err_l2 = np.sqrt(np.sum( err**2 * dx) / Lam)
    # err_l1_phis.append(err_l1)
    # err_l2_phis.append(err_l2)
#%%

# Convergence plots
#%%

bSaveplots = True
# plt.style.use('default')
fntA=10
fntB=15
fntC=20

# L1 errors
ord_th = 2
k2e = ((err_l1_ce[0]+err_l1_phie[0])/(1.5*dxe[0]**ord_th) )*np.array(dxe)**ord_th
k2s = ((err_l1_cs[0])/(1.2*dxe[0]**ord_th) )*np.array(dxs)**ord_th

ord_ce = np.average(np.gradient(np.log10(err_l1_ce), np.log10(dxe)))
ord_phie = np.average(np.gradient(np.log10(err_l1_phie), np.log10(dxe)))
ord_cs = np.average(np.gradient(np.log10(err_l1_cs), np.log10(dxs)))

fig = plt.figure(dpi=600)
plt.ylabel(r"$L_1 \ errors$", fontsize=fntA)
plt.xlabel(r"$\Delta_x/L^*$", fontsize=fntA)
plt.title(f"At time$=${tendDim}s with $\mathcal{{N}}=$ {[4*i for i in nx_vec]}", fontsize=fntA)


plt.loglog(dxe, err_l1_ce, color='b',
           marker='o', linestyle='-', markersize=7, markeredgewidth=1, mfc='none',
           label=f"$error\ in\ c_e\ \sim \mathcal{{O}}(\Delta x^{{{ord_ce:.2f}}})$")
plt.loglog(dxe, err_l1_phie, color='orange',
           marker='^', linestyle='-', markersize=7, markeredgewidth=1, mfc='none',
           label=r"$error\ in \ \varphi_e \sim$"+f"$\mathcal{{O}}(\Delta x^{{{ord_phie:.2f}}})$")
plt.loglog(dxs, err_l1_cs, color='g',
           marker='D', linestyle='-', markersize=7, markeredgewidth=1, mfc='none',
           label=f"$error\ in \ c_s\ \sim \mathcal{{O}}(\Delta x^{{{ord_cs:.2f}}})$")
plt.loglog(dxe, (k2e*k2s)**0.5, 'k--', 
           label=r"$\mathcal{O}(\Delta x^2)$")

plt.grid()
plt.legend(framealpha=0.75, fancybox=True,
            loc=0, numpoints=1, fontsize=fntA)
plt.tight_layout()
if bSaveplots:
    plt.savefig(f"../tests/convergence_test/fig_space_conv_L1_at_t{int(tendDim)}s_with_{refSol}.png", dpi=600)
#%%

# L2 errors 
#%%
import matplotlib.pyplot as plt
from matplotlib import rcParams
# plt.style.use = 'default'
# rcParams.update(rcParamsDefault)
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
# plt.style.use = 'science'
rcParams['text.usetex'] = True
# rcParams['font.family'] = "cm"
# rcParams['mathtext.fontset'] = "cm"
fzA=15
plt.style.use = 'classic'
ord_th = 2
k2e = ((err_l2_ce[0]+err_l2_phie[0])/(1.5*dxe[0]**ord_th) )*np.array(dxe)**ord_th
k2s = ((err_l2_cs[0])/(1.2*dxe[0]**ord_th) )*np.array(dxs)**ord_th

ord_ce = np.average(np.gradient(np.log10(err_l2_ce), np.log10(dxe)))
ord_phie = np.average(np.gradient(np.log10(err_l2_phie), np.log10(dxe)))
ord_cs = np.average(np.gradient(np.log10(err_l2_cs), np.log10(dxs)))

print(f"Order for ce: {ord_ce:.2f}")
print(f"Order for phie: {ord_phie:.2f}")
print(f"Order for cs: {ord_cs:.2f}")

clr = {'b':'#1f77b4',
       'o':'#ff7f0e',
       'g':'#2ca02c'}

plt.figure(dpi=600)
plt.ylabel(r"$\mathrm{error\ corrected}$", fontsize=fzA)
plt.xlabel(r"$\Delta x/L^*$", fontsize=fzA)
plt.title(f"$\mathcal{{N}}=\mathrm{[4*i for i in nx_vec]}\ at\ t={tendDim}\ s$")

plt.loglog(dxe, err_l2_ce, color=clr['b'],
            marker='o', linestyle='-', markersize=7, markeredgewidth=1, mfc='none',
            label=r"$\mathrm{error\ in\ \mathit{c_e}}$")# $\sim$ $\mathcal{{O}}(\Delta \tilde{{kk}} x^{{{ord_ce:.2f}}})$")
plt.loglog(dxe, err_l2_phie, color=clr['o'],
           marker='^', linestyle='-', markersize=7, markeredgewidth=1, mfc='none',
           label=r"$\mathrm{error\ in\  \varphi_{\mathit{e}}}$")#+f" $\sim$ $\mathcal{{O}}(\Delta \tilde x^{{{ord_phie:.2f}}})$")
plt.loglog(dxs, err_l2_cs, color=clr['g'],
           marker='D', linestyle='-', markersize=7, markeredgewidth=1, mfc='none',
           label=r"$\mathrm{error\ in\  \mathit{c_s}}$")# $\sim$ $\mathcal{{O}}(\Delta \tilde{{x}}^{{{ord_cs:.2f}}})$")
plt.loglog(dxe, (k2e*k2s)**0.5, 'k--', alpha=.75,
            label=r"$\mathcal{O}(\Delta x^2)$")

plt.grid(ls=':')
plt.legend(framealpha=0.75, fancybox=True,
           loc=0, numpoints=1, fontsize=fzA-4)
plt.tight_layout()
bSaveplots=True 
if bSaveplots:
    plt.savefig(f"../tests/convergence_test/fig_space_conv_L2_at_t{int(tendDim)}s_with_{refSol}.pdf", dpi=600)
    plt.savefig(f"../tests/convergence_test/fig_space_conv_L2_at_t{int(tendDim)}s_with_{refSol}.png", dpi=600)
plt.show()
#%%

# save the data for error vs nt
#%%
bSaveData=True
if bSaveData:
    if tendDim < 1.:
        tdis = f"p{int(tendDim*10)}"
    else:
        tdis = f"{int(tendDim)}"
    savefile_name = f"../tests/convergence_test/data_space_conv_L2_at_t{tdis}s_with_{refSol}.dat"
    
    save_array=[]
    for i in range(len(nx_vec)):
        save_array.append([nx_vec[i], dxe[i], err_l2_ce[i], err_l2_phie[i], dxs[i], err_l2_cs[i]])
    np.savetxt(savefile_name, np.array(save_array))
#%%

#%%
# bSaveplots = True
# plt.style.use('default')
# fntA=10
# fntB=15
# fntC=20

# # L1 errors
# ord_th = 2
# k2phie = ((err_l1_phie[0] + err_l2_phie[0])/(2.*dxe[0]**ord_th)) * np.array(dxe)**ord_th
# k2ce = ((err_l1_ce[0] + err_l2_ce[0])/(2.*dxe[0]**ord_th)) * np.array(dxe)**ord_th

# k2cs = ((err_l1_cs[0] + err_l2_cs[0] )/(2.*dxs[0]**ord_th) ) * np.array(dxs)**ord_th


# fig = plt.figure(dpi=600)
# plt.ylabel(r"$errors$", fontsize=fntB)
# plt.xlabel(r"$\Delta_x/L^*$", fontsize=fntB)
# plt.title(f"At time$=${tendDim}s with $\mathcal{{N}}=$ {[i for i in nx_vec]}", fontsize=fntA)


# plt.loglog(dxe, err_l1_ce, color='c',
#             marker='p', linestyle=' ', lw=.75, markersize=10, markeredgewidth=1, mfc='none')
# plt.loglog(dxe, err_l1_phie, color='orange',
#             marker='s', linestyle=' ', lw=.75, markersize=10, markeredgewidth=1, mfc='none')
# plt.loglog(dxs, err_l1_cs, color='g',
#             marker='D', linestyle=' ', lw=.75, markersize=10, markeredgewidth=1, mfc='none')

# plt.loglog(dxe, err_l2_ce, color='c',
#            marker='p', linestyle=' ', lw=.75, markersize=5, markeredgewidth=1)
# plt.loglog(dxe, err_l2_phie, color='orange',
#            marker='s', linestyle=' ', lw=.75, markersize=5, markeredgewidth=1)
# plt.loglog(dxs, err_l2_cs, color='g',
#            marker='D', linestyle=' ', lw=.75, markersize=5, markeredgewidth=1)

# plt.loglog(dxe, k2ce,
#            'k--', alpha=.75)
# plt.loglog(dxe, k2phie,
#            'k--', alpha=.75)
# plt.loglog(dxe, k2cs,
#            'k--', alpha=.75)
# plt.loglog(np.nan, np.nan,
#            'k--', alpha=.75,
#            label=r"$\mathcal{O}(\Delta x^2)$")

# plt.loglog(np.nan, np.nan, color='c',
#            marker='p', linestyle=' ', lw=.75, markersize=6, markeredgewidth=1, mfc='none',
#            label=r"$errors\ in\ c_e$")
# plt.loglog(np.nan, np.nan, color='orange',
#            marker='s', linestyle=' ', lw=.75, markersize=6, markeredgewidth=1, mfc='none',
#            label=r"$errors\ in\ \varphi_e$")
# plt.loglog(np.nan, np.nan, color='g',
#            marker='D', linestyle=' ', lw=.75, markersize=6, markeredgewidth=1, mfc='none',
#            label=r"$errors\ in\ c_s$")

# plt.grid()
# plt.legend(framealpha=0.25, fancybox=True,
#            loc=0, numpoints=1, fontsize=fntA)
# plt.tight_layout()

# if False:
#     plt.savefig(f"../tests/convergence_test/fig_space_conv_L1L2_at_t{int(tendDim)}s_with_{refSol}.pdf", dpi=600)
#%%


# CC solution profiles
#%%
method=RadauDAE
atol = 1e-13
rtol = 1e-12

sim_options["ChargeRate"]["type"]["value"] = "CC"
sim_options["ChargeRate"]["xi"]["value"] = 1.0
sim_options["ChargeRate"]["c_s_i"]["value"] = 13000.0

sim_options["Mesh"]["ne"]["value"]= 100
sim_options["Mesh"]["nam"]["value"]= 50
sim_options["Mesh"]["ncc"]["value"]= 50

options_electrolyte, options_cathode = setup.getSetup(phy_options, sim_options)
options_cathode['sim_type']="monolithic"
options_electrolyte['sim_type']="monolithic"

tc = options_electrolyte['parameters']['t_c']
tini = tiniDim/tc 
tend = tendDim/tc 

Ui = np.r_[options_electrolyte['y0'], options_cathode['y0']]
massM = np.diag(np.r_[options_electrolyte['mass'], options_cathode['mass']])

var_idx = abs( (np.r_[options_electrolyte['mass'], options_cathode['mass']]==0 )*1 - 1)

uband=6; lband=-uband
offsets = [i for i in range(lband,uband)]
sparsity_pattern = sparse.diags(diagonals=[np.ones((Ui.size - abs(i))) for i in offsets], offsets=offsets)

outSol = solve_ivp(fun = lambda t, u:model.func_U(u, t, options_electrolyte, options_cathode),
                t_span = (tini, tend),
                y0 = Ui,
                max_step=np.inf,
                rtol=rtol, atol=atol,
                var_index = var_idx,
                jac=None, jac_sparsity=sparsity_pattern,
                method=method, vectorized=False, first_step=1e-8, dense_output=True,
                mass=massM, bPrint=False)


#%%


#%%

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
rcParams['text.usetex'] = True
fzA=15
clr = plt.get_cmap("tab10")
plt.style.use = 'classic'

Lc = 40e-6 # m
Le = 20e-6 # m
Lam = 10e-6 # m

ne = options_electrolyte['nCells']
tc = options_electrolyte['parameters']['t_c']
cec = options_electrolyte['parameters']['c_e_c']
phic = options_electrolyte['parameters']['phi_c']
csc = options_cathode['activematerial']['parameters']['c_s_c']

t_ind = np.array([ .01, .1, .5, 1., 4., 5.])/tc
# t_ind = np.linspace(tini, tend, 6)
sol_i = outSol.sol(t_ind)

t_i = t_ind*tc
# sol_i = ref_sol_cc.sol(cosim_sols_cc[0].t)[:,0 : idxt]  

from src.lib.jac_reorder import  jac_reorder as tf
kk = 5
xe = np.r_[options_electrolyte['mesh']['faceX'][0],
            options_electrolyte['mesh']['cellX'][:-1:kk],
            options_electrolyte['mesh']['cellX'][-1],
            options_electrolyte['mesh']['faceX'][-1]]
xs = np.r_[options_cathode['activematerial']['mesh']['faceX'][0],
            np.r_[options_cathode['activematerial']['mesh']['cellX'],
            options_cathode['currentcollector']['mesh']['cellX']][::kk] ]

xe1 = np.r_[options_electrolyte['mesh']['faceX'][0],
            options_electrolyte['mesh']['cellX'][:-1],
            options_electrolyte['mesh']['cellX'][-1],
            options_electrolyte['mesh']['faceX'][-1]]
xs1 = np.r_[options_cathode['activematerial']['mesh']['faceX'][0],
            options_cathode['activematerial']['mesh']['cellX'],
            options_cathode['currentcollector']['mesh']['cellX'] ]

cex = []
phiex = []
csx = []
csx1 = []
phisx = []
phisx1 = []


iseA = []
iseC = []
lineformat = ['s-', '^-', 'd-', '1-', '2-', 'o-']
lineformat1 = ['s', '^', 'd', '1', '2', 'o']
lineformat2 = ['-', '-', '-', '-', '-', '-']

fig = plt.figure(figsize=(10,7), dpi=600)
# fig.suptitle("Solution profiles for CC mode at C-rate=1.0")
ax1 = fig.add_subplot(2,2,1)  
ax2 = fig.add_subplot(2,2,3)  
ax3 = fig.add_subplot(2,2,2)  
ax4 = fig.add_subplot(2,2,4)

for i in range(t_i.size):
    iauxa, ice, iphie, iauxc, ics, iphis = tf.rev_X(sol_i[:,i], options_electrolyte['nCells'],options_cathode['nCells'])
    
    
    # icsam = np.r_[ics[np.where(xs[1:]<=.75)], np.repeat(np.nan, len(np.where(xs[1:]>.75)[0]) ) ]
    icsam1 = np.r_[ics[np.where(xs1[1:]<=.75)], np.repeat(np.nan, len(np.where(xs1[1:]>.75)[0]) ) ]
    
    cex.append(np.r_[iauxa[0], ice[:-1:kk], ice[-1], iauxc[0]]*cec)
    phiex.append(np.r_[iauxa[1], iphie[:-1:kk], iphie[-1], iauxc[1]]*phic)
    
    csx.append(np.r_[iauxc[2]*csc, icsam1[::kk]*csc])
    csx1.append(np.r_[iauxc[2]*csc, icsam1*csc])
    
    phisx.append(np.r_[iauxc[3], iphis[::kk]]/iauxc[3])
    phisx1.append(np.r_[iauxc[3], iphis]/iauxc[3])
    
    ax1.plot(xe*Lc/1e-6, cex[i],
              lineformat1[i], c=clr(i), lw=1, mfc='none')
    ax2.plot(xe*Lc/1e-6, phiex[i],
              lineformat1[i], c=clr(i), lw=1, mfc='none')
    ax3.plot(xs*Lc/1e-6, csx[i],
              lineformat1[i], c=clr(i), lw=1, mfc='none')
    ax4.plot(xs*Lc/1e-6, phisx[i],
              lineformat1[i], c=clr(i), lw=0.5, mfc='none')
    
    ax1.plot(xe*Lc/1e-6, cex[i],
              lineformat2[i], c=clr(i), lw=1)
    ax2.plot(xe*Lc/1e-6, phiex[i],
              lineformat2[i], c=clr(i), lw=1)
    ax3.plot(xs1*Lc/1e-6, csx1[i],
              lineformat2[i], c=clr(i), lw=1)
    ax4.plot(xs1*Lc/1e-6, phisx1[i],
              lineformat2[i], c=clr(i), lw=0.5)
   
    ax1.plot(np.nan, np.nan,
              lineformat[i], c=clr(i), lw=1, mfc='none', 
              label=f"$t = {t_i[i]:.2f}\ \mathrm{{s}}$")
    
ax1.set_ylabel(r"$c_e\ \mathrm{(mol/m^3)}$", fontsize=fzA)
ax1.set_xlabel(r"$x\ \mathrm{(\mu m)}$", fontsize=fzA)
# ax1.set_xlim(-.025,.5+.025)

ax1.legend(loc=0, framealpha=0.75, ncol=2, numpoints=1, fontsize=fzA-4)
ax1.grid(ls=':')
plt.tight_layout()

   
ax2.set_ylabel(r"$\varphi_e\ \mathrm{(V)}$", fontsize=fzA)
ax2.set_xlabel(r"$x\ \mathrm{(\mu m)}$", fontsize=fzA)
ax2.grid(ls=':')
# ax2.set_xlim(-.025,.5+.025)
plt.tight_layout()

ax3.set_ylabel(r"$c_s\ \mathrm{(mol/m^3)}$", fontsize=fzA)
ax3.set_xlabel(r"$x\ \mathrm{(\mu m)}$", fontsize=fzA)


ax3.axvline((Le+Lam)/1e-6, color='k', lw=1, ls='--', alpha=.5)
ax3.text(25.2, 12850, r"$\Omega_{am}$", fontsize=fzA)
ax3.text(32.7, 12850, r"$\Omega_{cc}$", fontsize=fzA)


# ax3.set_ylim(11800, 13200)
ax3.set_xlim(ax4.get_xlim()[0], ax4.get_xlim()[-1])
ax3.grid(ls=':')
plt.tight_layout()

ax4.set_ylabel(r"$\frac{\phi_s}{\phi_{s}(L_e,t)}$", fontsize=fzA)
ax4.set_xlabel(r"$x\ \mathrm{(\mu m)}$", fontsize=fzA)

ax4.axvline((Le+Lam)/1e-6, color='k', lw=1, ls='--', alpha=.5)
ax4.text(25.2, 1. + 2.75e-6, r"$\Omega_{am}$", fontsize=fzA)
ax4.text(32.7, 1. + 2.75e-6, r"$\Omega_{cc}$", fontsize=fzA)

ax4.set_ylim(1.-.15e-6, 1. + 3.e-6)

#ax4.set_xlim(-.025+.5, 1.+.025)
ax4.grid(ls=':')
plt.tight_layout()


# plt.savefig("../tests/convergence_test/sol_profiles_monolithic_rad5.png", dpi=600)
if False:
    loc = "/Users/aliasad/Work/reports_and_docs/latex/sisc-paper/figs/"    
    plt.savefig(loc+"sol_profiles_monolithic_rad5.pdf", dpi=600)
plt.show()            

#%%

