import numpy as np
from math import asinh, pi
from typing import Tuple

from src.lib.analytical import U0_am
from src.lib.jac_reorder import  jac_reorder as tf


class lib1DCCSol:
    def __init__(self, phy_options, sim_options):
        self.optPhy = phy_options
        self.optSim = sim_options
        
        self.F = phy_options['Constants']['F']['value']
        self.R = phy_options['Constants']['R']['value']
        self.T = phy_options['Constants']['T']['value'] 
        
        # Elelctrolyte and Li foil params
        self.Le = phy_options['Electrolyte']['L_e']['value']
        self.De = phy_options['Electrolyte']['D_e_c']['value']
        self.i0_Li = phy_options['ButlerVolmerParams']['i_0_Li']['value']
        self.kappae = phy_options['Electrolyte']['kappa_e_c']['value']
        self.t0p = phy_options['Electrolyte']['t_0_plus']['value']
        
        # Active material and current collector parameters
        self.Lam = phy_options['Electrode']['L_am']['value']
        self.Lcc = phy_options['CurrentCollector']['L_cc']['value'] 
        self.Dam = phy_options['Electrode']['D_am_c']['value']
        self.cam_max = phy_options['Electrode']['c_s_max']['value']
        self.sigma_am = phy_options['Electrode']['sigma_am_c']['value']
        self.Fk0_am = phy_options['ButlerVolmerParams']['k_0_C']['value']
        self.sigma_cc = phy_options['CurrentCollector']['sigma_cc_c']['value']
        
        # IC/BC parameters
        self.ce_init = phy_options['Electrolyte']['c_e_i']['value']
        self.cam_init = sim_options['ChargeRate']['c_s_i']['value']
        self.Crate = sim_options['ChargeRate']['xi']['value']
        
        self.i_ext = self.Crate * self.F * self.cam_max * self.Lam / 3600  # A/m2
        self.betae = (1 - self.t0p) * self.i_ext / (self.F * self.De)  # mol/m3/m
        self.phie0 = (2 * self.R * self.T / self.F) * asinh(0.5 * self.i_ext / self.i0_Li)
        self.betas = self.i_ext / (self.F * self.Dam)  # mol/m3/m
        
        # Fourier coefficients of initial condition (modified IBVP)
        self.N = 10000
        self.N_vector = np.arange(1, self.N + 1, 1, dtype=int)
        self.be_n = (2 * self.Le / pi ** 2) * (1 - (-1) ** self.N_vector) / (self.N_vector ** 2)
        self.bs_0 = -self.Lam / 3
        self.bs_n = (2 * self.Lam / pi ** 2) / (self.N_vector ** 2)
#
# F = 96487 #96485  # Faraday's constant in C/mol
# R = 8.314  # Universal gas constant in J/mol/K
# T = 298.15 #293.15  # Kelvin
# #
# # Electrolyte and Li foil parameters
# Le = 20e-6  # m
# De = 1e-10  # m2/s
# kappae = 1  # S/m
# t0p = 0.4  # dimensionless
# i0_Li = 10  # A/m2
# #
# # Active material and current collector parameters
# Lam = 10e-6  # m
# Lcc = 10e-6  # m
# Material_Str = "Graphite"
# if Material_Str == "Graphite":
#     Dam = 3e-14  # m2/s
#     cam_max = 26000  # mol/m3
#     sigma_am = 100  # S/m
#     Fk0_am = 8.9e-7  # A m^2.5 mol^-1.5
# elif Material_Str == "NMC":
#     Dam = 4e-14  # m2/s
#     cam_max = 50400  # mol/m3
#     sigma_am = 2.8  # S/m
#     Fk0_am = 1.8e-6  # A m^2.5 mol^-1.5
# else:
#     print("Unknown active material type")
# sigma_cc = 3700  # S/m
#
# IC/BC parameters
# ce_init = 1000  # mol/m3
# cam_init = 13000  # mol/m3
# Crate = 0.5 # imposed C-rate
# i_ext = Crate * F * cam_max * Lam / 3600  # A/m2
# betae = (1 - t0p) * i_ext / (F * De)  # mol/m3/m
# phie0 = (2 * R * T / F) * asinh(0.5 * i_ext / i0_Li)
# betas = i_ext / (F * Dam)  # mol/m3/m


# # Fourier coefficients of initial condition (modified IBVP)
# N = 10000
# N_vector = np.arange(1, N + 1, 1, dtype=int)
# be_n = (2 * Le / pi ** 2) * (1 - (-1) ** N_vector) / (N_vector ** 2)
# bs_0 = -Lam / 3
# bs_n = (2 * Lam / pi ** 2) / (N_vector ** 2)


    def get_analytical_cell_voltage(self, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculation of cell voltage vs. time"""
        cs0 = np.zeros(len(time))
        eta0 = np.zeros(len(time))
        U = np.zeros(len(time))
        for k in range(len(time)):
            tfac = np.exp(-((pi * self.N_vector / self.Le) ** 2) * self.De * time[k])
            ce0_t = self.ce_init - self.betae * self.Le / 2 + self.betae * np.dot(self.be_n, tfac)
            ce1_t = self.ce_init + self.betae * self.Le / 2 - self.betae * np.dot(self.be_n, tfac)
            tfac = np.exp(-((pi * self.N_vector / self.Lam) ** 2) * self.Dam * time[k])
            cs0[k] = self.cam_init + self.betas * (-self.Dam * time[k] / self.Lam + self.bs_0 + np.dot(self.bs_n, tfac))
            i0_am = self.Fk0_am * np.sqrt(ce1_t * cs0[k] * (self.cam_max - cs0[k]))
            eta0[k] = (2 * self.R * self.T / self.F) * asinh(0.5 * self.i_ext / i0_am)
            U[k] = (
                self.phie0
                + (2 * self.R * self.T / self.F) * (1 - self.t0p) * np.log(ce1_t / ce0_t)
                + (self.Le / self.kappae) * self.i_ext
            )
            U[k] = (
                U[k]
                + U0_am.Ueq(cs0[k] / self.cam_max) 
                + eta0[k]
                + (self.Lam / self.sigma_am + self.Lcc / self.sigma_cc) * self.i_ext
            )
        return U, cs0
    
    
    def get_spatial_profiles_c_phi(self, t, xe, xs, get_phis=False):
        """Get spatial profiles of electrolyte and solid concentrations and potential vs. x
        x should contain 0 point?
        """
        
        # if(xe[0]!=0.0):
        #     xe = np.insert(np.array(xe), 0, 0.0)
        #     modReturn=True
        
        ce = np.zeros(len(xe) )
        cs = np.zeros(len(xs) )
           
    
        # Electrolyte domain
        tfac = np.exp(-((pi * self.N_vector / self.Le) ** 2) * self.De * t)
        for i in range(len(xe)):
            xfac = self.be_n * np.cos(self.N_vector * pi * xe[i] / self.Le)
            ce[i] = self.ce_init + self.betae * (xe[i] - self.Le / 2 + np.dot(xfac, tfac))
    
        if(xe[0]==0.0):
            ce0_t = ce[0]
        else:
            ce0_t = self.ce_init - self.betae * self.Le / 2 + self.betae * np.dot(self.be_n, tfac) 
        phie = (
            self.phie0 + (2 * self.R * self.T / self.F) * (1 - self.t0p) * np.log(ce / ce0_t) + (self.i_ext / self.kappae) * xe
        )
        # Active material
        tfac = np.exp(-((pi * self.N_vector / self.Lam) ** 2) * self.Dam * t)
        for i in range(len(xs)):
            xfac = self.bs_n * np.cos(self.N_vector * pi * xs[i] / self.Lam)
            cs[i] = self.cam_init + self.betas * (
                xs[i] * (1 - 0.5 * xs[i] / self.Lam) - self.Dam * t / self.Lam + self.bs_0 + np.dot(xfac, tfac)
            )
        
        if get_phis:
            phis0 = self.get_analytical_cell_voltage(t*np.ones(1))[0]
            phis = phis0 + self.i_ext/self.sigma_am*xs
            return ce, cs, phie, phis
        else:    
            return ce, cs, phie
    
    def get_analytical_y(self, t, opt_electrolyte, opt_cathode, non_dim=True):
        
        tc = opt_electrolyte['parameters']['t_c']
        Lc = opt_electrolyte['parameters']['L_c']
        t = t*tc
            
        xeCC = opt_electrolyte['mesh']['cellX']*Lc
        xe0 = opt_electrolyte['mesh']['faceX'][0]*Lc
        xem1 = opt_electrolyte['mesh']['faceX'][-1]*Lc
        xe = np.r_[xe0, xeCC, xem1]
        
        xamCC = opt_cathode['activematerial']['mesh']['cellX']*Lc - xem1
        xccCC = opt_cathode['currentcollector']['mesh']['cellX']*Lc - xem1
        xam0 = opt_cathode['activematerial']['mesh']['faceX'][0]*Lc - xem1
        xam = np.r_[xam0, xamCC]
        xs = np.r_[xam, xccCC]
            
        
        ce, cs, phie = self.get_spatial_profiles_c_phi(t, xe, xam)
        phis0 = self.get_analytical_cell_voltage(t*np.ones(1))[0]
        
        phis_am = phis0 + self.i_ext/opt_cathode['activematerial']['parameters']['sigma_am_c']*xam
        phis_cc = phis0 + self.i_ext/opt_cathode['currentcollector']['parameters']['sigma_cc_c']*xccCC
        phis = np.r_[phis_am, phis_cc]
            
        if non_dim:
            cec = opt_electrolyte['parameters']['c_e_c']
            csc = opt_cathode['activematerial']['parameters']['c_s_c']
            phic = opt_electrolyte['parameters']['phi_c']
            
            ce = ce/cec
            cs = cs/csc
            phie = phie/phic
            phis = phis/phic
        
        ce0 = ce[0]
        cem1 = ce[-1]
        ceCC = ce[1:-1]
        
        cs0 = cs[0]
        csCC = np.r_[cs[1:], cs[-1]*np.ones(xccCC.size)]
        
        phie0 = phie[0]
        phiem1 = phie[-1]
        phieCC = phie[1:-1]      
        
        phis0 = phis[0]
        phisCC = phis[1:]
        
        y = np.r_[ce0, phie0, tf.gather_Xi(ceCC, phieCC), cem1, phiem1,
                  cs0, phis0, tf.gather_Xi(csCC, phisCC)]
        
        return y
    
    def get_analytical_y_t(self, t, opt_electrolyte, opt_cathode):
        yt = np.zeros(t.size)
        for i in range(t.size):
            yt[i]= self.get_analytical_y(t[i], opt_electrolyte, opt_cathode)
        return yt.T
    
    # # Calculation of solution profiles at specific times
    # _, ax_ce = plt.subplots()
    # _, ax_cs = plt.subplots()
    # _, ax_phie = plt.subplots()
    # _, ax_U = plt.subplots()
    # for tk in display_times:
    #     nx = 10
    #     xe = np.linspace(0, Le, nx)
    #     xs = np.linspace(0, Lam, nx // 2)
    #     ce, cs, phie = get_spatial_profiles_c_phi(tk, xe, xs)
    #     ax_ce.plot(xe, ce, label=f"time={tk} s", marker="d")
    #     ax_cs.plot(xs + Le, cs, label=f"time={tk} s", marker="d")
    #     ax_phie.plot(xe, phie, label=f"time={tk} s", marker="d")
    #     nx = 20
    #     xe = np.linspace(0, Le, nx - 1)
    #     xs = np.linspace(0, Lam, nx // 2 - 1)
    #     ce, cs, phie = get_spatial_profiles_c_phi(tk, xe, xs)
    #     ax_ce.plot(xe, ce, label=f"time={tk} s", marker="o")
    #     ax_cs.plot(xs + Le, cs, label=f"time={tk} s", marker="o")
    #     ax_phie.plot(xe, phie, label=f"time={tk} s", marker="o")
    #
    # ax_ce.set(
    #     xlabel="x, [m]",
    #     ylabel="Electrolyte concentration, [mol/m3]",
    #     title=f"Analytical solution, Crate: {Crate}",
    # )
    # ax_ce.grid()
    # ax_cs.set(
    #     xlabel="x, [m]",
    #     ylabel="Solid concentration, [mol/m3]",
    #     title=f"Analytical solution, Crate: {Crate}",
    # )
    # ax_cs.grid()
    # ax_phie.set(
    #     xlabel="x, [m]",
    #     ylabel="Electrolyte potential, [V]",
    #     title=f"Analytical solution, Crate: {Crate}",
    # )
    # ax_phie.grid()
    #
    #
    # ax_U.set_title(f"Analytical solution, Crate: {Crate}")
    # ax_U.set_xlabel("Time, [s]")
    # ax_U.set_ylabel("cell voltage, [V]")
    # time = np.linspace(0, Tf, 1000)
    # U = get_analytical_cell_voltage(time)
    # ax_U.plot(time, U)
    # ax_ce.legend()
    # ax_cs.legend()
    # ax_phie.legend()
    # ax_U.legend()
    # plt.show()
