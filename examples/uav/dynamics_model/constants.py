import numpy as np

from .containers import Controls

mass = 11.  # [kg]
rho = 1.2682  # air density, [kg / m^3]
g0 = 9.81  # gravity, [m/s^2]
b = 2.8956  # wing-span [m]
c = 0.18994  # wing chord [m]
S = 0.55  # wing area [m^2]
eos = 0.9  # Oswald's Efficiency Factor [dimensionless between 0 and 1]

# Some derived quantities
AR = b ** 2 / S
mg = mass * g0
rhoS = 0.5 * rho * S

alpha_stall = np.deg2rad(20.)  # stall angle of attack [deg]
aero_blend_rate = 50.  # barrier function coefficient for stall angle of attack

Jxx = 0.8244  # [kg m^2]
Jyy = 1.135  # [kg m^2]
Jzz = 1.759  # [kg m^2]
Jxz = 0.1204  # [kg m^2]

J_body = np.array([[Jxx, 0., -Jxz], [0., Jyy, 0.], [-Jxz, 0., Jzz]])
J_det = Jxx * Jzz - Jxz ** 2
J_inv_body = np.array([[Jzz, 0., Jxz], [0., J_det / Jyy, 0.], [Jxz, 0., Jxx]])
J_inv_body /= J_det

# Aerodynamic Partial Derivatives

# Lift
CL0 = 0.23  # zero angle of attack lift coefficient
CLalpha = 5.61  # given in supplement
CLq = 7.95  # needs to be normalized by c/2*Va
CLdeltaE = 0.13  # lift due to elevator deflection

# Drag
CD0 = 0.0437  # parasitic drag
CDalpha = 0.03  # drag slope
CDq = 0.  # drag wrt pitch rate
CDdeltaE = 0.0135  # drag due to elevator deflection

# Pitching Moment
Cm0 = 0.0135  # intercept of pitching moment
Cmalpha = -2.74  # pitching moment slope
Cmq = -38.21  # pitching moment wrt q
CmdeltaE = -0.99  # pitching moment from elevator

# Sideforce
CY0 = 0.
CYbeta = -0.83
CYp = 0.
CYr = 0.
CYdeltaA = 0.075
CYdeltaR = 0.19

# Rolling Moment
Cl0 = 0.
Clbeta = -0.13
Clp = -0.51
Clr = 0.25
CldeltaA = 0.17
CldeltaR = 0.0024

# Yawing Moment
Cn0 = 0.
Cnbeta = 0.073
Cnp = -0.069
Cnr = -0.095
CndeltaA = -0.011
CndeltaR = -0.069

# Basic propeller model
Sprop = 0.2027  # propeller area [m^2]
kmotor = 32.  # motor constant, DIFFERENT FROM BEARD
kTp = 0.  # motor torque constant
kOmega = 0.  # motor speed constant
Cprop = 0.45  # thrust efficiency coefficient, DIFFERENT FROM BEARD

# Alternate propeller Model
D_prop = 0.508  # prop diameter [m]
KV = 145.  # from datasheet [RPM/V]
KQ = 60. / (2. * np.pi * KV)  # [V-s/rad]
R_motor = 0.042  # [ohms]
i0 = 1.5  # no-load (zero-torque) current [A]
ncells = 12.
V_max = 3.7 * ncells  # max voltage for specified number of battery cells

# Propeller coefficients
C_Q2 = -0.01664
C_Q1 = 0.004970
C_Q0 = 0.005230
C_T2 = -0.1079
C_T1 = -0.06044
C_T0 = 0.09357

# Throttle setting for zero torque
zero_throttle = i0 * R_motor / V_max

# Control constraints
max_angle = np.deg2rad(25.)
min_controls = Controls(throttle=0.,
                        aileron=-max_angle,
                        elevator=-max_angle,
                        rudder=-max_angle)
max_controls = Controls(throttle=1.,
                        aileron=max_angle,
                        elevator=max_angle,
                        rudder=max_angle)
