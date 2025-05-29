import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
mu_Earth = 398600  
R_E = 6378         

SV_0 = np.array([-18676, 6246, 12474, 0.551, -1.946, -3.886])

# Propagation time
tf = 12 * 3600  # seconds

# Equation of motion (Two-Body Dynamics)
def F2BDyn(t, x):
    r_magnitude = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return [
        x[3],
        x[4],
        x[5],
        -mu_Earth * x[0] / (r_magnitude**3),
        -mu_Earth * x[1] / (r_magnitude**3),
        -mu_Earth * x[2] / (r_magnitude**3)
    ]

# Solve the ordinary differential equation
sol = solve_ivp(F2BDyn, [0, tf], SV_0, rtol=1e-6, atol=1e-6, method='RK45')

Tstep = sol.t
SVt = sol.y.T  

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Earth
N = 20
X = R_E * np.outer(np.cos(np.linspace(0, 2 * np.pi, N)), np.sin(np.linspace(0, np.pi, N)))
Y = R_E * np.outer(np.sin(np.linspace(0, 2 * np.pi, N)), np.sin(np.linspace(0, np.pi, N)))
Z = R_E * np.outer(np.ones(N), np.cos(np.linspace(0, np.pi, N)))
ax.plot_surface(X, Y, Z, color='b', alpha=0.6) 

# Plot the satellite trajectory
ax.plot(SVt[:, 0], SVt[:, 1], SVt[:, 2], 'r', label='Satellite Trajectory') 

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Satellite Trajectory around Earth')
ax.set_aspect('equal', adjustable='box') 
ax.view_init(elev=20, azim=75) 

plt.legend()
plt.grid(True)
plt.show()