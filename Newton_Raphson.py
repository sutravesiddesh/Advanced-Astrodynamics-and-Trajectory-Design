import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- Parameters ---
hp = 400  # Perigee altitude [km]
ha = 6000 # Apogee altitude [km]
R_E = 6378 # Earth radius [km]
u = 398600 # Earth's gravitational parameter [km^3/s^2]

# --- Orbital Elements Calculation ---
rp = R_E + hp  # Perigee radius [km]
ra = R_E + ha  # Apogee radius [km]
e = (ra - rp) / (ra + rp)  # Eccentricity
a = (ra + rp) / 2  # Semi-major axis [km]

# --- Time and Mean Anomaly ---
t = 0.65 * 3600  # Time from periapsis [s] (0.65 hours converted to seconds)
n = np.sqrt(u / a**3)  # Mean motion [rad/s]
M = n * t  # Mean Anomaly [radians]

print(f"Calculated Mean Anomaly (M): {M:.8f} radians")
print(f"Eccentricity (e): {e:.8f}")

# --- Solve Kepler's Equation using Newton-Raphson ---
E_i = M  # Initial guess 
tol = 1e-12  
max_iter = 1000 
delta_E = 1 
no_iter = 0 

print("\n--- Newton-Raphson Method ---")
while abs(delta_E) > tol and no_iter < max_iter:
    no_iter += 1
    f_E = E_i - e * np.sin(E_i) - M
    f_prime_E = 1 - e * np.cos(E_i)
    
    if f_prime_E == 0: # Avoid division by zero
        print("Warning: Derivative is zero, stopping Newton-Raphson.")
        break
        
    delta_E = - f_E / f_prime_E
    E_i = E_i + delta_E

E_f = E_i # Final Eccentric Anomaly from Newton-Raphson
print(f"Eccentric Anomaly (Newton-Raphson): {E_f:.8f} radians")
print(f"Number of iterations: {no_iter}")

# --- Solve Kepler's Equation using fsolve (SciPy equivalent of MATLAB's fzero) ---
# Define the Kepler's equation function for fsolve
def kepler_equation_func(E, eccentricity, mean_anomaly):
    return E - eccentricity * np.sin(E) - mean_anomaly

E_i_fsolve = M # Initial guess for fsolve
# fsolve returns an array, so take the first element
E_f_fsolve = fsolve(kepler_equation_func, E_i_fsolve, args=(e, M))[0]

print("\n--- fsolve Results ---")
print(f"Eccentric Anomaly (fsolve): {E_f_fsolve:.8f} radians")

# --- Comparison ---
difference = E_f - E_f_fsolve
print("\n--- Comparison ---")
print(f"Difference (Newton-Raphson - fsolve): {difference:.10f} radians")

# --- Graphical Representation ---
E_plot = np.linspace(0, 2 * np.pi, 500) 
f_E_plot = E_plot - e * np.sin(E_plot) - M 

plt.figure(figsize=(10, 6)) 
plt.plot(E_plot, f_E_plot, 'b-', linewidth=1.5, label='f(E) = E - e*sin(E) - M') 
plt.axhline(0, color='k', linestyle='--', linewidth=0.8, label='y=0') 

# Mark Newton-Raphson solution
plt.plot(E_f, 0, 'ro', markersize=8, markeredgewidth=2, label='Newton-Raphson Solution')
plt.text(E_f, 0.1, f'NR Solution: {E_f:.4f} rad',
         verticalalignment='bottom', horizontalalignment='left', color='r')

# Mark fsolve solution
plt.plot(E_f_fsolve, 0, 'gx', markersize=8, markeredgewidth=2, label='fsolve Solution')
plt.text(E_f_fsolve, -0.1, f'fsolve Solution: {E_f_fsolve:.4f} rad',
         verticalalignment='top', horizontalalignment='left', color='g')

plt.xlabel('Eccentric Anomaly, E (radians)')
plt.ylabel('f(E) = E - e*sin(E) - M')
plt.title('Graphical Verification of Kepler\'s Equation Root')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()