import numpy as np
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# --- Converted Helper Functions from your MATLAB files ---

def astroConstants(in_val):
    """
    astroConstants.m - Returns astrodynamic-related physical constants.
    """
    if isinstance(in_val, (int, float)):
        in_val = [in_val]
    
    out = np.zeros(len(in_val))
    
    for i, id_val in enumerate(in_val):
        if id_val == 1:
            out[i] = 6.67259e-20  # Universal gravity constant (G) [km^3/(kg*s^2)]
        elif id_val == 2:
            out[i] = 149597870.7  # Astronomical Unit (AU) [km]
        elif id_val == 3:
            out[i] = 700000  # Sun mean radius [km]
        elif id_val == 4:
            out[i] = 0.19891000000000E+31 * 6.67259e-20  # Sun planetary constant (mu) [km^3/s^2]
        elif id_val == 5:
            out[i] = 299792.458  # Speed of light in vacuum [km/s]
        elif id_val == 6:
            out[i] = 9.80665  # Standard free fall [m/s^2]
        elif id_val == 7:
            out[i] = 384401  # Mean distance Earth-Moon [km]
        elif id_val == 8:
            out[i] = 23.43928111 * np.pi / 180  # Obliquity of the ecliptic at Epoch 2000 [rad]
        elif id_val == 9:
            out[i] = 0.1082626925638815e-2  # Gravitational field constant of the Earth
        elif id_val == 11:
            out[i] = 0.33020000000000E+24 * 6.67259e-20  # mu_Mercury
        elif id_val == 12:
            out[i] = 0.48685000000000E+25 * 6.67259e-20  # mu_Venus
        elif id_val == 13:
            out[i] = 0.59736990612667E+25 * 6.67259e-20  # mu_Earth
        elif id_val == 14:
            out[i] = 0.64184999247389E+24 * 6.67259e-20  # mu_Mars
        elif id_val == 15:
            out[i] = 0.18986000000000E+28 * 6.67259e-20  # mu_Jupiter
        elif id_val == 16:
            out[i] = 0.56846000000000E+27 * 6.67259e-20  # mu_Saturn
        elif id_val == 17:
            out[i] = 0.86832000000000E+26 * 6.67259e-20  # mu_Uranus
        elif id_val == 18:
            out[i] = 0.10243000000000E+27 * 6.67259e-20  # mu_Neptune
        elif id_val == 19:
            out[i] = 0.14120000000000E+23 * 6.67259e-20  # mu_Pluto
        elif id_val == 20:
            out[i] = 0.73476418263373E+23 * 6.67259e-20  # mu_Moon
        elif id_val == 21:
            out[i] = 0.24400000000000E+04  # Radius_Mercury
        elif id_val == 22:
            out[i] = 0.60518000000000E+04  # Radius_Venus
        elif id_val == 23:
            out[i] = 0.63781600000000E+04  # Radius_Earth
        elif id_val == 24:
            out[i] = 0.33899200000000E+04  # Radius_Mars
        elif id_val == 25:
            out[i] = 0.69911000000000E+05  # Radius_Jupiter
        elif id_val == 26:
            out[i] = 0.58232000000000E+05  # Radius_Saturn
        elif id_val == 27:
            out[i] = 0.25362000000000E+05  # Radius_Uranus
        elif id_val == 28:
            out[i] = 0.24624000000000E+05  # Radius_Neptune
        elif id_val == 29:
            out[i] = 0.11510000000000E+04  # Radius_Pluto
        elif id_val == 30:
            out[i] = 0.17380000000000E+04  # Radius_Moon
        elif id_val == 31:
            out[i] = 1367  # Energy flux density of the Sun [W/m2 at 1 AU]
        else:
            # print(f"Warning: Constant identifier {id_val} is not defined!")
            out[i] = 0
            
    if len(out) == 1:
        return out[0] # Return scalar if only one input was given
    else:
        return out # Return array if multiple inputs were given

def hms2fracday(hrs, mn, sec):
    """
    Helper function for date2jd.
    Converts hours, minutes, seconds to fractional day.
    """
    return hrs / 24 + mn / (24 * 60) + sec / (24 * 3600)

def date2jd(date):
    """
    date2jd.m - Julian day number from Gregorian date.
    date: list/tuple [year, month, day, hour, minute, second]
    """
    if not (isinstance(date, (list, tuple)) and len(date) == 6):
        raise ValueError("DATE2JD: The input should be a 6-elements vector: [year, month, day, hour, minute, second]")

    Y, M, D, hrs, mn, sec = date

    jd = 367 * Y - np.floor(7 * (Y + np.floor((M + 9) / 12)) / 4) \
         - np.floor(3 * np.floor((Y + (M - 9) / 7) / 100 + 1) / 4) \
         + np.floor(275 * M / 9) \
         + D + 1721028.5 + hms2fracday(hrs, mn, sec)
    return jd

def date2mjd2000(date):
    """
    date2mjd2000.m - Modified Julian Day 2000 number from Gregorian date.
    date: list/tuple [year, month, day, hour, minute, second]
    """
    mjd2000 = date2jd(date) - 2400000.5 - 51544.5
    return mjd2000

def ephMoon(MJD2000):
    """
    ephMoon.m - Ephemerides (cartesian position and velocity) of the Moon.
    Algorithm taken from "Fundamentals of Astrodynamics and Applications"
    (3rd edition), D. A. Vallado, p.290 (algorithm 31).
    """
    T_TDB = MJD2000 / 36525

    angles = np.array([
        134.9 + 477198.85 * T_TDB,
        259.2 - 413335.38 * T_TDB,
        235.7 + 890534.23 * T_TDB,
        269.9 + 954397.7 * T_TDB,
        357.5 + 35999.05 * T_TDB,
        186.6 + 966404.05 * T_TDB,
        93.3 + 483202.03 * T_TDB,
        228.2 + 960400.87 * T_TDB,
        318.3 + 6003.18 * T_TDB,
        217.6 - 407332.2 * T_TDB
    ])

    angles = angles * np.pi / 180

    s = np.sin(angles[0:10])
    c = np.cos(angles[0:4])

    L_ecl = (218.32 + 481267.883 * T_TDB) \
            + np.array([+6.29, -1.27, +0.66, +0.21, -0.19, -0.11]) @ s[0:6]
        
    phi_ecl = np.array([+5.13, +0.28, -0.28, -0.17]) @ s[6:10]

    P = 0.9508 \
        + np.array([+0.0518, +0.0095, +0.0078, +0.0028]) @ c

    eps = 23.439291 - 0.0130042 * T_TDB - 1.64e-7 * T_TDB**2 + 5.04e-7 * T_TDB**3

    L_ecl *= np.pi / 180
    phi_ecl *= np.pi / 180
    P *= np.pi / 180
    eps *= np.pi / 180

    r_mag = 1 / np.sin(P) * 6378.16  # Radius of the Earth copied from astro_constants for speed
    xP = r_mag * np.array([
        np.cos(L_ecl) * np.cos(phi_ecl),
        np.cos(eps) * np.cos(phi_ecl) * np.sin(L_ecl) - np.sin(eps) * np.sin(phi_ecl),
        np.sin(eps) * np.cos(phi_ecl) * np.sin(L_ecl) + np.cos(eps) * np.sin(phi_ecl)
    ])

    dt_seconds = 1
    dt_days = dt_seconds / 86400.0

    def _ephMoon_pos_only(mjd2000_pos_only):
        T_TDB_pos_only = mjd2000_pos_only / 36525
        angles_pos_only = np.array([
            134.9 + 477198.85 * T_TDB_pos_only, 259.2 - 413335.38 * T_TDB_pos_only,
            235.7 + 890534.23 * T_TDB_pos_only, 269.9 + 954397.7 * T_TDB_pos_only,
            357.5 + 35999.05 * T_TDB_pos_only, 186.6 + 966404.05 * T_TDB_pos_only,
            93.3 + 483202.03 * T_TDB_pos_only, 228.2 + 960400.87 * T_TDB_pos_only,
            318.3 + 6003.18 * T_TDB_pos_only, 217.6 - 407332.2 * T_TDB_pos_only
        ]) * np.pi / 180
        s_pos_only = np.sin(angles_pos_only)
        c_pos_only = np.cos(angles_pos_only[0:4])
        L_ecl_pos_only = (218.32 + 481267.883 * T_TDB_pos_only) + np.array([+6.29, -1.27, +0.66, +0.21, -0.19, -0.11]) @ s_pos_only[0:6]
        phi_ecl_pos_only = np.array([+5.13, +0.28, -0.28, -0.17]) @ s_pos_only[6:10]
        P_pos_only = 0.9508 + np.array([+0.0518, +0.0095, +0.0078, +0.0028]) @ c_pos_only
        eps_pos_only = 23.439291 - 0.0130042 * T_TDB_pos_only - 1.64e-7 * T_TDB_pos_only**2 + 5.04e-7 * T_TDB_pos_only**3
        L_ecl_pos_only *= np.pi / 180; phi_ecl_pos_only *= np.pi / 180; P_pos_only *= np.pi / 180; eps_pos_only *= np.pi / 180
        r_mag_pos_only = 1 / np.sin(P_pos_only) * 6378.16
        xP_pos_only = r_mag_pos_only * np.array([
            np.cos(L_ecl_pos_only) * np.cos(phi_ecl_pos_only),
            np.cos(eps_pos_only) * np.cos(phi_ecl_pos_only) * np.sin(L_ecl_pos_only) - np.sin(eps_pos_only) * np.sin(phi_ecl_pos_only),
            np.sin(eps_pos_only) * np.cos(phi_ecl_pos_only) * np.sin(L_ecl_pos_only) + np.cos(eps_pos_only) * np.sin(phi_ecl_pos_only)
        ])
        return xP_pos_only

    xP_dt = _ephMoon_pos_only(MJD2000 + dt_days)
    vP = (xP_dt - xP) / dt_seconds

    return xP, vP

def uplanet(mjd2000, ibody):
    """
    uplanet.m - Analytical ephemerides for planets.
    Returns mean Keplerian elements.
    """
    DEG2RAD = np.pi / 180.0
    G = astroConstants(1)
    ksun = astroConstants(4)
    KM = astroConstants(2)

    T = (mjd2000 + 36525) / 36525.00
    TT = T * T
    TTT = T * TT
    
    kep = np.zeros(6)

    if round(ibody) == 1: # Mercury
        kep[0] = 0.38709860
        kep[1] = 0.205614210 + 0.000020460*T - 0.000000030*TT
        kep[2] = 7.002880555555555560 + 1.86083333333333333e-3*T - 1.83333333333333333e-5*TT
        kep[3] = 4.71459444444444444e+1 + 1.185208333333333330*T + 1.73888888888888889e-4*TT
        kep[4] = 2.87537527777777778e+1 + 3.70280555555555556e-1*T +1.20833333333333333e-4*TT
        XM   = 1.49472515288888889e+5 + 6.38888888888888889e-6*T
        kep[5] = 1.02279380555555556e2 + XM*T
    elif round(ibody) == 2: # Venus
        kep[0] = 0.72333160
        kep[1] = 0.006820690 - 0.000047740*T + 0.0000000910*TT
        kep[2] = 3.393630555555555560 + 1.00583333333333333e-3*T - 9.72222222222222222e-7*TT
        kep[3] = 7.57796472222222222e+1 + 8.9985e-1*T + 4.1e-4*TT
        kep[4] = 5.43841861111111111e+1 + 5.08186111111111111e-1*T -1.38638888888888889e-3*TT
        XM   = 5.8517803875e+4 + 1.28605555555555556e-3*T
        kep[5] = 2.12603219444444444e2 + XM*T
    elif round(ibody) == 3: # Earth
        kep[0] = 1.000000230
        kep[1] = 0.016751040 - 0.000041800*T - 0.0000001260*TT
        kep[2] = 0.00
        kep[3] = 0.00
        kep[4] = 1.01220833333333333e+2 + 1.7191750*T + 4.52777777777777778e-4*TT + 3.33333333333333333e-6*TTT
        XM   = 3.599904975e+4 - 1.50277777777777778e-4*T - 3.33333333333333333e-6*TT
        kep[5] = 3.58475844444444444e2 + XM*T
    elif round(ibody) == 4: # Mars
        kep[0] = 1.5236883990
        kep[1] = 0.093312900 + 0.0000920640*T - 0.0000000770*TT
        kep[2] = 1.850333333333333330 - 6.75e-4*T + 1.26111111111111111e-5*TT
        kep[3] = 4.87864416666666667e+1 + 7.70991666666666667e-1*T - 1.38888888888888889e-6*TT - 5.33333333333333333e-6*TTT
        kep[4] = 2.85431761111111111e+2 + 1.069766666666666670*T +  1.3125e-4*TT + 4.13888888888888889e-6*TTT
        XM   = 1.91398585e+4 + 1.80805555555555556e-4*T + 1.19444444444444444e-6*TT
        kep[5] = 3.19529425e2 + XM*T
    elif round(ibody) == 5: # Jupiter
        kep[0] = 5.2025610
        kep[1] = 0.048334750 + 0.000164180*T  - 0.00000046760*TT -0.00000000170*TTT
        kep[2] = 1.308736111111111110 - 5.69611111111111111e-3*T +  3.88888888888888889e-6*TT
        kep[3] = 9.94433861111111111e+1 + 1.010530*T + 3.52222222222222222e-4*TT - 8.51111111111111111e-6*TTT
        kep[4] = 2.73277541666666667e+2 + 5.99431666666666667e-1*T + 7.0405e-4*TT + 5.07777777777777778e-6*TTT
        XM   = 3.03469202388888889e+3 - 7.21588888888888889e-4*T + 1.78444444444444444e-6*TT
        kep[5] = 2.25328327777777778e2 + XM*T
    elif round(ibody) == 6: # Saturn
        kep[0] = 9.5547470
        kep[1] = 0.055892320 - 0.00034550*T - 0.0000007280*TT + 0.000000000740*TTT
        kep[2] = 2.492519444444444440 - 3.91888888888888889e-3*T - 1.54888888888888889e-5*TT + 4.44444444444444444e-8*TTT
        kep[3] = 1.12790388888888889e+2 + 8.73195138888888889e-1*T -1.52180555555555556e-4*TT - 5.30555555555555556e-6*TTT
        kep[4] = 3.38307772222222222e+2 + 1.085220694444444440*T + 9.78541666666666667e-4*TT + 9.91666666666666667e-6*TTT
        XM   = 1.22155146777777778e+3 - 5.01819444444444444e-4*T - 5.19444444444444444e-6*TT
        kep[5] = 1.75466216666666667e2 + XM*T
    elif round(ibody) == 7: # Uranus
        kep[0] = 19.218140
        kep[1] = 0.04634440 - 0.000026580*T + 0.0000000770*TT
        kep[2] = 7.72463888888888889e-1 + 6.25277777777777778e-4*T + 3.95e-5*TT
        kep[3] = 7.34770972222222222e+1 + 4.98667777777777778e-1*T + 1.31166666666666667e-3*TT
        kep[4] = 9.80715527777777778e+1 + 9.85765e-1*T - 1.07447222222222222e-3*TT - 6.05555555555555556e-7*TTT
        XM   = 4.28379113055555556e+2 + 7.88444444444444444e-5*T + 1.11111111111111111e-9*TT
        kep[5] = 7.26488194444444444e1 + XM*T
    elif round(ibody) == 8: # Neptune
        kep[0] = 30.109570
        kep[1] = 0.008997040 + 0.0000063300*T - 0.0000000020*TT
        kep[2] = 1.779241666666666670 - 9.54361111111111111e-3*T - 9.11111111111111111e-6*TT
        kep[3] = 1.30681358333333333e+2 + 1.0989350*T + 2.49866666666666667e-4*TT - 4.71777777777777778e-6*TTT
        kep[4] = 2.76045966666666667e+2 + 3.25639444444444444e-1*T + 1.4095e-4*TT + 4.11333333333333333e-6*TTT
        XM   = 2.18461339722222222e+2 - 7.03333333333333333e-5*T
        kep[5] = 3.77306694444444444e1 + XM*T
    elif round(ibody) == 9: # Pluto
        kep[0] = 39.481686778174627
        kep[1] = 2.4467e-001
        kep[2] = 17.150918639446061
        kep[3] = 110.27718682882954
        kep[4] = 113.77222937912757
        XM   = 4.5982945101558835e-008
        kep[5] = 1.5021e+001 + XM*mjd2000*86400
    elif round(ibody) == 10: # Sun
        kep = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        if round(ibody) == 11:
            raise ValueError("No planet in the list. For the Moon use EphSS_kep instead")
        else:
            raise ValueError(f"No planet in the list (ID: {ibody})")
            
    kep[0]   = kep[0]*KM       # a [km]
    kep[2:6] = kep[2:6]*DEG2RAD    # Transform from deg to rad
    kep[5]   = np.fmod(kep[5], 2*np.pi)

    M = kep[5]
    e = kep[1]
    
    phi = M
    for _ in range(5):
        g = M - (phi - e * np.sin(phi))
        g_prime = (-1 + e * np.cos(phi))
        if abs(g_prime) < 1e-12:
            break
        phi = phi - g / g_prime
    
    if np.abs(np.cos(phi/2)) < 1e-12 and np.abs(np.sin(phi/2)) > 0:
        theta = np.pi if np.sin(phi/2) > 0 else -np.pi
    else:
        theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(phi / 2))
    
    kep[5] = theta

    return kep, ksun

def ephNEO(time, id_neo):
    """
    ephNEO.m - Ephemerides of Near Earth Objects
    This function relies on loading .mat files which is not straightforward in Python
    without scipy.io.loadmat. For now, it will raise an error.
    """
    raise NotImplementedError(
        "ephNEO requires loading .mat files ('PHAEphemeris_ATATD_2018_2019.mat' or 'allNEOEphemeris_ATATD_2018_2019.mat'). "
        "Please implement the data loading or provide the 'neo' data directly in Python."
    )

def kep2car(kep_elements, mu):
    """
    Converts Keplerian elements to Cartesian position and velocity vectors.
    kep_elements = [a, e, i, RAAN, arg_periapsis, true_anomaly] in km and radians
    mu = gravitational parameter in km^3/s^2
    Returns car = [x, y, z, vx, vy, vz]
    """
    a, e, i, RAAN, argp, nu = kep_elements

    r_mag = a * (1 - e**2) / (1 + e * np.cos(nu))
    
    x_perifocal = r_mag * np.cos(nu)
    y_perifocal = r_mag * np.sin(nu)
    z_perifocal = 0.0

    h = np.sqrt(mu * a * (1 - e**2))
    
    vx_perifocal = - (mu / h) * np.sin(nu)
    vy_perifocal = (mu / h) * (e + np.cos(nu))
    vz_perifocal = 0.0

    R_argp = np.array([
        [np.cos(argp), -np.sin(argp), 0],
        [np.sin(argp), np.cos(argp), 0],
        [0, 0, 1]
    ])
    
    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R_RAAN = np.array([
        [np.cos(RAAN), -np.sin(RAAN), 0],
        [np.sin(RAAN), np.cos(RAAN), 0],
        [0, 0, 1]
    ])
    
    R_perifocal_to_inertial = R_RAAN @ R_i @ R_argp

    r_perifocal = np.array([x_perifocal, y_perifocal, z_perifocal])
    v_perifocal = np.array([vx_perifocal, vy_perifocal, vz_perifocal])

    r_inertial = R_perifocal_to_inertial @ r_perifocal
    v_inertial = R_perifocal_to_inertial @ v_perifocal

    return np.concatenate((r_inertial, v_inertial))

def EphSS_car(n, t):
    """
    EphSS_car.m - Ephemerides of the solar system (Python equivalent).
    Outputs cartesian position and velocity of the body.
    """
    kep = None
    r = None
    v = None

    if n < 11:  # uplanet needed
        kep, _ = uplanet(t, n) 
    elif n == 11:  # ephMoon needed
        r, v = ephMoon(t)
    else:  # NeoEphemeris needed
        kep, _, _ = ephNEO(t, n)

    if n != 11:  # Planet or asteroid, Sun-centered
        mu = astroConstants(4)
        
        car = kep2car(kep, mu)
        r   = car[0:3]
        v   = car[3:6]
    
    return r, v

# --- User's Custom FGKepler_trA (Angular Propagation) ---
def FGKepler_trA(r0, r0_dot, delta_theta, mu):
    """
    FGKepler_trA calculates the final position vector using F and G solutions
    based on angular displacement (delta_theta), not time.
    Inputs:
        r0: Initial position vector (1x3 array)
        r0_dot: Initial velocity vector (1x3 array)
        delta_theta: Change in true anomaly (radians)
        mu: Gravitational parameter of the central body (e.g., Sun)
    Output:
        rf: Final position vector (1x3 array)
    """
    r0_norm = np.linalg.norm(r0)
    h_vec = np.cross(r0, r0_dot)
    h_norm = np.linalg.norm(h_vec)
    p = h_norm**2 / mu # Semi-latus rectum

    # sigma_0 is often denoted as vr0_over_sqrt_mu or similar in universal variable formulation
    # It's dot(r0, v0) / sqrt(mu)
    sigma_0 = np.dot(r0, r0_dot) / np.sqrt(mu)

    # Calculate rf_norm (magnitude of final position vector) using the orbital equation
    # This formula is derived from the radial equation of motion in polar coordinates
    rf_norm = p * r0_norm / (r0_norm + (p - r0_norm) * np.cos(delta_theta) - np.sqrt(p) * sigma_0 * np.sin(delta_theta))

    # F and G coefficients based on angular displacement
    F = 1 - (rf_norm / p) * (1 - np.cos(delta_theta))
    G = (r0_norm * rf_norm * np.sin(delta_theta)) / np.sqrt(mu * p)
    
    # Calculate final position vector
    rf = F * r0 + G * r0_dot
    
    return rf

# --- Numerical Integration Function ---
def twoBodyEOM(t, x, mu):
    """
    twoBodyEOM defines the first-order differential equations for the two-body problem.
    Inputs:
        t: Time (scalar, not used in this specific EOM for constant mu)
        x: State vector [rx; ry; rz; vx; vy; vz] (6x1 array)
        mu: Gravitational parameter of the central body (scalar)
    Output:
        dxdt: Derivative of the state vector [vx; vy; vz; ax; ay; az] (6x1 array)
    """
    r_vec = x[0:3]
    v_vec = x[3:6]
    
    r_norm = np.linalg.norm(r_vec) 
    a_vec = - (mu / r_norm**3) * r_vec
    
    dxdt = np.zeros(6)
    dxdt[0:3] = v_vec
    dxdt[3:6] = a_vec
    return dxdt

# --- Main Script ---

# Get the Sun's gravitational parameter
mu_Sun = astroConstants(4)

# Define departure and arrival dates
date_earth_depart = [2016, 3, 14, 12, 0, 0]
mjd2000_earth = date2mjd2000(date_earth_depart)
date_mars_arrival = [2016, 10, 15, 12, 0, 0]
mjd2000_mars = date2mjd2000(date_mars_arrival)

# Get Earth's position and velocity at departure from the Sun
r1, v1 = EphSS_car(3, mjd2000_earth)
# Get Mars' position and velocity at arrival from the Sun
r2, v2 = EphSS_car(4, mjd2000_mars)

# Calculate magnitudes and chord length
c_vec = r2 - r1
r1_norm = np.linalg.norm(r1)
r2_norm = np.linalg.norm(r2)
c_norm = np.linalg.norm(c_vec)

# Calculate parameters for the minimum energy orbit
a_min = (r1_norm + r2_norm + c_norm) / 4

# delta_theta between r1 and r2
cos_delta_theta = np.dot(r1, r2) / (r1_norm * r2_norm)
# Ensure value is within valid range for acos due to floating point inaccuracies
cos_delta_theta = np.clip(cos_delta_theta, -1.0, 1.0)
delta_theta = np.arccos(cos_delta_theta)

p_min = (r1_norm * r2_norm * (1 - cos_delta_theta)) / c_norm
e_min = np.sqrt(1 - (p_min / a_min))

beta_e = 2 * np.arcsin(np.sqrt((2 * a_min - c_norm) / (2 * a_min)))
t_m = 1
t_min_seconds = np.sqrt(a_min**3 / mu_Sun) * (np.pi - t_m * (beta_e - np.sin(beta_e)))
t_min_days = t_min_seconds / (24 * 3600)

print(f"Earth Position Vector (km): {r1}")
print(f"Earth Velocity Vector (km/s): {v1}")
print(f"Mars Position Vector (km): {r2}")
print(f"Mars Velocity Vector (km/s): {v2}")
print(f"Calculated Minimum Energy Time of Flight (seconds): {t_min_seconds:.4f}")
print(f"Calculated Minimum Energy Time of Flight (days): {t_min_days:.4f}")
print(f"Eccentricity of transfer orbit (e_min): {e_min:.4f}")

# Lagrange Coefficients F and G for the transfer orbit
# Note: delta_theta here is the angular separation between r1 and r2
F_transfer = 1 - (r2_norm / p_min) * (1 - np.cos(delta_theta))
G_transfer = (r1_norm * r2_norm * np.sin(delta_theta)) / np.sqrt(mu_Sun * p_min)

# Calculate the departure velocity for the transfer orbit (v1_transfer)
# This is derived from r2 = F*r1 + G*v1_transfer
v1_transfer = (r2 - F_transfer * r1) / G_transfer

print(f"\nTransfer Departure Velocity (v1_transfer) (km/s): {v1_transfer}")

# --- Plotting ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal') # Ensure aspect ratio is equal for correct orbital shape

# Plot Sun
ax.scatter(0, 0, 0, color='yellow', s=100, label='Sun (Origin)')

nPoints_plot = 360 # Number of points for plotting planetary orbits

# --- Plot Earth's Orbit ---
# Get Earth's initial state for plotting its full orbit
r_earth_initial_for_plot, v_earth_initial_for_plot = EphSS_car(3, mjd2000_earth)
Dtheta_Earth = np.linspace(0, 2 * np.pi, nPoints_plot)
rEarth_motion = np.zeros((nPoints_plot, 3))
for iT in range(nPoints_plot):
    # Propagate Earth's orbit using FGKepler_trA based on angular displacement
    rEarth_motion[iT, :] = FGKepler_trA(r_earth_initial_for_plot, v_earth_initial_for_plot, Dtheta_Earth[iT], mu_Sun)
ax.plot(rEarth_motion[:, 0], rEarth_motion[:, 1], rEarth_motion[:, 2], 'b-', label='Earth Orbit')

# --- Plot Mars' Orbit ---
# Get Mars' initial state for plotting its full orbit (starting from Earth's departure MJD for visual consistency)
# Or, as in your MATLAB code, starting from Mars' arrival MJD and propagating
# Let's use Mars' position at Earth's departure for a full orbit visualization
[r_mars_at_earth_depart_for_plot, v_mars_at_earth_depart_for_plot] = EphSS_car(4, mjd2000_earth)
Dtheta_Mars = np.linspace(0, 2 * np.pi, nPoints_plot)
rMars_motion = np.zeros((nPoints_plot, 3))
for iT in range(nPoints_plot):
    # Propagate Mars' orbit using FGKepler_trA based on angular displacement
    rMars_motion[iT, :] = FGKepler_trA(r_mars_at_earth_depart_for_plot, v_mars_at_earth_depart_for_plot, Dtheta_Mars[iT], mu_Sun)
ax.plot(rMars_motion[:, 0], rMars_motion[:, 1], rMars_motion[:, 2], 'r-', label='Mars Orbit')

# --- Plot Minimum Energy Transfer Orbit (using FGKepler_trA) ---
nPoints_transfer = 500 # More points for smoother transfer orbit
# The angular range for the transfer is delta_theta
Dtheta_transfer = np.linspace(0, delta_theta, nPoints_transfer)
rTransfer_motion_FG = np.zeros((nPoints_transfer, 3))
for iT in range(nPoints_transfer):
    # Propagate transfer orbit using FGKepler_trA based on angular displacement
    rTransfer_motion_FG[iT, :] = FGKepler_trA(r1, v1_transfer, Dtheta_transfer[iT], mu_Sun)
ax.plot(rTransfer_motion_FG[:, 0], rTransfer_motion_FG[:, 1], rTransfer_motion_FG[:, 2], 'g--', linewidth=2, label='Minimum Transfer Orbit')

# --- Plot Numerically Integrated Transfer Orbit (for verification) ---
# Initial state vector for ODE solver: [r_x, r_y, r_z, v_x, v_y, v_z]
initial_state_vector_ode = np.concatenate((r1, v1_transfer))
tspan_ode = [0, t_min_seconds] # Time span for integration in seconds

# Solve the ODE
sol = solve_ivp(lambda t, x: twoBodyEOM(t, x, mu_Sun), tspan_ode, initial_state_vector_ode, 
                rtol=1e-9, atol=1e-9, dense_output=True)

# Generate points from the solution for plotting
t_ode_plot = np.linspace(tspan_ode[0], tspan_ode[1], nPoints_transfer)
Y_ode_plot = sol.sol(t_ode_plot)
r_integrated_motion = Y_ode_plot[0:3, :].T # Transpose to get (N, 3) shape

ax.plot(r_integrated_motion[:, 0], r_integrated_motion[:, 1], r_integrated_motion[:, 2], 'm:', linewidth=1.5, label='ODE Integrated Transfer')

# --- Plot Start and End Points of the Transfer ---
ax.scatter(r1[0], r1[1], r1[2], color='blue', s=100, marker='o', label='Earth Departure (r1)')
ax.scatter(r2[0], r2[1], r2[2], color='red', s=100, marker='o', label='Mars Arrival (r2)')

# Set plot labels and title
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Orbits of Earth, Mars, and ExoMars TGO Transfer')
ax.legend(loc='best', fontsize='small')

# Adjust axis limits to encompass all orbits
all_points = np.concatenate((rEarth_motion, rMars_motion, rTransfer_motion_FG, r_integrated_motion))
max_range = np.max(np.abs(all_points))
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range]) # Keep Z-limit proportional to X/Y for visual consistency

plt.show()