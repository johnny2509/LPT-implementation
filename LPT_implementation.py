'''This is the .py file for the Bachelor's thesis
"Alternative Approaches to LPT-based Forward 
Modeling", for exam number: 197402

Various parts of the code have been used from
Aaron Mach, Hussein Abbas and Simon Schmidt 
Thomsen, which is marked as "AARON", "HUSSEIN",
or "SIMON".

Generative AI (GAI) supplied blocks of code 
from "ChatGPT-3.5", "GitHub Copilot Chat" 
(version GTP-4o) and "Deepseek Chat version 2.0" 
(version DeepSeek-V3), marked accordingly. 
If the same code appears more than once, it will 
not be marked again.'''

# %%

'''importation of needed libraries and class'''

from classy import Class
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import scipy.interpolate as scint
from scipy.interpolate import interp1d
import scipy as ss
from scipy.integrate import solve_ivp
import matplotlib.ticker as ticker

'''parameters and constants'''

h = 0.6736 # Hubble rate
L = 500.0 # length of the box SI: [Mpc/h]
N = 64 # sum of grid cells (corners)
cell_size = L / N # size of one cell 
H0 = 100 * h # Hubble parameter
V = L**3 # Volume of the box
fac = np.sqrt(2 * np.pi / V) * (N**3) # Normalization to match CLASS
k_quist = np.pi * N / L # Nyquist freq.

'''power spectrum computation from class'''

def get_power_spectrum(z, #HUSSEIN, ChatGPT-3.5
                       k=np.logspace(-4, 0, 500), 
                       returnk=False, 
                       plot=False): 
    params = {
        'output': 'mPk',
        'H0': 67.36,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'z_pk': z,
        'P_k_max_h/Mpc': 10,
        'non linear': 'none',
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute() 

    k_vals = k.flatten() 
    Pk = np.array([cosmo.pk(ki, z) for ki in k_vals]) 

    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.loglog(k_vals, Pk, lw=1, color='navy') 
        plt.xlabel(r'$k\ \left[h/\mathrm{Mpc}\right]$', fontsize=14)
        plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc}/h)^3\right]$', fontsize=14)
        plt.title(f"Linear Matter Power Spectrum at z = {params['z_pk']}", fontsize=16)
        plt.grid(True, which='both', ls='--', alpha=0.7)
        plt.tight_layout()
    
    cosmo.struct_cleanup()

    if returnk == True: 
        return k_vals, Pk 
    else:
        return Pk

'''determination of growth factors D1 and D2'''

omega_b = 0.02237
omega_cdm = 0.12
Omega_M = (omega_b + omega_cdm) / h**2
Omega_L = 1.0 - Omega_M

cosmo = Class()
cosmo.set({
    'h': h,
    'Omega_b': omega_b,
    'Omega_cdm': omega_cdm,
    'output': 'mTk',
    'z_max_pk': 100
})
cosmo.compute()
cosmo.struct_cleanup()
cosmo.empty()

def event_stop(t, y): #AARON
    return y[0] - 1   

event_stop.terminal = True 
event_stop.direction = 0   

def dydt(t, y): #AARON (modified)
    a, D, Ddot, D2, D2dot = y

    E_a = np.sqrt(Omega_L + (Omega_M / a**3))
    adot = a**2 * H0 * E_a

    Ddotdot = - (1/a) * adot * Ddot + (3/2) * Omega_M * H0**2/a * D
    D2dotdot = - (1/a) * adot * D2dot + (3/2) * Omega_M * H0**2/a * (D2 + D**2)

    return [adot, Ddot, Ddotdot, D2dot, D2dotdot]

initial = [1e-9, 1e-9, 0.0, 0.0, 0.0] #AARON

t_span = (0, 0.5) #AARON
t_eval = np.linspace(*t_span, 100000) #AARON

sol = solve_ivp(dydt, #AARON
                t_span=t_span, 
                y0=initial, 
                t_eval=t_eval,
                rtol=1e-10,
                events=event_stop,
                method='BDF')

aval = sol.y[0] #AARON
D1 = sol.y[1] #AARON
D2 = sol.y[3] #AARON

D1n = D1 / D1[-1] #AARON
D2n = -D2 / D1[-1]**2 #AARON

D1_interp = interp1d(aval, D1n, kind='cubic', bounds_error=False, fill_value='extrapolate')
D2_interp = interp1d(aval, D2n, kind='cubic', bounds_error=False, fill_value='extrapolate')
  
# function that returns D1, D2 in terms of scale factor
def get_D1D2_at_z(z): #ChatGPT-3.5
    a_eval = 1 / (1 + z)
    D1z = D1_interp(a_eval)
    D2z = D2_interp(a_eval)
    return D1z, D2z

'''function for cloud-in-cell (CIC)'''

def cic_interpolation(xd, yd, zd, Psi_x, Psi_y, Psi_z, N, L): #SIMON

    # particle positions are added to displacement and normalized and "% N" ensures periodicity
    X = ((xd + Psi_x.real.flatten()) / (L/N)) % N
    Y = ((yd + Psi_y.real.flatten()) / (L/N)) % N
    Z = ((zd + Psi_z.real.flatten()) / (L/N)) % N

    mass_grid = np.zeros((N, N, N)) 

    # integer part of normalized positions 
    i_corner = np.floor(X).astype(int) % N
    j_corner = np.floor(Y).astype(int) % N
    k_corner = np.floor(Z).astype(int) % N

    # offset calculation
    x_off = X - np.floor(X)
    y_off = Y - np.floor(Y)
    z_off = Z - np.floor(Z)

    # determination of weight on all 8 corners of cell
    for ox, oy, oz in [(0,0,0), (0,0,1), (0,1,0), (0,1,1),
                   (1,0,0), (1,0,1), (1,1,0), (1,1,1)]:

        wx = (1 - x_off) if ox == 0 else x_off
        wy = (1 - y_off) if oy == 0 else y_off
        wz = (1 - z_off) if oz == 0 else z_off
        weight = wx * wy * wz 
    
        i_s = (i_corner + ox) % N
        j_s = (j_corner + oy) % N
        k_s = (k_corner + oz) % N
    
        np.add.at(mass_grid, (i_s, j_s, k_s), weight) 

    return mass_grid

'''gradient function used for 2LPT'''

def grad(f, ks): #ChatGPT-3.5
   return np.fft.ifftn(1j * ks * np.fft.fftn(f)).real 

'''power spectrum function which takes given delta-field''' 

def compute_ps(delta_x, #HUSSEIN (modified)
               kx, ky, kz, L, N, 
               correct_cic=False,
               extend_k=True): 
    
    delta_k = np.fft.fftn(delta_x) 

    # cic correction function for delta-fields from cic
    if correct_cic: 
        k_quist = np.pi * N / L
        W_CIC_corrected = (
            np.sinc(kx / (2 * k_quist)) * 
            np.sinc(ky / (2 * k_quist)) * 
            np.sinc(kz / (2 * k_quist))
        ) ** 2  
        W_CIC_corrected[W_CIC_corrected < 0.1] = 0.1  # not dividing by 0
        delta_k /= W_CIC_corrected

    Pk = np.abs(delta_k)**2 / fac**2

    k_vals = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kxg, kyg, kzg = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_mag = np.sqrt(kxg**2 + kyg**2 + kzg**2)  

    k_min = 2 * np.pi / L  
    k_max = np.pi * N / L

    if extend_k:
        k_max *= 10

    # binning for power spectrum
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), num=250)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    Pk_binned = np.zeros_like(k_centers)
    counts = np.zeros_like(k_centers)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                k_val = k_mag[i, j, k]
                if k_val > 0:
                    bin_idx = np.digitize(k_val, k_bins) - 1
                    if 0 <= bin_idx < len(Pk_binned):
                        Pk_binned[bin_idx] += Pk[i, j, k]
                        counts[bin_idx] += 1

    valid_bins = counts > 0
    Pk_binned[valid_bins] /= counts[valid_bins]

    return k_centers[valid_bins], Pk_binned[valid_bins]


'''comutation of power spectra for different z-values'''

z_values = [0, 1, 2, 3, 4, 5]
results = []

for z in z_values: #GitHub Copilot Chat assisted expanding to expand for a list of z-values
    print(f"Initializes pipeline for for z = {z}...")

    D1z, D2z = get_D1D2_at_z(z)
    
    k_class, P_k_class = get_power_spectrum(z, returnk=True)
    valid_indices = P_k_class >= 0
    Pk_values_cleaned = P_k_class[valid_indices]
    k_values_cleaned = k_class[valid_indices]
    Pk_interp = interp1d(k_values_cleaned, 
                         Pk_values_cleaned,
                         kind='cubic',  
                         bounds_error=False, 
                         fill_value="extrapolate")
    
    real_part = np.random.normal(0, 1/np.sqrt(2), size=(N, N, N)) #ChatGPT-3.5, HUSSEIN
    imag_part = np.random.normal(0, 1/np.sqrt(2), size=(N, N, N))
    R = (real_part + 1j * imag_part)  

    # enforcing hermitian symmetry
    for i in range(N): #HUSSEIN
        for j in range(N):
            for k in range(N):
                sym_i, sym_j, sym_k = (-i) % N, (-j) % N, (-k) % N
                R[sym_i, sym_j, sym_k] = np.conj(R[i, j, k])

    # applying nyquist condition
    if N % 2 == 0: #AARON
        R[N//2, :, :] = R[N//2, :, :].real
        R[:, N//2, :] = R[:, N//2, :].real
        R[:, :, N//2] = R[:, :, N//2].real

    R[0, 0, 0] = R[0, 0, 0].real #HUSSEIN

    k_vals = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[k_mag == 0] = 1e-10

    delta_k = np.sqrt(Pk_interp(k_mag)) * R * fac #HUSSEIN
    delta_k[0, 0, 0] = 0

    delta_x = np.fft.ifftn(delta_k).real 

    psi_kx = 1j * kx / (k_mag**2) * delta_k 
    psi_ky = 1j * ky / (k_mag**2) * delta_k
    psi_kz = 1j * kz / (k_mag**2) * delta_k 

    # first order displacements
    psi_x = np.fft.ifftn(psi_kx).real 
    psi_y = np.fft.ifftn(psi_ky).real
    psi_z = np.fft.ifftn(psi_kz).real 

    # 2LPT calculation starts
    phi1_k = -delta_k / (k_mag**2) #ChatGPT-3.5
    phi1_k[0, 0, 0] = 0.0
    phi1 = np.fft.ifftn(phi1_k).real

    Phi_xx = grad(grad(phi1, kx), kx) #ChatGPT-3.5
    Phi_yy = grad(grad(phi1, ky), ky)
    Phi_zz = grad(grad(phi1, kz), kz)

    Phi_xy = grad(grad(phi1, kx), ky)
    Phi_xz = grad(grad(phi1, kx), kz)
    Phi_yz = grad(grad(phi1, ky), kz)

    S = (Phi_xx * Phi_yy - Phi_xy**2 + #ChatGPT-3.5
         Phi_xx * Phi_zz - Phi_xz**2 +
         Phi_yy * Phi_zz - Phi_yz**2)

    S_k = np.fft.fftn(S) #ChatGPT-3.5
    S_k[0, 0, 0] = 0.0

    psi2_kx = -1j * kx / (k_mag**2) * S_k #ChatGPT-3.5
    psi2_ky = -1j * ky / (k_mag**2) * S_k
    psi2_kz = -1j * kz / (k_mag**2) * S_k

    # second order displacements
    psi2_x = np.fft.ifftn(psi2_kx).real #ChatGPT-3.5
    psi2_y = np.fft.ifftn(psi2_ky).real 
    psi2_z = np.fft.ifftn(psi2_kz).real 

    psi_2_x = psi_x + D2z * psi2_x 
    psi_2_y = psi_y + D2z * psi2_y
    psi_2_z = psi_z + D2z * psi2_z

    # setting up a grid with particle positions
    pos = np.linspace(0, L, N, endpoint = False) #SIMON
    xs, ys, zs = np.meshgrid(pos, pos, pos, indexing = 'ij')

    # converting to 1D-array
    xs = xs.flatten() #SIMON
    ys = ys.flatten()
    zs = zs.flatten()

    cic1 = cic_interpolation(xs, ys, zs, psi_x, psi_y, psi_z, N, L) #SIMON
    delta_cic1 = cic1 - 1 # capture density contrast
    #print(f"delta_1 = {np.max(delta_cic1)}") #used for values of tab. 5.1

    cic2 = cic_interpolation(xs, ys, zs, psi_2_x, psi_2_y, psi_2_z, N, L) #SIMON
    delta_cic2 = cic2 - 1 
    #print(f"delta_2 = {np.max(delta_cic2)}") #used for values of tab. 5.1

    k_fft, Pk_fft = compute_ps(delta_x, kx=kx, ky=ky, kz=kz, L=L, N=N, correct_cic=False)
    k_cic1, Pk_cic1 = compute_ps(delta_cic1, kx=kx, ky=ky, kz=kz, L=L, N=N, correct_cic=True)
    k_cic2, Pk_cic2 = compute_ps(delta_cic2, kx=kx, ky=ky, kz=kz, L=L, N=N, correct_cic=True)

    results.append((z, k_class, P_k_class, k_fft, Pk_fft, k_cic1, Pk_cic1, k_cic2, Pk_cic2))


'''plotting the different power spectra'''

# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, axes = plt.subplots(2, 3, figsize=(22, 13), constrained_layout=True)

fig.suptitle(f"Power Spectra for Different Redshifts $z$, with $N={N}$, $L={L}$ Mpc/h", fontsize=34, y=1.05)

fig.supxlabel(r'$k$ [$h$/Mpc]', fontsize=30, y=-0.04)  
fig.supylabel(r'$P(k)$ [$(h^{-1}$Mpc)$^3$]', fontsize=30, x=-0.03)  

axes = axes.flatten()

zoom_limits = {
    1: {"xlim": (10**(-0.6), 0.4), "ylim": (10**(2), 10**(2.6))},
    2: {"xlim": (10**(-0.6), 0.5), "ylim": (10**(2), 10**(2.5))},
    3: {"xlim": (10**(-0.7), 0.75), "ylim": (10**(0.7), 10**(2.7))},
    4: {"xlim": (10**(-0.7), 0.75), "ylim": (10**(0.7), 10**(2.7))},
    5: {"xlim": (10**(-0.7), 0.75), "ylim": (10**(0.7), 10**(2.7))},
}

#GitHub Copilot Chat assisted expanding to expand for a list of z-values

for i, (z, k_class, P_k_class, k_fft, Pk_fft, k_cic1, Pk_cic1, k_cic2, Pk_cic2) in enumerate(results):
    ax = axes[i]  
    ax.loglog(k_class, P_k_class, 'k-', lw=2, label='CLASS')
    ax.loglog(k_fft, Pk_fft, 'o', color='#95a5a6', markersize=6, markeredgecolor='k', markeredgewidth=0.5, label='Randomized CLASS')
    ax.loglog(k_cic1, Pk_cic1, 's', markersize=6, color='#3498db', markeredgecolor='k', markeredgewidth=0.5, label='1LPT CIC corrected')
    ax.loglog(k_cic2, Pk_cic2, 'D', markersize=6, color='#e74c3c', markeredgecolor='k', markeredgewidth=0.5, label='2LPT CIC corrected')
    ax.axvline(k_quist, color='r', linestyle='--', linewidth=2, label='Nyquist freq.')
    ax.set_title(f'$z = {z}$', fontsize=30, pad=10)

    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.set_xlim(10**(-2), 1.1)

    if i == 0:
        ax.legend(fontsize=19, loc='lower left', frameon=True, framealpha=0.9, edgecolor='0.8')

    if i >= 1:  
        inset_ax = inset_axes(ax, width="45%", height="45%", loc='lower left', borderpad=2)
        inset_ax.loglog(k_class, P_k_class, 'k-', lw=2)
        inset_ax.loglog(k_fft, Pk_fft, 'o', color='#95a5a6', markersize=6, markeredgecolor='k', markeredgewidth=0.5)
        inset_ax.loglog(k_cic1, Pk_cic1, 's', markersize=6, color='#3498db', markeredgecolor='k', markeredgewidth=0.5)
        inset_ax.loglog(k_cic2, Pk_cic2, 'D', markersize=6, color='#e74c3c', markeredgecolor='k', markeredgewidth=0.5)
        inset_ax.axvline(k_quist, color='r', linestyle='--', linewidth=2)

        if i in zoom_limits:
            inset_ax.set_xlim(zoom_limits[i]["xlim"])
            inset_ax.set_ylim(zoom_limits[i]["ylim"])

        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.tick_params(
            axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False
        )
        inset_ax.grid(True, which='both', ls='--', alpha=0.5)

for j in range(len(z_values), len(axes)):
    fig.delaxes(axes[j])

plt.show()

# %%

'''Jacobian, to determine amount of shell crossing for different z'''

z_values = [0, 1, 2, 3, 4, 5]

fig, axes = plt.subplots(2, 3, figsize=(22, 13), constrained_layout=True, sharey=True)

fig.suptitle("Shell Crossing for 1LPT and 2LPT at Different Redshifts $z$", fontsize=34, y=1.05)

fig.supxlabel("Determinant of Jacobian", fontsize=30, y=-0.04)
fig.supylabel("Normalized Count", fontsize=30, x=-0.03)

for i, z in enumerate(z_values): #GitHub Copilot Chat assisted expanding to expand for a list of z-values

    D1z, D2z = get_D1D2_at_z(z)

    k_class, P_k_class = get_power_spectrum(z, returnk=True)
    valid_indices = P_k_class >= 0
    Pk_values_cleaned = P_k_class[valid_indices]
    k_values_cleaned = k_class[valid_indices]
    Pk_interp = interp1d(k_values_cleaned, 
                         Pk_values_cleaned,
                         kind='cubic',  
                         bounds_error=False, 
                         fill_value="extrapolate")

    delta_k = np.sqrt(Pk_interp(k_mag)) * R * fac
    delta_k[0, 0, 0] = 0

    psi_kx = 1j * kx / (k_mag**2) * delta_k
    psi_ky = 1j * ky / (k_mag**2) * delta_k
    psi_kz = 1j * kz / (k_mag**2) * delta_k

    psi_x = np.fft.ifftn(psi_kx).real
    psi_y = np.fft.ifftn(psi_ky).real
    psi_z = np.fft.ifftn(psi_kz).real

    phi1_k = -delta_k / (k_mag**2) #ChatGPT-3.5
    phi1_k[0, 0, 0] = 0.0
    phi1 = np.fft.ifftn(phi1_k).real

    Phi_xx = grad(grad(phi1, kx), kx) #ChatGPT-3.5
    Phi_yy = grad(grad(phi1, ky), ky)
    Phi_zz = grad(grad(phi1, kz), kz)

    Phi_xy = grad(grad(phi1, kx), ky)
    Phi_xz = grad(grad(phi1, kx), kz)
    Phi_yz = grad(grad(phi1, ky), kz)

    S = (Phi_xx * Phi_yy - Phi_xy**2 + #ChatGPT-3.5
         Phi_xx * Phi_zz - Phi_xz**2 +
         Phi_yy * Phi_zz - Phi_yz**2)

    S_k = np.fft.fftn(S) #ChatGPT-3.5
    S_k[0, 0, 0] = 0.0

    psi2_kx = -1j * kx / (k_mag**2) * S_k #ChatGPT-3.5
    psi2_ky = -1j * ky / (k_mag**2) * S_k
    psi2_kz = -1j * kz / (k_mag**2) * S_k

    psi2_x = np.fft.ifftn(psi2_kx).real #ChatGPT-3.5
    psi2_y = np.fft.ifftn(psi2_ky).real 
    psi2_z = np.fft.ifftn(psi2_kz).real 

    psi_2_x = psi_x + D2z * psi2_x 
    psi_2_y = psi_y + D2z * psi2_y
    psi_2_z = psi_z + D2z * psi2_z

    dPxdx = np.gradient(psi_x, cell_size, axis=0) #ChatGPT-3.5
    dPxdy = np.gradient(psi_x, cell_size, axis=1)
    dPxdz = np.gradient(psi_x, cell_size, axis=2)

    dPydx = np.gradient(psi_y, cell_size, axis=0)
    dPydy = np.gradient(psi_y, cell_size, axis=1)
    dPydz = np.gradient(psi_y, cell_size, axis=2)

    dPzdx = np.gradient(psi_z, cell_size, axis=0)
    dPzdy = np.gradient(psi_z, cell_size, axis=1)
    dPzdz = np.gradient(psi_z, cell_size, axis=2)

    dP2xdx = np.gradient(psi_2_x, cell_size, axis=0) #ChatGPT-3.5
    dP2xdy = np.gradient(psi_2_x, cell_size, axis=1)
    dP2xdz = np.gradient(psi_2_x, cell_size, axis=2)

    dP2ydx = np.gradient(psi_2_y, cell_size, axis=0)
    dP2ydy = np.gradient(psi_2_y, cell_size, axis=1)
    dP2ydz = np.gradient(psi_2_y, cell_size, axis=2)

    dP2zdx = np.gradient(psi_2_z, cell_size, axis=0)
    dP2zdy = np.gradient(psi_2_z, cell_size, axis=1)
    dP2zdz = np.gradient(psi_2_z, cell_size, axis=2)

    J11 = 1 + dPxdx #ChatGPT-3.5
    J12 = dPxdy
    J13 = dPxdz
    J21 = dPydx
    J22 = 1 + dPydy
    J23 = dPydz
    J31 = dPzdx
    J32 = dPzdy
    J33 = 1 + dPzdz

    J11_2 = 1 + dP2xdx #ChatGPT-3.5
    J12_2 = dP2xdy
    J13_2 = dP2xdz
    J21_2 = dP2ydx
    J22_2 = 1 + dP2ydy
    J23_2 = dP2ydz
    J31_2 = dP2zdx
    J32_2 = dP2zdy
    J33_2 = 1 + dP2zdz

    # calculation of determinant 
    detJ = (J11 * (J22 * J33 - J23 * J32) #ChatGPT-3.5
            - J12 * (J21 * J33 - J23 * J31)
            + J13 * (J21 * J32 - J22 * J31))
    
    detJ_2 = (J11_2 * (J22_2 * J33_2 - J23_2 * J32_2) #ChatGPT-3.5
            - J12_2 * (J21_2 * J33_2 - J23_2 * J31_2)
            + J13_2 * (J21_2 * J32_2 - J22_2 * J31_2))

    shell_crossing_mask = detJ <= 0 #ChatGPT-3.5
    n_crossings = np.sum(shell_crossing_mask)
    total_particles = N**3
    crossing_percentage = (n_crossings / total_particles) * 100
    #print(f"shell crossing 1LPT at z = {z}: {crossing_percentage}") #used to get values for tab. 5.1

    shell_crossing_mask_2 = detJ_2 <= 0 #ChatGPT-3.5
    n_crossings_2 = np.sum(shell_crossing_mask_2)
    total_particles = N**3
    crossing_percentage_2 = (n_crossings_2 / total_particles) * 100
    #print(f"shell crossing 2LPT at z = {z}: {crossing_percentage_2}") #used to get values for tab. 5.1

    ax = axes[0, i]
    ax.hist(detJ.flatten(), bins=100, range=(-5, 5), density=True, color='blue', alpha=0.7)
    ax.axvline(0, color='red', linestyle='-', linewidth=4, label='Shell crossing threshold')
    ax.set_title(f'1LPT at z = {z}', fontsize=30, pad=10)
    ax.legend([f"Particles: {n_crossings}\n({crossing_percentage:.2f}%)"], loc='upper left', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)
    ax.grid(True)

    ax = axes[1, i]
    ax.hist(detJ_2.flatten(), bins=100, range=(-5, 5), density=True, color='green', alpha=0.7)
    ax.axvline(0, color='red', linestyle='-', linewidth=4, label='Shell crossing threshold')
    ax.set_title(f'2LPT at z = {z}', fontsize=30, pad=10)
    ax.legend([f"Particles: {n_crossings_2}\n({crossing_percentage_2:.2f}%)"], loc='upper left', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)
    ax.grid(True)

plt.show()

# %%

'''To generate fig. 2.1, ChatGPT-3.5'''

import numpy as np
import matplotlib.pyplot as plt

z = np.logspace(-3, 5, 1000)
a = 1 / (1 + z)

# parameters taken from "Introduction to Cosmology" by Barbara Ryden
Omega_r0 = 9e-5
Omega_m0 = 0.31
Omega_L0 = 0.69
H0 = 67.36

E2 = Omega_r0 / a**4 + Omega_m0 / a**3 + Omega_L0

Omega_r = (Omega_r0 / a**4) / E2
Omega_m = (Omega_m0 / a**3) / E2
Omega_L = Omega_L0 / E2

a_eq = Omega_r0 / Omega_m0
z_eq = 1 / a_eq - 1
print(f"Matter-Radiation Equality occurs at z = {z_eq:.0f}")

a_de = (Omega_m0 / Omega_L0)**(1/3)
z_de = 1 / a_de - 1
print(f"Dark Energy Dominance occurs at z = {z_de:.2f}")

plt.figure(figsize=(8, 5))  
plt.loglog(z, Omega_r, label=r'$\Omega_r$', color='orange', linewidth=2)
plt.loglog(z, Omega_m, label=r'$\Omega_m$', color='blue', linewidth=2)
plt.loglog(z, Omega_L, label=r'$\Omega_\Lambda$', color='green', linewidth=2)

plt.axvline(z_eq, color='red', linestyle='--', linewidth=2, label=f'$\Omega_r$-$\Omega_m$ - equality (z={z_eq:.0f})')
plt.axvline(z_de, color='black', linestyle='--', linewidth=2, label=f'$\Omega_m$-$\Omega_\Lambda$ - equality (z={z_de:.2f})')

plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel(r'$z$', fontsize=20)
plt.ylabel(r'$\Omega_i(z)$', fontsize=20)
plt.title('$\Omega_i$ as a Function of $z$', fontsize=18, pad=15)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=12)

plt.grid(True, which='both', ls='--', alpha=0.5)

plt.legend(fontsize=12, loc='lower right', frameon=True)
plt.xlim(10**5, 10**(-3))
plt.ylim(10**(-9), 5)

plt.tight_layout()
plt.show()

# %%

'''To generate fig. 4.1, typical P(k) from class'''

from classy import Class
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import scipy.interpolate as scint
from scipy.interpolate import interp1d
import scipy as ss
from scipy.integrate import solve_ivp

z = 0 # redshift
h = 0.6736 # Hubble rate 
H0 = 100 * h 

def get_power_spectrum(z, 
                       k=np.logspace(-4, 0, 500), 
                       returnk=False, 
                       plot=False): 
    params = {
        'output': 'mPk',
        'H0': 67.36,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'z_pk': z,
        'P_k_max_h/Mpc': 10,
        'non linear': 'none',
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    k_vals = k.flatten() 
    Pk = np.array([cosmo.pk(ki, z) for ki in k_vals]) 

    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.loglog(k_vals, Pk, lw=2, color='navy') 

        plt.xlabel(r'$k\ \left[h/\mathrm{Mpc}\right]$', fontsize=16)
        plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc}/h)^3\right]$', fontsize=16)
        plt.title(f"Power Spectrum from CLASS at z = {params['z_pk']}", fontsize=16)

        plt.tick_params(axis='both', which='major', labelsize=16)  
        plt.tick_params(axis='both', which='minor', labelsize=16)

        plt.grid(True, which='both', ls='--', alpha=0.7)
        plt.legend(fontsize=14)
        plt.tight_layout()
    
    cosmo.struct_cleanup()

    if returnk == True: 
        return k_vals, Pk 
    else:
        return Pk
    
get_power_spectrum(z, plot=True)

# %%

'''To generate fig. 5.1, comparison of growth factors and 
approximation and analytical hypergeometric solution'''

from classy import Class
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import scipy.interpolate as scint
from scipy.interpolate import interp1d
import scipy as ss
from scipy.integrate import solve_ivp
import matplotlib.ticker as ticker

z = 0
h = 0.6736
H0 = 100 * h
omega_b = 0.02237
omega_cdm = 0.12
Omega_M = (omega_b + omega_cdm) / h**2
Omega_L = 1.0 - Omega_M

cosmo = Class()
cosmo.set({
    'h': h,
    'Omega_b': omega_b,
    'Omega_cdm': omega_cdm,
    'output': 'mTk',
    'z_max_pk': 100
})
cosmo.compute()
cosmo.struct_cleanup()
cosmo.empty()

# analytical hypergeometric solution
u = lambda a: - (Omega_L / Omega_M)  * a**3 #AARON
F_an = lambda u:  ss.special.hyp2f1(1/3, 1, 11/6, u)  

def event_stop(t, y): 
    return y[0] - 1   

event_stop.terminal = True 
event_stop.direction = 0   

def dydt(t, y): 
    a, D, Ddot, D2, D2dot = y

    E_a = np.sqrt(Omega_L + (Omega_M / a**3))
    adot = a**2 * H0 * E_a

    Ddotdot = - (1/a) * adot * Ddot + (3/2) * Omega_M * H0**2/a * D
    D2dotdot = - (1/a) * adot * D2dot + (3/2) * Omega_M * H0**2/a * (D2 + D**2)

    return [adot, Ddot, Ddotdot, D2dot, D2dotdot]

initial = [1e-9, 1e-9, 0.0, 0.0, 0.0]

t_span = (0, 0.5) 
t_eval = np.linspace(*t_span, 100000)

sol = solve_ivp(dydt, 
                t_span=t_span, 
                y0=initial, 
                t_eval=t_eval,
                rtol=1e-10,
                events=event_stop,
                method='BDF')

aval = sol.y[0]
D1 = sol.y[1]
D2 = sol.y[3]

D1n = D1 / D1[-1]
D2n = -D2 / D1[-1]**2

# a values from 1e-8 to 1 (today) in 10.000 steps as list 
avals = np.linspace(1e-9, 1, 100000).tolist() #AARON
D1_class = np.array([cosmo.scale_independent_growth_factor(1/i - 1) for i in aval])

D1_interp = interp1d(aval, D1n, kind='cubic', bounds_error=False, fill_value='extrapolate')
D2_interp = interp1d(aval, D2n, kind='cubic', bounds_error=False, fill_value='extrapolate')
  
def get_D1D2_at_z(z): # Function that returns D1 and D2 for specific z
    a_eval = 1 / (1 + z)
    D1z = D1_interp(a_eval)
    D2z = D2_interp(a_eval)
    return D1z, D2z

D1z, D2z = get_D1D2_at_z(z) # obtain D1(z), D2(z)
D2_approx = - 3/7 * D1n**2 * Omega_M**(-1/143) 

u = lambda a: -(Omega_L / Omega_M)*a**3 # AARON
F_hyp = lambda u: ss.special.hyp2f1(1/3, 1, 11/6, u) # AARON
D1_hyp = aval * F_hyp(u(aval)) / F_hyp(u(1)) # AARON

z_vals = 1 / aval - 1

'''plotting normalized D1 and D2 with hypergeometric solution and approximations'''

plt.figure(figsize=(12, 6))
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

axes[0].loglog(z_vals, D1_class, color='k', linestyle='-', linewidth=2, label=r'$D_1$ (CLASS)')
axes[0].loglog(z_vals, D1n, color='r', linestyle='--', linewidth=2, label=r'$D_1$ (ODE)')
axes[0].loglog(z_vals, D1_hyp, color='g', linestyle='-', linewidth=2, label=r'$D_1$ (hyp2f1)')
axes[0].set_xlabel(r'redshift $z$', fontsize=18)
axes[0].set_ylabel(r'Magnitude of $D_1$', fontsize=18)
axes[0].set_title('Growth Factor $D_1$', fontsize=22, pad=15)
axes[0].grid(True, which='both', linestyle='--', alpha=0.6)
axes[0].legend(fontsize=16, loc='upper right')
axes[0].tick_params(axis='both', which='major', labelsize=17)
axes[0].tick_params(axis='both', which='minor', labelsize=17)

axes[1].loglog(z_vals, -D2_interp(aval), color='b', linestyle=':', linewidth=2, label=r'$D_2$ (ODE)')
axes[1].loglog(z_vals, -D2_approx, color='purple', marker='s', linestyle='-', markersize=2, label=r'$D_2 \approx -\frac{3}{7}D_1^2\Omega^{-1/143}$')
axes[1].set_xlabel(r'redshift $z$', fontsize=18)
axes[1].set_ylabel(r'Magnitude of $D_2$', fontsize=18)
axes[1].set_title('Growth Factor $D_2$', fontsize=22, pad=15)
axes[1].grid(True, which='both', linestyle='--', alpha=0.6)
axes[1].legend(fontsize=16, loc='upper right')
axes[1].tick_params(axis='both', which='major', labelsize=17)
axes[1].tick_params(axis='both', which='minor', labelsize=17)

plt.show()

#%%

'''To generate fig. 5.5, for visualization of 2D-slice of the density 
contrast of 1LPT and 2LPT modified with GitHub Copilot Chat'''

from classy import Class
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import scipy.interpolate as scint
from scipy.interpolate import interp1d
import scipy as ss
from scipy.integrate import solve_ivp

z = 0 # redshift
h = 0.6736 # Hubble rate
L = 500.0 # length of the box SI: [Mpc/h]
N = 64 # sum of grid cells (corners)
cell_size = L / N # size of one cell 
H0 = 100 * h # Hubble constant
V = L**3 # Volume of box
fac = np.sqrt(2 * np.pi / V) * (N**3) # Normalization to match CLASS
print(f'FOR z = {z}: ')

def get_power_spectrum(z, 
                       k=np.logspace(-4, 0, 500), 
                       returnk=False, 
                       plot=False): 
    params = {
        'output': 'mPk',
        'H0': 67.36,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'z_pk': z,
        'P_k_max_h/Mpc': 10,
        'non linear': 'none',
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    k_vals = k.flatten() 
    Pk = np.array([cosmo.pk(ki, z) for ki in k_vals]) 

    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.loglog(k_vals, Pk, lw=1, color='navy') 

        plt.xlabel(r'$k\ \left[h/\mathrm{Mpc}\right]$', fontsize=14)
        plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc}/h)^3\right]$', fontsize=14)
        plt.title(f"Linear Matter Power Spectrum at z = {params['z_pk']}", fontsize=16)

        plt.grid(True, which='both', ls='--', alpha=0.7)
        plt.tight_layout()
    
    cosmo.struct_cleanup()

    if returnk == True: 
        return k_vals, Pk 
    else:
        return Pk

k_class, P_k_class = get_power_spectrum(z, returnk=True)
valid_indices = P_k_class >= 0

Pk_values_cleaned = P_k_class[valid_indices]
k_values_cleaned = k_class[valid_indices]

Pk_interp = interp1d(k_values_cleaned, 
                     Pk_values_cleaned,
                     kind='cubic',  
                     bounds_error=False, 
                     fill_value = "extrapolate")
 
real_part = np.random.normal(0, 1/np.sqrt(2), size=(N,N,N)) 
imag_part = np.random.normal(0, 1/np.sqrt(2), size=(N,N,N))
R = (real_part + 1j * imag_part)  

for i in range(N):
    for j in range(N):
        for k in range(N):
            sym_i, sym_j, sym_k = (-i) % N, (-j) % N, (-k) % N
            R[sym_i,sym_j,sym_k] = np.conj(R[i,j,k])

if N % 2 == 0:
    R[N//2, :, :] = R[N//2, :, :].real
    R[:, N//2, :] = R[:, N//2, :].real
    R[:, :, N//2] = R[:, :, N//2].real

R[0, 0, 0] = R[0, 0, 0].real

k_vals = 2*np.pi*np.fft.fftfreq(N, d=L/N)
kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
k_mag[k_mag == 0] = 1e-10

delta_k = np.sqrt(Pk_interp(k_mag)) * R * fac
delta_k[0, 0, 0] = 0

delta_x = np.fft.ifftn(delta_k).real 

psi_kx = 1j * kx / (k_mag**2) * delta_k 
psi_ky = 1j * ky / (k_mag**2) * delta_k
psi_kz = 1j * kz / (k_mag**2) * delta_k 

psi_x = np.fft.ifftn(psi_kx).real 
psi_y = np.fft.ifftn(psi_ky).real
psi_z = np.fft.ifftn(psi_kz).real 

omega_b = 0.02237
omega_cdm = 0.12
Omega_M = (omega_b + omega_cdm) / h**2
Omega_L = 1.0 - Omega_M

cosmo = Class()
cosmo.set({
    'h': h,
    'Omega_b': omega_b,
    'Omega_cdm': omega_cdm,
    'output': 'mTk',
    'z_max_pk': 100
})
cosmo.compute()
cosmo.struct_cleanup()
cosmo.empty()

u = lambda a: - (Omega_L / Omega_M)  * a**3
F_an = lambda u:  ss.special.hyp2f1(1/3, 1, 11/6, u)  

def event_stop(t, y): 
    return y[0] - 1   

event_stop.terminal = True 
event_stop.direction = 0   

def dydt(t, y):
    a, D, Ddot, D2, D2dot = y

    E_a = np.sqrt(Omega_L + (Omega_M / a**3))
    adot = a**2 * H0 * E_a

    Ddotdot = - (1/a) * adot * Ddot + (3/2) * Omega_M * H0**2/a * D
    D2dotdot = - (1/a) * adot * D2dot + (3/2) * Omega_M * H0**2/a * (D2 + D**2)

    return [adot, Ddot, Ddotdot, D2dot, D2dotdot]

initial = [1e-9, 1e-9, 0.0, 0.0, 0.0]

t_span = (0, 0.5) 
t_eval = np.linspace(*t_span, 100000)

sol = solve_ivp(dydt, 
                t_span=t_span, 
                y0=initial, 
                t_eval=t_eval,
                rtol=1e-10,
                events=event_stop,
                method='BDF')

aval = sol.y[0]
D1 = sol.y[1]
D2 = sol.y[3]

D1n = D1 / D1[-1]
D2n = -D2 / D1[-1]**2

avals = np.linspace(1e-9, 1, 100000).tolist()
D1_class = np.array([cosmo.scale_independent_growth_factor(1/i - 1) for i in aval])

D1_interp = interp1d(aval, D1n, kind='cubic', bounds_error=False, fill_value='extrapolate')
D2_interp = interp1d(aval, D2n, kind='cubic', bounds_error=False, fill_value='extrapolate')
  
def get_D1D2_at_z(z): 
    a_eval = 1 / (1 + z)
    D1z = D1_interp(a_eval)
    D2z = D2_interp(a_eval)
    return D1z, D2z

D1z, D2z = get_D1D2_at_z(z) 
D2_approx = - 3/7 * D1n**2 * Omega_M**(-1/143) 

u = lambda a: -(Omega_L / Omega_M)*a**3
F_hyp = lambda u: ss.special.hyp2f1(1/3, 1, 11/6, u)
D1_hyp = aval * F_hyp(u(aval)) / F_hyp(u(1))

z_vals = 1 / aval - 1

def grad(f, ks):
   return np.fft.ifftn(1j * ks * np.fft.fftn(f)).real 

phi1_k = -delta_k / (k_mag**2)
phi1_k[0, 0, 0] = 0.0
phi1 = np.fft.ifftn(phi1_k).real

Phi_xx = grad(grad(phi1, kx), kx)
Phi_yy = grad(grad(phi1, ky), ky)
Phi_zz = grad(grad(phi1, kz), kz)

Phi_xy = grad(grad(phi1, kx), ky)
Phi_xz = grad(grad(phi1, kx), kz)
Phi_yz = grad(grad(phi1, ky), kz)

S = (Phi_xx * Phi_yy - Phi_xy**2 +
     Phi_xx * Phi_zz - Phi_xz**2 +
     Phi_yy * Phi_zz - Phi_yz**2)

S_k = np.fft.fftn(S) 
S_k[0, 0, 0] = 0.0

psi2_kx = -1j * kx / (k_mag**2) * S_k 
psi2_ky = -1j * ky / (k_mag**2) * S_k
psi2_kz = -1j * kz / (k_mag**2) * S_k

psi2_x = np.fft.ifftn(psi2_kx).real 
psi2_y = np.fft.ifftn(psi2_ky).real 
psi2_z = np.fft.ifftn(psi2_kz).real 

psi_2_x = psi_x + D2z * psi2_x 
psi_2_y = psi_y + D2z * psi2_y
psi_2_z = psi_z + D2z * psi2_z

def cic_interpolation(xd, yd, zd, Psi_x, Psi_y, Psi_z, N, L):
    
    X = ((xd + Psi_x.real.flatten()) / (L/N)) % N
    Y = ((yd + Psi_y.real.flatten()) / (L/N)) % N
    Z = ((zd + Psi_z.real.flatten()) / (L/N)) % N

    mass_grid = np.zeros((N, N, N)) 

    i_corner = np.floor(X).astype(int) % N
    j_corner = np.floor(Y).astype(int) % N
    k_corner = np.floor(Z).astype(int) % N

    x_off = X - np.floor(X)
    y_off = Y - np.floor(Y)
    z_off = Z - np.floor(Z)

    for ox, oy, oz in [(0,0,0), (0,0,1), (0,1,0), (0,1,1),
                   (1,0,0), (1,0,1), (1,1,0), (1,1,1)]:

        wx = (1 - x_off) if ox == 0 else x_off
        wy = (1 - y_off) if oy == 0 else y_off
        wz = (1 - z_off) if oz == 0 else z_off
        weight = wx * wy * wz 
    
        i_s = (i_corner + ox) % N
        j_s = (j_corner + oy) % N
        k_s = (k_corner + oz) % N
    
        np.add.at(mass_grid, (i_s, j_s, k_s), weight) 

    return mass_grid

pos = np.linspace(0, L, N, endpoint = False)
xs, ys, zs = np.meshgrid(pos, pos, pos, indexing = 'ij')

xs = xs.flatten()
ys = ys.flatten()
zs = zs.flatten()

cic1 = cic_interpolation(xs, ys, zs, psi_x, psi_y, psi_z, N, L)
delta_cic1 = cic1 - 1 

cic2 = cic_interpolation(xs, ys, zs, psi_2_x, psi_2_y, psi_2_z, N, L)
delta_cic2 = cic2 - 1 

vmax = max(abs(delta_cic1).max(), abs(delta_cic2).max())
vmin = -vmax

#GitHub Copilot assisted in making the plot:

import matplotlib.patches as patches

plt.figure(figsize=(12, 5))

ax1 = plt.subplot(121)
im1 = ax1.imshow(delta_cic1[:, :, N//2], cmap='seismic', vmin=vmin, vmax=vmax)
ax1.set_title(f'1LPT CIC density contrast (z={z})', fontsize=17)
ax1.tick_params(axis='both', which='major', labelsize=16)  
ax1.tick_params(axis='both', which='minor', labelsize=16)

slice_idx = N // 2
max_in_slice_cic1 = np.max(delta_cic1[:, :, slice_idx])
max_idx_in_slice_cic1 = np.unravel_index(np.argmax(delta_cic1[:, :, slice_idx]), delta_cic1[:, :, slice_idx].shape)
max_idx_in_slice_cic1 = tuple(map(int, max_idx_in_slice_cic1))

ax1.set_xlabel(f"max. $\delta$-contrast: {max_in_slice_cic1:.2f}\nposition in slice: {max_idx_in_slice_cic1}", fontsize=16, labelpad=10)
ax1.set_xticks([10, 20, 30, 40, 50, 60])
ax1.set_xticklabels([f"{tick}" for tick in [10, 20, 30, 40, 50, 60]])

# rectangle that indicates max density contrast in slice
rect1 = patches.Rectangle(
    (max_idx_in_slice_cic1[1] - 6, max_idx_in_slice_cic1[0] - 6), 
    12, 12,  
    linewidth=2.5, edgecolor='black', facecolor='none'
)
ax1.add_patch(rect1)

ax2 = plt.subplot(122)
im2 = ax2.imshow(delta_cic2[:, :, N//2], cmap='seismic', vmin=vmin, vmax=vmax)
cbar = plt.colorbar(im2, ax=ax2, label='Density contrast')
cbar.set_label('Density contrast', fontsize=17)
cbar.ax.tick_params(labelsize=17)

ax2.set_title(f'2LPT CIC density contrast (z={z})', fontsize=17)
ax2.tick_params(axis='both', which='major', labelsize=16)  
ax2.tick_params(axis='both', which='minor', labelsize=16)

max_in_slice_cic2 = np.max(delta_cic2[:, :, slice_idx])
max_idx_in_slice_cic2 = np.unravel_index(np.argmax(delta_cic2[:, :, slice_idx]), delta_cic2[:, :, slice_idx].shape)
max_idx_in_slice_cic2 = tuple(map(int, max_idx_in_slice_cic2))

ax2.set_xlabel(f"max. $\delta$-contrast: {max_in_slice_cic2:.2f}\nposition in slice: {max_idx_in_slice_cic2}", fontsize=16, labelpad=10)
ax2.set_xticks([10, 20, 30, 40, 50, 60])
ax2.set_xticklabels([f"{tick}" for tick in [10, 20, 30, 40, 50, 60]])

rect2 = patches.Rectangle(
    (max_idx_in_slice_cic2[1] - 6, max_idx_in_slice_cic2[0] - 6),  
    12, 12,  
    linewidth=2.5, edgecolor='black', facecolor='none'
)
ax2.add_patch(rect2)

plt.tight_layout()
plt.show()
#%%

'''End of the implementation.'''