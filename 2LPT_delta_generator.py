'''This file was used by Simon Schmidt Thomsen
to generate 100 samples of 2LPT density contrast
fields, with the corresponding A_s value. This 
was done in using parts of the file 
"LPT-implementation.py". The data was plotted 
using the file "network_A_s.py".'''


"""Imports & setting up træningdata"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*layer.add_variable.*")          #Tror ikke umiddelbart de her warnings er noget der kan gøres noget ved
warnings.filterwarnings("ignore", message=".*RandomNormal is unseeded.*")         #Tror ikke umiddelbart de her warnings er noget der kan gøres noget ved, måske der kan ved den her?
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                   #Fjerner en notice om noget numerisk precision or smth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'                    #Fjerner notice om at CPU bliver brugt til at optimize stuff, kan måske fjerne relevante ting også not sure so be careful
import numpy as np
import matplotlib.pyplot as plt
import time
from classy import Class
import psutil
import gc
import scipy.stats as ss
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
plt.rc("axes", labelsize=30, titlesize=32)   # skriftstørrelse af xlabel, ylabel og title
plt.rc("xtick", labelsize=26, top=True, direction="in")  # skriftstørrelse af ticks, vis også ticks øverst og vend ticks indad
plt.rc("ytick", labelsize=26, right=True, direction="in") # samme som ovenstående
plt.rc("legend", fontsize=30) # skriftstørrelse af figurers legends
plt.rcParams["font.size"] = "20"
plt.rcParams["figure.figsize"] = (16,9)


def createR(N, dim = 3, mu = 0, sig = 1/np.sqrt(2)):
    """Funktion til at lave R
    args:
    ----------
    N : int
        Sidelængde for kassen
    """
    size = [N] * dim
    c_1 = np.random.normal(mu, sig, size = size)
    c_2 = np.random.normal(mu, sig, size = size)

    R = c_1 + 1j * c_2

    """Enforcer hermitisk symmetri"""
    R[0,0,0] = R[0,0,0].real     
    for i in np.arange(N):
        for j in np.arange(N):
            for k in np.arange(N):
                # R[i,j,k] = np.conj(R[N - i - 1, N - j - 1, N - k - 1])      #forstår ummidelbart ikke hvorfor den her line ikke virker, burde gøre det samme som de to næste samlet no?
                ii, jj, kk = -i % N, -j % N, -k % N
                # if (i,j,k) != (ii,jj,kk):                                     #Ikke sikker på det her overhovedet sparer tid. hvertfald ikke meget hvis det gør
                R[ii, jj, kk] = np.conj(R[i, j, k])
   
    return R

def power_spectrum(A_s = 2.1e-9, n_s = 0.9649, omega_b = 0.02237, omega_cdm = 0.12, z = 50, k = np.logspace(-4, 0, 500), plot = False, returnk = False):
    """In ChatGPT we trust"""
    # Set cosmological parameters (ΛCDM model with Planck 2018 values)
    params = {
        'output': 'mPk',          # Output matter power spectrum
        'H0': 67.36,              # Hubble parameter [km/s/Mpc]
        'omega_b': omega_b,       # Baryon density
        'omega_cdm': omega_cdm,        # Cold Dark Matter density
        'Omega_k': 0.0,           # Spatial curvature (flat universe)
        'n_s': n_s,            # Scalar spectral index
        'A_s': A_s,            # Primordial curvature power
        'z_pk': z,              # Redshift for power spectrum calculation
        'non linear': 'none',     # Linear power spectrum (set to True for nonlinear)
        'z_max_pk': 1000            #gør ingenting me thinks, vist bare max for z_pk
    }

    # Initialize CLASS and compute the cosmology
    k_max = np.max(k_mag)
    params['P_k_max_1/Mpc'] = k_max * 1.05   # 5% safety margin, prøver at prevent error med for stor N/L ratio

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    #Gør k 1D
    k = k.flatten()

    # Calculate P(k) at z 
    Pk = np.array([cosmo.pk(ki, params['z_pk']) for ki in k])
    # Pk = np.array([cosmo.pk(ki, params['z_pk']) if k > 0 else 0 for ki in k])

    # Plotting
    if plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(k, Pk, lw=2, color='navy')

        plt.xlabel(r'$k\ \left[h/\mathrm{Mpc}\right]$', fontsize=14)
        plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc}/h)^3\right]$', fontsize=14)
        plt.title(f'Linear Matter Power Spectrum at z = {params['z_pk']}', fontsize=16)
        plt.grid(True, which='both', ls='--', alpha=0.7)
        plt.xlim(1e-4, 1e0)
        plt.tight_layout()

    # Clean up CLASS instance
    # cosmo.struct_cleanup()
    cosmo.struct_cleanup()
    # cosmo.empty()
    # del cosmo

    if returnk == True:
        return k, Pk#, Hz, D_z
    else:
        return Pk

def power_spectrum(z, A_s,
                       k=np.logspace(-4, 0, 500), 
                       returnk=False, 
                       plot=False): 
    params = {
        'output': 'mPk',
        'H0': 67.36,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'n_s': 0.9649,
        'A_s': A_s,
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
        # plt.xlim(k[1:].min(), k.max())
        plt.xlim(5e-3, 5e-1) ; plt.ylim(1e2, 5e5)
    
    cosmo.struct_cleanup()

    if returnk == True: 
        return k_vals, Pk 
    else:
        return Pk
    
h = 0.6736 # Hubble rate
L = 1000 # length of the box SI: [Mpc/h]
N = 32 # sum of grid cells (corners)
cell_size = L / N # size of one cell 
H0 = 100 * h 
V = L**3 
fac = np.sqrt(2 * np.pi / V) * (N**3) 
k_quist = np.pi * N / L
omega_b = 0.02237
omega_cdm = 0.12
Omega_M = (omega_b + omega_cdm) / h**2
Omega_L = 1.0 - Omega_M

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

D1_interp = interp1d(aval, D1n, kind='cubic', bounds_error=False, fill_value='extrapolate')
D2_interp = interp1d(aval, D2n, kind='cubic', bounds_error=False, fill_value='extrapolate')
  
def get_D1D2_at_z(z): 
    a_eval = 1 / (1 + z)
    D1z = D1_interp(a_eval)
    D2z = D2_interp(a_eval)
    return D1z, D2z

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

def grad(f, ks):
   return np.fft.ifftn(1j * ks * np.fft.fftn(f)).real 

def find_delta(z, L, N, A_s = 2.1e-9):

    D1z, D2z = get_D1D2_at_z(z)
    
    k_vals, Pk = power_spectrum(z, A_s, returnk=True, plot= False)
    Pk_interp = interp1d(k_vals, Pk, kind='cubic', bounds_error=False, fill_value="extrapolate")

    real_part = np.random.normal(0, 1/np.sqrt(2), size=(N, N, N))
    imag_part = np.random.normal(0, 1/np.sqrt(2), size=(N, N, N))
    R = real_part + 1j * imag_part

    for i in range(N):
        for j in range(N):
            for k in range(N):
                sym_i, sym_j, sym_k = (-i) % N, (-j) % N, (-k) % N
                R[sym_i, sym_j, sym_k] = np.conj(R[i, j, k])

    if N % 2 == 0:
        R[N//2, :, :] = R[N//2, :, :].real
        R[:, N//2, :] = R[:, N//2, :].real
        R[:, :, N//2] = R[:, :, N//2].real

    R[0, 0, 0] = R[0, 0, 0].real

    k_vals = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[k_mag == 0] = 1e-10

    delta_k = np.sqrt(Pk_interp(k_mag)) * R * fac
    delta_k[0, 0, 0] = 0

    psi_kx = 1j * kx / (k_mag**2) * delta_k 
    psi_ky = 1j * ky / (k_mag**2) * delta_k
    psi_kz = 1j * kz / (k_mag**2) * delta_k 

    psi_x = np.fft.ifftn(psi_kx).real 
    psi_y = np.fft.ifftn(psi_ky).real
    psi_z = np.fft.ifftn(psi_kz).real 

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

    pos = np.linspace(0, L, N, endpoint = False)
    xs, ys, zs = np.meshgrid(pos, pos, pos, indexing = 'ij')

    xs = xs.flatten()
    ys = ys.flatten()
    zs = zs.flatten()

    cic2 = cic_interpolation(xs, ys, zs, psi_2_x, psi_2_y, psi_2_z, N, L)
    delta_cic2 = cic2 - 1

    return delta_cic2




def makeFolders(savepath):
    """Laver folders til at gemme data i"""
    files = ('', 'Training & val data', 'Test data')

    for file in files:
        if not os.path.exists(savepath + file):
            os.mkdir(savepath + file)
            print(f"Folder '{savepath + file}' created.")
        else:
            print(f"Folder '{savepath + file}' already exists.")

def saveDelta(i,z, L, N, savepath, A_s = 2.1e-9, n_s = 0.9649, omega_cdm = 0.12, N_samples = 5000, overWrite = False, omega_b = 0.02237, TrainingNoise = False):
    # cumtime = 0
    if not overWrite:
        if os.path.exists(savepath + f'Training & val data/delta_train_id-{i + 1}.npy'):                      #Hvis filen allerede eksisterer så skip til næste, gør at der kan laves store sæt over flere gange
            return                                                                                            #Bare for at stoppe den, returner none though men vidst ikke en bedre måde at gøre det på
    # temptime = time.time()

    delta = find_delta(z, L, N, A_s = A_s)
    if TrainingNoise:                                                                                   #Mulighed for at tilføje noise til træningssættet, "if" lader ikke til at slow det down (like 0.5 sec max for 10 mil loops)
        delta += np.random.normal(0, 0.1 * np.max(delta), (N, N, N))
    np.save(savepath + f'Training & val data/delta_train_id-{i + 1}', delta)                          #Gemmer dem med unik ID for hver fil, burde gøre at man kan gemme så mange der er plads til på hardisken i stedet for at være limited af ram :)
    

    # del delta
    # gc.collect()
    # if i % 10 == 0 and i != 0:                                                                      #Bare for at kunne se progress på dem der tager lidt længere tid
    #     cumtime += (time.time() - temptime) * 10                                                        #Kan godt flyttes udenfor for at blive mere præcis i guess (og så ikke gange med 10) men er bare et estimat anyways
    #     print(f'Progress: {np.round((i / N_samples) * 100, 4)} % Expected remaining time: {round(cumtime/i * (N_samples - i), 2)} s')


def createData(N_samples, z, L, N, ValSize = 0.2, A_s_min = 2.1e-9, A_s_max = 2.1e-9, n_s_min = 0.9649, n_s_max = 0.9649, omega_cdm_min = 0.12, omega_cdm_max = 0.12, omega_b = 0.02237, TestData = False, TrainingNoise = False, savepath = None, SameTrain = False, paramRepeat = 1, overWrite = False):
    """Function to create data for training and sampling
    """
    """Sætter op de potentiele variable, hopefully foolproof til så at kunne implementere de andre senere, ved ikke om det måske kunne laves pænere med params dictionary eller noget - kig på det hvis tid"""
    if A_s_max != A_s_min:
        np.random.seed(420) ; A_s_train = np.random.uniform(A_s_min, A_s_max, size = int(N_samples * (1 - ValSize)))                     #Sætter random seed for at kunne fortsætte kode
        np.random.seed(7) ; A_s_val = np.random.uniform(A_s_min, A_s_max, size = int(N_samples * ValSize))                              #sætter nyt random seed for at validation data ikke bare svarer til den første del af træningsdataet (og 7 er det mest tilfældige tal)
    else:
        A_s_train = [A_s_max] * int(N_samples * (1 - ValSize))
        A_s_val = [A_s_max] * int(N_samples * ValSize)

    if n_s_max != n_s_min:
        np.random.seed(420) ; n_s_train = np.random.uniform(n_s_min, n_s_max, size = int(N_samples * (1 - ValSize)))
        np.random.seed(7) ; n_s_val = np.random.uniform(n_s_min, n_s_max, size = int(N_samples * ValSize))
    else:
        n_s_train = [n_s_max] * int(N_samples * (1 - ValSize))
        n_s_val = [n_s_max] * int(N_samples * ValSize)

    if omega_cdm_max != omega_cdm_min:
        np.random.seed(420) ; omega_cdm_train = np.random.uniform(omega_cdm_min, omega_cdm_max, size = int(N_samples * (1 - ValSize)))
        np.random.seed(7) ; omega_cdm_val = np.random.uniform(omega_cdm_min, omega_cdm_max, size = int(N_samples * ValSize))
    else:
        omega_cdm_train = [omega_cdm_max] * int(N_samples * (1 - ValSize))
        omega_cdm_val = [omega_cdm_max] * int(N_samples * ValSize)


                                       #Har unik A,n,omega for hver delta (fastholder ikke nogen af dem )
    

    A_s_train = np.repeat(A_s_train, paramRepeat)               ; A_s_val = np.repeat(A_s_val, paramRepeat)             #Hvis man vil lavere flere deltasæt for hver sæt af parametre. np.repeat er noget hurtigere end manuel list comprehension
    n_s_train = np.repeat(n_s_train, paramRepeat)               ; n_s_val = np.repeat(n_s_val, paramRepeat)
    omega_cdm_train = np.repeat(omega_cdm_train, paramRepeat)   ; omega_cdm_val = np.repeat(omega_cdm_val, paramRepeat)


    if savepath is not None and TestData == False:                                                          #Gemmer parametrene brugt til at lave træningssæt, gemmes bare i en array da det ikke fylder særlig meget alligevel og det gør det nemmere at sortere ting senere
        with open(savepath + 'TrainingParams.txt', mode = 'w') as f:                                        #Kunne godt bruge np.save men havde allerede lavet det her da jeg fandt ud af at den eksisterede so here we are
            f.write('A_s \t n_s \t omega_cdm \n') 
            for nrA, A in enumerate(A_s_train):
                f.write(f'{A} \t {n_s_train[nrA]} \t {omega_cdm_train[nrA]} \n')   


        with open(savepath + 'ValParams.txt', mode = 'w') as f:                                             #Samme som ovenfor bare med validation parametre
            f.write('A_s \t n_s \t omega_cdm \n') 
            for nrA, A in enumerate(A_s_val):
                f.write(f'{A} \t {n_s_val[nrA]} \t {omega_cdm_val[nrA]} \n')   

    np.random.seed()                                                                                                    #Resetter random seed for at R-felterne er random

    Parallel(n_jobs=-1, verbose = 10)(delayed(saveDelta)(i, z, L, N, savepath, A_s_train[i], n_s_train[i], omega_cdm_train[i], N_samples, overWrite)            #Parallelisering af at lave data, tager for some reason arguments i en seperat parantes, n_jobs = -1 betyder brug alle cores
        for i in range(len(A_s_train)))                                                                                                              #https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html
                                                                                                                                                     #Lader til at batch_size = 1 er det den oftest lander på, kunne overveje at fastsætte den der for at den ikke springer så meget når man resumer
    Parallel(n_jobs=-1, verbose = 10)(delayed(saveDelta)(i + len(A_s_train), z, L, N, savepath, A_s_val[i], n_s_val[i], omega_cdm_val[i], N_samples, overWrite) #same men testdata
        for i in range(len(A_s_val)))

    """Parallel er omkring ~1.7 gange hurtigere (706 kontra 417 sek). For 4 cores er bedst teoretisk speedup apparently 2.44x (Amdahl's law) hvis 90% af koden er perfekt paralleliseret -hvilket den ikke er da der også er overhead & stuff
    Er desuden noget pickling (???????????????????? er det navn xd) der tager noget tid.
    https://chatgpt.com/c/68091c21-cfcc-8009-b661-41f9ff506d7e   for nogle tips der måske kan speede det lidt mere op - at udkommentere manuel del delta og gc lader til at speede lidt up
    Måske batch sizes men standard er auto så tror det er fine? 
    Dask måske bedre end joblib for store opgaver
    """

def createTestData(N_samples, z, L, N, A_s_min = 2.1e-9, A_s_max = 2.1e-9, n_s_min = 0.9649, n_s_max = 0.9649, omega_cdm_min = 0.12, omega_cdm_max = 0.12, omega_b = 0.02237, TestData = False, TrainingNoise = False, savepath = None, SameTrain = False):
    """Function to create data for training and sampling
    """
    """Sætter op de potentiele variable, hopefully foolproof til så at kunne implementere de andre senere, ved ikke om det måske kunne laves pænere med params dictionary eller noget - kig på det hvis tid"""
    np.random.seed(39)                          #Vil ikke have at testdata har samme seed som træningsdata
    if A_s_max != A_s_min:
        A_s_train = np.random.uniform(A_s_min, A_s_max, size = N_samples)
    else:
        A_s_train = [A_s_max] * N_samples

    if n_s_max != n_s_min:
        n_s_train = np.random.uniform(n_s_min, n_s_max, size = N_samples)
    else:
        n_s_train = [n_s_max] * N_samples

    if omega_cdm_max != omega_cdm_min:
        omega_cdm_train = np.random.uniform(omega_cdm_min, omega_cdm_max, size = N_samples)
    else:
        omega_cdm_train = [omega_cdm_max] * N_samples

    np.random.seed()                                                                            #Resetter random seed for at R-felterne er random

    cumtime = 0
    for nrA, (A, n, o) in enumerate(zip(A_s_train, n_s_train, omega_cdm_train)):
        temptime = time.time()
        delta = find_delta(z, L, N, A_s = A)
        if TrainingNoise:                                                                       #Mulighed for at tilføje noise til træningssættet, "if" lader ikke til at slow det down (like 0.5 sec max for 10 mil loops)
            delta += np.random.normal(0, 0.1 * np.max(delta), (N, N))

        np.save(savepath + f'Test data/delta_test_id-{nrA + 1}', delta)                         #Gemmer dem med unik ID for hver fil, burde gøre at man kan gemme så mange der er plads til på hardisken i stedet for at være limited af ram :)

        if nrA % 10 == 0 and nrA != 0:                                                          #Bare for at kunne se progress på dem der tager lidt længere tid
            cumtime += (time.time() - temptime) * 10                                            #Kan godt flyttes udenfor for at blive mere præcis i guess (og så ikke gange med 10) men er bare et estimat anyways
            print(f'Progress: {np.round((nrA / N_samples) * 100, 4)} % Expected remaining time: {round(cumtime/nrA * (N_samples - nrA), 2)} s')
            # mem = psutil.virtual_memory()
            # print(f"Memory usage: {mem.percent}% used")

    with open(savepath + 'TestParams.txt', mode = 'w') as f:                                             #Samme som ovenfor bare med validation parametre
        f.write('A_s \t n_s \t omega_cdm \n') 
        for nrA, A in enumerate(A_s_train):
            f.write(f'{A} \t {n_s_train[nrA]} \t {omega_cdm_train[nrA]} \n')   

def power_spectrum_from_density(delta, k_mag, nbins = 50, plot = True, fromcic = False, kx = 0, ky = 0, kz = 0, returnP = False):
    """Funktion til at finde power spectrum fra en given densitet i reelt rum
    args:
    -------------
    delta : N-dimensional array
        delta der bruges til at finde P(k)

    k_mag : N-dimensional array
        Punkter i fourierrum hvor delta evalueres. (Overvej at gøre så den selv laver k_mag fra L & N, er mere independant men tager længere tid når det alligevel allerede gøres...)

    nbins : int
        Antal bins power-spektret skal inddeles i
    
    plot : bool
        True plotter fundet spektrum
    """
    fac = np.sqrt(2*np.pi/L**3) * N**3
    delta_k = np.fft.fftn(delta) * fac**(-1)
    p_k_vec = delta_k * np.conj(delta_k)                                                                        #Tager normkvadratet

    """Correcter power spektret hvis det kommer fra cloud in cell"""
    if fromcic:
        if np.all(kx==0) or np.all(ky==0) or np.all(kz==0):                                                     #Hvis alle værdierne i en af kx,ky,kz er 0 så er det most likely fordi der ikke er givet en kx,ky,kz som der skal
            raise Exception('Skal have givet kx,ky,kz hvis fromcic = True')

        else:
            # W_CIC = (np.sinc(kx / 2) * np.sinc(ky / 2) * np.sinc(kz / 2)) ** 2                                #Nyquist = pi N / L
            k_Nyq = np.pi * N / L     
            W_CIC = (np.sinc(kx / (2 * k_Nyq)) * np.sinc(ky / (2 * k_Nyq)) * np.sinc(kz / (2 * k_Nyq))) ** 2    #W_CIC = gange sum over x,y,z af sinc(pi*k_i / (2k_Nyq))
            # print(W_CIC)
            W_CIC[W_CIC == 0] = 1  # Avoid division by zero
            p_k_vec /= W_CIC**2  # Correcting for CIC interpolation effects, skal vel være W_CIC^2 da den som sådan hører til delta feltet

        
    k_mag_flat = k_mag.flatten().real
    p_k_flat = p_k_vec.flatten().real

    p_k_mean, bin_edges, binnumber = ss.binned_statistic(k_mag_flat, p_k_flat, statistic='mean', bins=nbins)    #Binner og finder automatisk mean for hver bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])                                                        #len(bin_edges) er len(p_k_mean) + 1 så tager -1 element, den ene tager uden det sidste og den anden uden det første så man altid får to sammenhængende og finder gennemsnit af dem
    
    if plot:
        # plt.figure()
        plt.scatter(bin_centers, p_k_mean)
        plt.xscale('log') ; plt.yscale('log')
        plt.xlabel('k [1/Mpc]') ; plt.ylabel(r'P(k) [$Mpc^3$/h]')  ; plt.title('Binned Power Spectrum')
    
    if returnP:
        return bin_centers, p_k_mean, k_mag_flat, p_k_flat
    else:
        return bin_centers, p_k_mean

z = 0
L = 1000
N = 32
N_samples = 100
N_samples_test = 10

sigmaMult = 10
A_s_min = (2.105 - 0.030 * sigmaMult) * 1e-9                #https://arxiv.org/pdf/1807.06209#page=19&zoom=100,56,722 side 16 omega_cdm bare den de kalder omega_c i assume
A_s_max = (2.105 + 0.030 * sigmaMult) * 1e-9                #Prøv måske at implementer joblib parallel (tror den hedder delayed den der skal bruges) processing
n_s_min = (0.9665 - 0.0038 * sigmaMult)                     
n_s_max = (0.9665 + 0.0038 * sigmaMult)
omega_cdm_min = (0.11933 - 0.00091 * sigmaMult)              
omega_cdm_max = (0.11933 + 0.00091 * sigmaMult)

#For CONCEPT anbefaler chatGPT at holde A_s indenfor ~1-5, n_s ~0.9-1.1 og omega_cdm ~(0.1-0.5)h^2
#RESULTATER lader umiddelbart til at være bedre hvis den får lov at træne på +-10 sigma (kan måske endda gå større?) også selvom testdata kun er +-5 sigma
n_s_min = 0.7 ; n_s_max = 1.1

# A_s_min = 2.1e-9 ; A_s_max = 2.1e-9
n_s_min = 0.9649 ; n_s_max = 0.9649 
omega_cdm_min = 0.12 ; omega_cdm_max = 0.12

"""Bare for at undgå at køre dem inde i funktionen mange gange"""
k_vals = 2 * np.pi * np.fft.fftfreq(N, d= L/N)                              #d = afstand mellem punkter, N = antal punkter
kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing = 'ij')           #Lav det til et grid
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)                                      #Størrelsen i hvert punkt af vores grid
k_mag[0,0,0] = 1e-10




savepath = r'/home/candifloos/Bachelor/NN models/Created data//'
makeFolders(savepath = savepath)                                    #Laver folders til at gemme data i       

with open(savepath + 'BoxParams.txt', mode = 'w') as f:             #Gemmer values brugt for boks og parametre
    f.write(f'{z} \t {L} \t {N}')
with open(savepath + 'MaxMinParams.txt', mode = 'w') as f:
    f.write(f'{A_s_min} \t {A_s_max} \t {n_s_min} \t {n_s_max} \t {omega_cdm_min} \t {omega_cdm_max} \t {N_samples}')


createTestData(N_samples_test, z, L, N, A_s_min = A_s_min, A_s_max = A_s_max, n_s_min = n_s_min, n_s_max = n_s_max, omega_cdm_min = omega_cdm_min, omega_cdm_max = omega_cdm_max, savepath = savepath, TestData = True)

createData(N_samples, z, L, N, 
                                A_s_min = A_s_min, A_s_max = A_s_max,
                                n_s_min = n_s_min, n_s_max = n_s_max,
                                omega_cdm_min = omega_cdm_min, omega_cdm_max = omega_cdm_max, 
                                savepath = savepath, SameTrain = False, paramRepeat = 1, overWrite = True)            #Hurtig test med paramRepeat = 5 lader ikke til at være meget forskellig fra 1, om noget slightly værre (total mængde samples 5k i begge tilfælde)




"""
spørgsmål:

"""
