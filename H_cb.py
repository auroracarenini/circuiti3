import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 
current_dir = Path(__file__).parent
pi = np.pi

graph_pathHabb = current_dir/"RCH_abb.png"


path_file = current_dir/"RC.csv"
df = pd.read_csv(path_file)

#calcolo del tempo caratteristico 
R,sigma_R = 4648, 2
C,sigma_C = 68*1e-9,1*1e-9

tau = R*C 
sigma_tau = np.sqrt((R*sigma_C)**2+(C*sigma_R)**2)

#funzione per calcolare la differenza di fase 
def d_fase (f,dt): 
    return 2*pi*f *dt*1e6 

# dt1 = tA-tB  (MISURE IN mus)
dt1 = df["tch1"]-df["tch2"] 

#dt2 = t_math - tA  (MISURE IN mus)
dt2 = df["tmath"]-df["tch1"]

freq = df["f"]
def H_abb_C (tau,f): 
    return 1/(1+tau*2*pi*f)

def min_quad(f,val,var):
    return None



fig,ax =plt.subplots(1,2,figsize = (12,7)) 
fs = np.linspace(10,60000,700)
H_modulo = H_abb_C(tau,fs)

H_moduli = df["math"]/df["ch1"] 
ax[0].plot(fs,H_modulo,linestyle = "--")
ax[0].scatter(freq,H_moduli,label = "plot dati raccolti")

H_dB = 20 * np.log10(H_modulo)

# Usiamo semilogx così l'asse delle frequenze diventa logaritmico (10, 100, 1000...)
ax[1].semilogx(fs, H_dB, linestyle="--", color="red", label="Bode (dB)")
plt.savefig(graph_pathHabb)

