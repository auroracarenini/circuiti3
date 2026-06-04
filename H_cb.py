import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 
from iminuit import Minuit
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
#errore slle tensioni 
sigma_v = 20*np.sqrt(2)*1e-3

#funzione per calcolare la differenza di fase 
def d_fase (f,dt): 
    return 2*pi*f *dt*1e6 

# dt1 = tA-tB  (MISURE IN mus)
dt1 = df["tch1"]-df["tch2"] 

#dt2 = t_math - tA  (MISURE IN mus)
dt2 = df["tmath"]-df["tch1"]
#frequeze 
freq = df["f"]



def H_abb_C (tau,f): 
    return 1/np.sqrt(1+(tau*2*pi*f)**2)

def min_quad(f,val,var):
    return (f-val)**2/var

# calcolo per il transitorio ai capi del condensatore H_abb
H_moduli = df["math"]/df["ch1"]
#errore sul transitorio 
sigma_H = sigma_v * np.sqrt((1/(df["ch1"])**2)+(H_moduli/df["ch1"])**2)


def cost_func(tau): 
    f = H_abb_C(tau,freq)
    Q = min_quad(f,H_moduli,sigma_H)
    return np.sum(Q) 

M = Minuit(cost_func,tau = 3e-6)
M.migrad()
M.hesse()

tau_s = M.values["tau"]  
sigma_tau_s = M.errors["tau"]



print(f"Risultato Fit: \u03c4 = {tau_s:.3e} \u00b1 {sigma_tau_s:.3e} s")


# --- PARTE GRAFICA ---
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Generiamo i punti per la curva continua del fit
fs = np.linspace(10, 60000, 700)
H_modulo_fit = H_abb_C(tau_s, fs)
H_moduli_dati= H_moduli 

# --- Grafico 1: Scala Lineare ---
ax[0].plot(
    fs, H_modulo_fit, linestyle="--", color="blue", label="Curva di Fit (\u03c4)"
)
ax[0].scatter(
    freq, H_moduli_dati, color="black", marker="o", label="Dati raccolti"
)
ax[0].set_xlabel("Frequenza (Hz)")
ax[0].set_ylabel("|H| (Rapporto ampiezze)")
ax[0].set_title("Risposta in Frequenza (Lineare)")
ax[0].grid(True)
ax[0].legend()

# --- Grafico 2: Diagramma di Bode (Scala Logaritmica) ---
H_dB_fit = 20 * np.log10(H_modulo_fit)
H_dB_dati = 20 * np.log10(H_moduli_dati)  # Nuova variabile per non sovrascrivere

ax[1].semilogx(fs, H_dB_fit, linestyle="--", color="red", label="Bode Fit (dB)")
ax[1].semilogx(
    freq,
    H_dB_dati,
    linestyle="",
    marker=".",
    color="black",
    label="Dati raccolti (dB)",
)
ax[1].set_xlabel("Frequenza (Hz, scala log)")
ax[1].set_ylabel("Ampiezza (dB)")
ax[1].set_title("Diagramma di Bode")
ax[1].grid(True, which="both", linestyle=":")  # Griglia logaritmica fitta
ax[1].legend()

plt.tight_layout()
plt.show()
plt.savefig(graph_pathHabb)

