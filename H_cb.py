import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 
from iminuit import Minuit

current_dir = Path(__file__).parent
pi = np.pi

graph_pathHabb = current_dir / "RCH_abb.png"
path_file = current_dir / "RC.csv"
df = pd.read_csv(path_file)

# --- DATI COMPONENTI E CALCOLO TAU NOMINALE (IN SECONDI) ---
R, sigma_R = 4648, 2          # Ohm
C, sigma_C = 68 * 1e-9, 1 * 1e-9  # Farad

tau_nominale = R * C 
sigma_tau_nominale = np.sqrt((R * sigma_C)**2 + (C * sigma_R)**2)

# Errore sulle tensioni (es. risoluzione oscilloscopio)
sigma_v = 20 * np.sqrt(2) * 1e-3

# --- ANALISI DEI TEMPI E DELLE FREQUENZE ---
freq = df["f"] # in Hz

# Funzione per calcolare la differenza di fase (dt convertito da mus a s)
def d_fase(f, dt_mus): 
    return 2 * pi * f * (dt_mus * 1e-6) 

# dt1 = tA - tB e dt2 = t_math - tA (mantenuti in mus per la funzione d_fase)
dt1 = df["tch1"] - df["tch2"] 
dt2 = df["tmath"] - df["tch1"]

# --- FUNZIONE DI TRASFERIMENTO (MODULO SUL CONDENSATORE) ---
def H_abb_C(tau, f): 
    return 1 / np.sqrt(1 + (tau * 2 * pi * f)**2)

# Calcolo del modulo sperimentale H
H_moduli = np.abs(df["math"] / df["ch1"])

# Corretta propagazione dell'errore sul rapporto (H)
sigma_H = (sigma_v / df["ch1"]) * np.sqrt(1 + H_moduli**2)

# --- CONFIGURAZIONE DEL FIT (MINIMI QUADRATI PESATI) ---
def min_quad(f_teorico, val_sperimentale, sigma):
    # Il denominatore DEVE essere elevato al quadrato (varianza)
    return (f_teorico - val_sperimentale)**2 / (sigma**2) 

def cost_func(tau): 
    f_teorico = H_abb_C(tau, freq)
    Q = min_quad(f_teorico, H_moduli, sigma_H)
    return np.sum(Q) 

# Inizializzazione di Minuit con il valore di guess fisico (tau_nominale)
M = Minuit(cost_func, tau=tau_nominale)
M.migrad()
M.hesse()

tau_s = M.values["tau"]  
sigma_tau_s = M.errors["tau"]

print(f"--- RISULTATI ---")
print(f"Campione componenti: \u03c4_nominale = {tau_nominale:.3e} \u00b1 {sigma_tau_nominale:.3e} s")
print(f"Risultato del Fit:   \u03c4_fit      = {tau_s:.3e} \u00b1 {sigma_tau_s:.3e} s")

# --- PARTE GRAFICA ---
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Generiamo i punti per la curva continua del fit (fino a 150 kHz come da dispensa)
fs = np.linspace(10, 150000, 1000)
H_modulo_fit = H_abb_C(tau_s, fs)
H_moduli_dati= H_moduli

# --- Grafico 1: Scala Lineare ---
ax[0].plot(fs, H_modulo_fit, linestyle="--", color="blue", label="Curva di Fit (\u03c4)")
ax[0].errorbar(freq, H_moduli, yerr=sigma_H, fmt="o", color="black", ecolor="gray", capsize=3, label="Dati raccolti")
ax[0].set_xlabel("Frequenza (Hz)")
ax[0].set_ylabel("|H| (Rapporto ampiezze)")
ax[0].set_title("Risposta in Frequenza (Lineare)")
ax[0].grid(True)
ax[0].legend()

# --- Grafico 2: Diagramma di Bode (Scala Logaritmica) ---
H_dB_fit = 20 * np.log10(H_modulo_fit)
H_dB_dati = 20 * np.log10(H_moduli)  

ax[1].semilogx(fs, H_dB_fit, linestyle="--", color="red", label="Bode Fit (dB)")
ax[1].scatter(freq, H_dB_dati, color="black", marker=".", label="Dati raccolti (dB)")
ax[1].set_xlabel("Frequenza (Hz, scala log)")
ax[1].set_ylabel("Ampiezza (dB)")
ax[1].set_title("Diagramma di Bode")
ax[1].grid(True, which="both", linestyle=":")  
ax[1].legend()

plt.tight_layout()
plt.savefig(graph_pathHabb)  # Salva l'immagine prima che plt.show() svuoti la figura
plt.show()

