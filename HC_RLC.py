import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- CONFIGURAZIONE E DATI ---
R_valore = 767 
R_L = 27.3 
sigma_V = 20 * np.sqrt(2) * 1e-3  # 20*sqrt(2) mV convertiti in Volt
sigma_t = 2e-10                  # CORRETTO: 2 microsecondi = 2e-6 secondi

file_path = 'RLC_C.csv'

col_f = 'f'         # Frequenza
col_Va = 'ch1'      # Tensione Va
col_Vb = 'ch2'      # Tensione Vb
col_t_Va = 'tch1'   # Tempo di Va
col_t_Vb = 'tch2'   # Tempo di Vb

# --- FUNZIONE DI TRASFERIMENTO ---

def HC_rlc(f,L,C): 
    omega = 2*np.pi*f 
    cond = 1/(omega*C) #cariavile del condensatore (è sempre uguale) 
    return cond/np.sqrt((R_valore+R_L)**2+(omega*L-cond)**2)



def passa_alto_fase_R(f, L, C):
    omega = 2 * np.pi * f
    # Fase per circuito RLC con uscita ai capi di R
    return np.pi/2+np.arctan((omega * L - 1 / (omega * C)) / (R_valore + R_L))

# --- CARICAMENTO DATI ---

df = pd.read_csv(file_path).sort_values(by=col_f)

f_dati = df[col_f].values
Va = df[col_Va].values
Vb = df[col_Vb].values
tch1 = df[col_t_Va].values * 1e-6
tch2 = df[col_t_Vb].values * 1e-6

H = Vb / Va

# Errore del modulo
err_H_HP = H * np.sqrt((sigma_V / Vb)**2 + (sigma_V / Va)**2)

# --- CALCOLO FASE SPERIMENTALE ---
delta_t = tch2 - tch1
fase_dati = 2 * np.pi * f_dati * delta_t

# Normalizzazione della fase nell'intervallo [-pi, pi]
fase_dati = (fase_dati + np.pi) % (2 * np.pi) - np.pi

# Propagazione errore sulla fase: sigma_delta_t = sqrt(2)*sigma_t
err_fase = 2 * np.pi * f_dati * np.sqrt(2) * sigma_t

# Stime iniziali per il fit
p0_L = 1e-2      # 10 mH
c0_C = 68 * 1e-9 # 68 nF

# --- ESECUZIONE DEL FIT ---
popt_HP, pcov_HP = curve_fit(HC_rlc, f_dati, H, p0=[p0_L, c0_C], sigma=err_H_HP, absolute_sigma=True)

# Estrazione dei risultati del fit per entrambi i parametri
L_fit_HP, C_fit_HP = popt_HP[0], popt_HP[1]
L_err_HP, C_err_HP = np.sqrt(pcov_HP[0][0]), np.sqrt(pcov_HP[1][1])

print("--- RISULTATI DEI FIT (MODULO) ---")
print(f"L ottimizzato:  {L_fit_HP*1e3:.2f} ± {L_err_HP*1e3:.2f} mH")
print(f"C ottimizzato:  {C_fit_HP*1e9:.2f} ± {C_err_HP*1e9:.2f} nF")

# --- GRAFICI ---
f_fit = np.logspace(np.log10(min(f_dati)), np.log10(max(f_dati)), 6000)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# --- [0, 0] Modulo Scala Lineare ---
axs[0, 0].errorbar(f_dati, H, yerr=err_H_HP, fmt='o', color='blue', label='Dati Sperimentali', capsize=3)
axs[0, 0].plot(f_fit, HC_rlc(f_fit, L_fit_HP, C_fit_HP), 'b-', label='Modello Teorico')
axs[0, 0].set_title("Modulo del Rapporto di Trasferimento (Lineare)")
axs[0, 0].set_xlabel("Frequenza [Hz]")
axs[0, 0].set_ylabel("|H(f)|")
axs[0, 0].grid(True, which="both", ls="--")
axs[0, 0].legend()

# --- [0, 1] Modulo Scala Log-Log ---
axs[0, 1].loglog(f_fit, HC_rlc(f_fit, L_fit_HP, C_fit_HP), 'b-', label='Modello Teorico')
axs[0, 1].errorbar(f_dati, H, yerr=err_H_HP, fmt='o', color='blue', capsize=3, label='Dati Sperimentali')
axs[0, 1].set_title("Modulo del Rapporto di Trasferimento (Log-Log)")
axs[0, 1].set_xlabel("Frequenza [Hz]")
axs[0, 1].set_ylabel("|H(f)|")
axs[0, 1].grid(True, which="both", ls="--")
axs[0, 1].legend()

# --- [1, 0] Fase Scala Lineare ---
axs[1, 0].errorbar(f_dati, fase_dati, yerr=err_fase, fmt='o', color='red', label='Dati Sperimentali', capsize=3)
axs[1, 0].plot(f_fit, passa_alto_fase_R(f_fit, L_fit_HP, C_fit_HP), 'r-', label='Modello Teorico')
axs[1, 0].set_title("Fase della Funzione di Trasferimento (Lineare)")
axs[1, 0].set_xlabel("Frequenza [Hz]")
axs[1, 0].set_ylabel("Fase [rad]")
axs[1, 0].grid(True, which="both", ls="--")
axs[1, 0].legend()

# --- [1, 1] Fase Scala Semilogaritmica (Asse X Log) ---
axs[1, 1].semilogx(f_fit, passa_alto_fase_R(f_fit, L_fit_HP, C_fit_HP), 'r-', label='Modello Teorico')
axs[1, 1].errorbar(f_dati, fase_dati, yerr=err_fase, fmt='o', color='red', capsize=3, label='Dati Sperimentali')
axs[1, 1].set_title("Fase della Funzione di Trasferimento (Semilog-X)")
axs[1, 1].set_xlabel("Frequenza [Hz]")
axs[1, 1].set_ylabel("Fase [rad]")
axs[1, 1].grid(True, which="both", ls="--")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('Analisi_Completa_RLC_C.png', dpi=300)
print("\nGrafico completo salvato come 'Analisi_Completa_RLC_C.png'")
plt.show()