import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist  # Per il calcolo del p-value

# --- CONFIGURAZIONE E DATI ---
R_valore = 767 
R_L = 27.3 
sigma_V = 20 * np.sqrt(2) * 1e-3  # 20*sqrt(2) mV convertiti in Volt

file_path = 'RLC_L.csv'

col_f = 'f'         # Frequenza
col_Va = 'ch1'      # Tensione Va
col_Vb = 'ch2'      # Tensione Vb
col_t_Va = 'tch1'   # Tempo di Va
col_t_Vb = 'tch2'   # Tempo di Vb

# --- FUNZIONI DI TRASFERIMENTO ---
def HL_rlc(f, L, C): 
    omega = 2 * np.pi * f 
    ind = omega * L 
    return ind / (np.sqrt((R_valore + R_L)**2 + (ind - 1 / (omega * C))**2))

def passa_alto_fase_R(f, L, C):
    omega = 2 * np.pi * f
    return -np.pi/2 + np.arctan((omega * L - 1 / (omega * C)) / (R_valore + R_L))

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

# --- CALCOLO ERRORE DINAMICO + RISOLUZIONE STRUMENTO ---
sigma_t_a_noise = sigma_V / (2 * np.pi * f_dati * Va)
sigma_t_b_noise = sigma_V / (2 * np.pi * f_dati * Vb)

sigma_t_str = 0.01 / f_dati  

sigma_t_a = np.sqrt(sigma_t_a_noise**2 + sigma_t_str**2)
sigma_t_b = np.sqrt(sigma_t_b_noise**2 + sigma_t_str**2)

sigma_delta_t = np.sqrt(sigma_t_a**2 + sigma_t_b**2)
err_fase = 2 * np.pi * f_dati * sigma_delta_t

# Stime iniziali per il fit del modulo
p0_L = 1e-2      # 10 mH
c0_C = 68 * 1e-9 # 68 nF

# --- ESECUZIONE DEL FIT (SOLO MODULO) ---
popt_HP, pcov_HP = curve_fit(HL_rlc, f_dati, H, p0=[p0_L, c0_C], sigma=err_H_HP, absolute_sigma=True)

L_fit_HP, C_fit_HP = popt_HP[0], popt_HP[1]
L_err_HP, C_err_HP = np.sqrt(pcov_HP[0][0]), np.sqrt(pcov_HP[1][1])

# --- STATISTICHE DELLA BONTÀ DEL FIT ---

# 1. Statistiche per il MODULO
residui_modulo = H - HL_rlc(f_dati, L_fit_HP, C_fit_HP)
chi2_modulo = np.sum((residui_modulo / err_H_HP) ** 2)
ndof_modulo = len(f_dati) - len(popt_HP)
p_value_modulo = chi2_dist.sf(chi2_modulo, ndof_modulo)

# 2. Statistiche per la FASE (Utilizza L e C ricavati dal modulo)
residui_fase = fase_dati - passa_alto_fase_R(f_dati, L_fit_HP, C_fit_HP)
chi2_fase = np.sum((residui_fase / err_fase) ** 2)
# ndof = N - 2 poiché fissiamo i 2 parametri stimati dal modulo
ndof_fase = len(f_dati) - len(popt_HP) 
p_value_fase = chi2_dist.sf(chi2_fase, ndof_fase)

# --- STAMPA RISULTATI SUL TERMINALE ---
print("==================================================")
print("             RISULTATI COMPLETI DEL FIT           ")
print("==================================================")
print(f"L ottimizzato da modulo: {L_fit_HP*1e3:.2f} ± {L_err_HP*1e3:.2f} mH")
print(f"C ottimizzato da modulo: {C_fit_HP*1e9:.2f} ± {C_err_HP*1e9:.2f} nF")
print("--------------------------------------------------")
print("STATISTICHE MODULO:")
print(f"  Chi2 Modulo:             {chi2_modulo:.2f}")
print(f"  Gradi di libertà (ndof): {ndof_modulo}")
print(f"  Chi2 ridotto (Chi2/ndof):{chi2_modulo/ndof_modulo:.2f}")
print(f"  p-value Modulo:          {p_value_modulo:.4e}")
print("--------------------------------------------------")
print("STATISTICHE FASE (Verifica indipendente):")
print(f"  Chi2 Fase:               {chi2_fase:.2f}")
print(f"  Gradi di libertà (ndof): {ndof_fase}")
print(f"  Chi2 ridotto (Chi2/ndof):{chi2_fase/ndof_fase:.2f}")
print(f"  p-value Fase:            {p_value_fase:.4e}")
print("==================================================")

# --- CREAZIONE GRAFICI ---
f_fit = np.logspace(np.log10(min(f_dati)), np.log10(max(f_dati)), 6000)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Limiti intelligenti per l'asse X (rimuove lo spazio vuoto iniziale nei log)
x_lim_min = min(f_dati) * 0.8
x_lim_max = max(f_dati) * 1.2

# --- [0, 0] Modulo Scala Lineare ---
axs[0, 0].errorbar(f_dati, H, yerr=err_H_HP, fmt='o', color='blue', label='Dati Sperimentali', capsize=3)
axs[0, 0].plot(f_fit, HL_rlc(f_fit, L_fit_HP, C_fit_HP), 'b-', label='Modello Teorico')
axs[0, 0].set_title("Modulo del Rapporto di Trasferimento (Lineare)")
axs[0, 0].set_xlabel("Frequenza [Hz]")
axs[0, 0].set_ylabel("|H(f)|")
axs[0, 0].grid(True, which="both", ls="--")
axs[0, 0].legend()

# --- [0, 1] Modulo Scala Log-Log (Aggiustata) ---
axs[0, 1].loglog(f_fit, HL_rlc(f_fit, L_fit_HP, C_fit_HP), 'b-', label='Modello Teorico')
axs[0, 1].errorbar(f_dati, H, yerr=err_H_HP, fmt='o', color='blue', capsize=3, label='Dati Sperimentali')
axs[0, 1].set_title("Modulo del Rapporto di Trasferimento (Log-Log)")
axs[0, 1].set_xlabel("Frequenza [Hz]")
axs[0, 1].set_ylabel("|H(f)|")
axs[0, 1].set_xlim(x_lim_min, x_lim_max)  # FIX SCALA LOG
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

# --- [1, 1] Fase Scala Semilogaritmica (Aggiustata) ---
axs[1, 1].semilogx(f_fit, passa_alto_fase_R(f_fit, L_fit_HP, C_fit_HP), 'r-', label='Modello Teorico')
axs[1, 1].errorbar(f_dati, fase_dati, yerr=err_fase, fmt='o', color='red', capsize=3, label='Dati Sperimentali')
axs[1, 1].set_title("Fase della Funzione di Trasferimento (Semilog-X)")
axs[1, 1].set_xlabel("Frequenza [Hz]")
axs[1, 1].set_ylabel("Fase [rad]")
axs[1, 1].set_xlim(x_lim_min, x_lim_max)  # FIX SCALA SEMILOG
axs[1, 1].grid(True, which="both", ls="--")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('Analisi_Completa_RLC_L.png', dpi=300)
print("\nGrafico completo salvato come 'Analisi_Completa_RLC_L.png'")
plt.show()