import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2  

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



def passa_basso_fase_R(f, L, C):
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
'''
# Propagazione errore sulla fase: sigma_delta_t = sqrt(2)*sigma_t
err_fase = 2 * np.pi * f_dati * np.sqrt(2) * sigma_t
'''
# Stime iniziali per il fit
p0_L = 1e-2      # 10 mH
c0_C = 68 * 1e-9 # 68 nF

# --- CALCOLO ERRORE DINAMICO + RISOLUZIONE STRUMENTO ---

# 1. Jitter temporale dovuto al rumore di ampiezza (dai tuoi appunti)
sigma_t_a_noise = sigma_V / (2 * np.pi * f_dati * Va)
sigma_t_b_noise = sigma_V / (2 * np.pi * f_dati * Vb)

# 2. Risoluzione temporale intrinseca dell'oscilloscopio (Incertezza di lettura)
# Valore stimato di circa 400 ns (adeguato se i punti ad alta frequenza deviano di ~0.1 rad)
sigma_t_str = 0.7e-6  

# 3. Composizione in quadratura degli errori per singolo canali
sigma_t_a = np.sqrt(sigma_t_a_noise**2 + sigma_t_str**2)
sigma_t_b = np.sqrt(sigma_t_b_noise**2 + sigma_t_str**2)

# 4. Propagazione finale sulla differenza di tempo e sulla fase
sigma_delta_t = np.sqrt(sigma_t_a**2 + sigma_t_b**2)
err_fase = 2 * np.pi * f_dati * sigma_delta_t

# --- ESECUZIONE DEL FIT ---
popt_HP, pcov_HP = curve_fit(HC_rlc, f_dati, H, p0=[p0_L, c0_C], sigma=err_H_HP, absolute_sigma=True)

# Estrazione dei risultati del fit per entrambi i parametri
L_fit_HP, C_fit_HP = popt_HP[0], popt_HP[1]
L_err_HP, C_err_HP = np.sqrt(pcov_HP[0][0]), np.sqrt(pcov_HP[1][1])

print("--- RISULTATI DEI FIT (MODULO) ---")
print(f"L ottimizzato:  {L_fit_HP*1e3:.2f} ± {L_err_HP*1e3:.2f} mH")
print(f"C ottimizzato:  {C_fit_HP*1e9:.2f} ± {C_err_HP*1e9:.2f} nF")

# --- CALCOLO STATISTICO: MODULO ---
valori_attesi_mod = HC_rlc(f_dati, L_fit_HP, C_fit_HP)
chi2_mod = np.sum(((H - valori_attesi_mod) / err_H_HP) ** 2)
dof_mod = len(f_dati) - len(popt_HP)
p_value_mod = chi2.sf(chi2_mod, dof_mod)

print("\n--- STATISTICHE DEL FIT (MODULO) ---")
print(f"Chi2 Modulo: {chi2_mod:.2f} / {dof_mod} DoF")
print(f"p-value Modulo: {p_value_mod:.4e}")

# --- CALCOLO STATISTICO: FASE ---
valori_attesi_fase = passa_basso_fase_R(f_dati, L_fit_HP, C_fit_HP)
chi2_fase = np.sum(((fase_dati - valori_attesi_fase) / err_fase) ** 2)
dof_fase = len(f_dati) - len(popt_HP)
p_value_fase = chi2.sf(chi2_fase, dof_fase)

print("\n--- STATISTICHE DELLA FASE (CONFRONTO MODELLO) ---")
print(f"Chi2 Fase: {chi2_fase:.2f} / {dof_fase} DoF")
print(f"p-value Fase: {p_value_fase:.4e}")
# --- GRAFICI ---
f_fit = np.logspace(np.log10(min(f_dati)), np.log10(max(f_dati)), 6000)

fig, axs = plt.subplots(1, 2, figsize= (14,7))

# --- [0, 0] Modulo Scala Lineare ---
axs[0].errorbar(f_dati, H, yerr=err_H_HP, fmt='o', color='blue', label='Dati Sperimentali', capsize=3)
axs[0].plot(f_fit, HC_rlc(f_fit, L_fit_HP, C_fit_HP), 'b-', label='Modello Teorico')
axs[0].set_title("Modulo del Rapporto di Trasferimento (Lineare)")
axs[0].set_xlabel("Frequenza [Hz]")
axs[0].set_ylabel("|H(f)|")
axs[0].grid(True, which="both", ls="--")
axs[0].legend()

# --- [0, 1] Modulo Scala Log-Log ---
axs[1].loglog(f_fit, HC_rlc(f_fit, L_fit_HP, C_fit_HP), 'b-', label='Modello Teorico')
axs[1].errorbar(f_dati, H, yerr=err_H_HP, fmt='o', color='blue', capsize=3, label='Dati Sperimentali')
axs[1].set_title("Modulo del Rapporto di Trasferimento (Log-Log)")
axs[1].set_xlabel("Frequenza [Hz]")
axs[1].set_ylabel("|H(f)|")
axs[1].grid(True, which="both", ls="--")
axs[1].legend()
'''
# --- [1, 0] Fase Scala Lineare ---
axs[1, 0].errorbar(f_dati, fase_dati, yerr=err_fase, fmt='o', color='red', label='Dati Sperimentali', capsize=3)
axs[1, 0].plot(f_fit, passa_basso_fase_R(f_fit, L_fit_HP, C_fit_HP), 'r-', label='Modello Teorico')
axs[1, 0].set_title("Fase della Funzione di Trasferimento (Lineare)")
axs[1, 0].set_xlabel("Frequenza [Hz]")
axs[1, 0].set_ylabel("Fase [rad]")
axs[1, 0].grid(True, which="both", ls="--")
axs[1, 0].legend()

# --- [1, 1] Fase Scala Semilogaritmica (Asse X Log) ---
axs[1, 1].semilogx(f_fit, passa_basso_fase_R(f_fit, L_fit_HP, C_fit_HP), 'r-', label='Modello Teorico')
axs[1, 1].errorbar(f_dati, fase_dati, yerr=err_fase, fmt='o', color='red', capsize=3, label='Dati Sperimentali')
axs[1, 1].set_title("Fase della Funzione di Trasferimento (Semilog-X)")
axs[1, 1].set_xlabel("Frequenza [Hz]")
axs[1, 1].set_ylabel("Fase [rad]")
axs[1, 1].grid(True, which="both", ls="--")
axs[1, 1].legend()
'''
plt.tight_layout()
plt.savefig('Analisi_Completa_RLC_C.png', dpi=300)
print("\nGrafico completo salvato come 'Analisi_Completa_RLC_C.png'")
plt.show()