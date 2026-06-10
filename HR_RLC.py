import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2  

# --- CONFIGURAZIONE E DATI ---
R_valore = 767 
R_L = 27.3
sigma_V = 20 * np.sqrt(2) * 1e-3  # Incertezza sull'ampiezza (Volt)

file_path = 'RLC_R.csv'

col_f = 'f'         
col_Va = 'ch1'      
col_Vb = 'ch2'      
col_t_Va = 'tch1'   
col_t_Vb = 'tch2'   

# --- FUNZIONE DI TRASFERIMENTO (Rinominata correttamente in Passa-Banda) ---
def passa_banda_modulo_R(f, L, C):
    omega = 2 * np.pi * f
    return R_valore / np.sqrt((R_valore + R_L)**2 + (omega * L - 1 / (omega * C))**2)

def passa_banda_fase_R(f, L, C):
    omega = 2 * np.pi * f
    return np.arctan((omega * L - 1 / (omega * C)) / (R_valore + R_L))

# --- CARICAMENTO DATI ---
df = pd.read_csv(file_path).sort_values(by=col_f)

f_dati = df[col_f].values
Va = df[col_Va].values
Vb = df[col_Vb].values
tch1 = df[col_t_Va].values * 1e-6
tch2 = df[col_t_Vb].values * 1e-6

H = Vb / Va

# Errore del modulo (Propagazione standard)
err_H_HP = H * np.sqrt((sigma_V / Vb)**2 + (sigma_V / Va)**2)

# --- CALCOLO FASE SPERIMENTALE ---
delta_t = tch2 - tch1
fase_dati = 2 * np.pi * f_dati * delta_t
fase_dati = (fase_dati + np.pi) % (2 * np.pi) - np.pi

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
p0_L = 1e-2      # 10 mH
c0_C = 70 * 1e-9 # 70 nF

popt_HP, pcov_HP = curve_fit(passa_banda_modulo_R, f_dati, H, p0=[p0_L, c0_C], sigma=err_H_HP, absolute_sigma=True)

L_fit_HP, C_fit_HP = popt_HP[0], popt_HP[1]
L_err_HP, C_err_HP = np.sqrt(pcov_HP[0][0]), np.sqrt(pcov_HP[1][1])

print("--- RISULTATI DEI FIT (MODULO) ---")
print(f"L ottimizzato:  {L_fit_HP*1e3:.2f} ± {L_err_HP*1e3:.2f} mH")
print(f"C ottimizzato:  {C_fit_HP*1e9:.2f} ± {C_err_HP*1e9:.2f} nF")

# --- CALCOLO STATISTICO: MODULO ---
valori_attesi_mod = passa_banda_modulo_R(f_dati, L_fit_HP, C_fit_HP)
chi2_mod = np.sum(((H - valori_attesi_mod) / err_H_HP) ** 2)
dof_mod = len(f_dati) - len(popt_HP)
p_value_mod = chi2.sf(chi2_mod, dof_mod)

print("\n--- STATISTICHE DEL FIT (MODULO) ---")
print(f"Chi2 Modulo: {chi2_mod:.2f} / {dof_mod} DoF")
print(f"p-value Modulo: {p_value_mod:.4e}")

# --- CALCOLO STATISTICO: FASE ---
valori_attesi_fase = passa_banda_fase_R(f_dati, L_fit_HP, C_fit_HP)
chi2_fase = np.sum(((fase_dati - valori_attesi_fase) / err_fase) ** 2)
dof_fase = len(f_dati) - len(popt_HP)
p_value_fase = chi2.sf(chi2_fase, dof_fase)

print("\n--- STATISTICHE DELLA FASE (CONFRONTO MODELLO) ---")
print(f"Chi2 Fase: {chi2_fase:.2f} / {dof_fase} DoF")
print(f"p-value Fase: {p_value_fase:.4e}")

# --- GRAFICI ---
f_fit = np.logspace(np.log10(min(f_dati)), np.log10(max(f_dati)), 6000)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# [0, 0] Modulo Lineare
axs[0, 0].errorbar(f_dati, H, yerr=err_H_HP, fmt='o', color='blue', label='Dati Sperimentali', capsize=3)
axs[0, 0].plot(f_fit, passa_banda_modulo_R(f_fit, L_fit_HP, C_fit_HP), 'b-', label='Modello Teorico')
axs[0, 0].set_title(f"Modulo (Lineare)\n$\\chi^2$={chi2_mod:.1f}, p-val={p_value_mod:.2e}")
axs[0, 0].set_xlabel("Frequenza [Hz]")
axs[0, 0].set_ylabel("|H(f)|")
axs[0, 0].grid(True, which="both", ls="--")
axs[0, 0].legend()

# [0, 1] Modulo Log-Log
axs[0, 1].loglog(f_fit, passa_banda_modulo_R(f_fit, L_fit_HP, C_fit_HP), 'b-', label='Modello Teorico')
axs[0, 1].errorbar(f_dati, H, yerr=err_H_HP, fmt='o', color='blue', capsize=3, label='Dati Sperimentali')
axs[0, 1].set_title(f"Modulo (Log-Log)\n$\\chi^2$={chi2_mod:.1f}, p-val={p_value_mod:.2e}")
axs[0, 1].set_xlabel("Frequenza [Hz]")
axs[0, 1].set_ylabel("|H(f)|")
axs[0, 1].grid(True, which="both", ls="--")
axs[0, 1].legend()

# [1, 0] Fase Lineare
axs[1, 0].errorbar(f_dati, fase_dati, yerr=err_fase, fmt='o', color='red', label='Dati Sperimentali', capsize=3)
axs[1, 0].plot(f_fit, passa_banda_fase_R(f_fit, L_fit_HP, C_fit_HP), 'r-', label='Modello Teorico')
axs[1, 0].set_title(f"Fase (Lineare)\n$\\chi^2$={chi2_fase:.1f}, p-val={p_value_fase:.2e}")
axs[1, 0].set_xlabel("Frequenza [Hz]")
axs[1, 0].set_ylabel("Fase [rad]")
axs[1, 0].grid(True, which="both", ls="--")
axs[1, 0].legend()

# [1, 1] Fase Semilog-X
axs[1, 1].semilogx(f_fit, passa_banda_fase_R(f_fit, L_fit_HP, C_fit_HP), 'r-', label='Modello Teorico')
axs[1, 1].errorbar(f_dati, fase_dati, yerr=err_fase, fmt='o', color='red', capsize=3, label='Dati Sperimentali')
axs[1, 1].set_title(f"Fase (Semilog-X)\n$\\chi^2$={chi2_fase:.1f}, p-val={p_value_fase:.2e}")
axs[1, 1].set_xlabel("Frequenza [Hz]")
axs[1, 1].set_ylabel("Fase [rad]")
axs[1, 1].grid(True, which="both", ls="--")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('Analisi_Completa_RLC.png', dpi=300)
plt.show()