import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


R_valore = 4648  

# Inserimento degli errori forniti
sigma_V = 20 * np.sqrt(2) * 1e-3  # 20*sqrt(2) mV convertiti in Volt
sigma_t = 2e-6                    # 2 microsecondi convertiti in secondi

file_path = 'RC.csv'

col_f = 'f'                # Frequenza
col_Va = 'ch1'             # Tensione Va
col_Vb = 'ch2'             # Tensione Vb
col_V_AB = 'math'          # Tensione Va - Vb
col_t_Va = 'tch1'      # Tempo di Va (se presente nel CSV per la fase)
col_t_Vb = 'tch2'      # Tempo di Vb
col_t_V_AB = 'tmath'  # Tempo di Va - Vb


def passa_basso_modolo(f, C):
    return 1 / np.sqrt(1 + (2 * np.pi * f * R_valore * C)**2)

def passa_alto_modolo(f, C):
    omega_RC = 2 * np.pi * f * R_valore * C
    return omega_RC / np.sqrt(1 + omega_RC**2)

# Fasi (in radianti)
def passa_basso_fase(f, C):
    return -np.arctan(2 * np.pi * f * R_valore * C)

def passa_alto_fase(f, C):
    return np.arctan(1 / (2 * np.pi * f * R_valore * C))


df = pd.read_csv(file_path).sort_values(by=col_f)

f_dati = df[col_f].values
Va = df[col_Va].values
Vb = df[col_Vb].values
V_AB = df[col_V_AB].values * 1.96

# --- MODULO ---
H_LP_exp = V_AB / Va
H_HP_exp = Vb / Va

# Propagazione errore del modulo: sigma_H = H * sqrt((sigma_V/Vout)^2 + (sigma_V/Vin)^2)
err_H_LP = H_LP_exp * np.sqrt((sigma_V / V_AB)**2 + (sigma_V / Va)**2)
err_H_HP = H_HP_exp * np.sqrt((sigma_V / Vb)**2 + (sigma_V / Va)**2)

if col_t_Va in df.columns and col_t_Vb in df.columns:
    dt_LP = - (df[col_t_V_AB].values - df[col_t_Va].values) * 1e-6 
    dt_HP = - (df[col_t_Vb].values - df[col_t_Va].values) * 1e-6
    dt_HP =  dt_HP
    # Da qui in poi calcola la fase correttamente usando i secondi
    fase_LP_exp = 2 * np.pi * f_dati * dt_LP
    fase_HP_exp = 2 * np.pi * f_dati * dt_HP
    
    # Errore sulla differenza di tempo e sulla fase
    sigma_dt = np.sqrt(2) * sigma_t
    err_fase_LP = 2 * np.pi * f_dati * sigma_dt
    err_fase_HP = 2 * np.pi * f_dati * sigma_dt
    DATO_FASE_PRESENTE = True
else:
    DATO_FASE_PRESENTE = False
    print("Colonne del tempo non trovate o non configurate.")


p0_C = 1e-7  # Stima iniziale (100 nF)

# Fit Modulo Passa-Basso
popt_LP, pcov_LP = curve_fit(passa_basso_modolo, f_dati, H_LP_exp, p0=[p0_C], sigma=err_H_LP, absolute_sigma=True)
C_fit_LP, C_err_LP = popt_LP[0], np.sqrt(pcov_LP[0][0])

# Fit Modulo Passa-Alto
popt_HP, pcov_HP = curve_fit(passa_alto_modolo, f_dati, H_HP_exp, p0=[p0_C], sigma=err_H_HP, absolute_sigma=True)
C_fit_HP, C_err_HP = popt_HP[0], np.sqrt(pcov_HP[0][0])

print("--- RISULTATI DEI FIT (MODULO) ---")
print(f"C (Passa-Basso): {C_fit_LP*1e9:.2f} ± {C_err_LP*1e9:.2f} nF")
print(f"C (Passa-Alto):  {C_fit_HP*1e9:.2f} ± {C_err_HP*1e9:.2f} nF")


f_fit = np.logspace(np.log10(min(f_dati)), np.log10(max(f_dati)), 500)

# Creiamo una figura con 2 grafici per il Modulo (Lineare e Log)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
x_min_fissato = min(f_dati) * 0.8
x_max_fissato = max(f_dati) * 1.2
# --- Grafico 1: Modulo Scala Lineare ---
ax1.errorbar(f_dati, H_LP_exp, yerr=err_H_LP, fmt='o', color='blue', label='Dati Passa-Basso (V_AB/V_A)', capsize=3)
ax1.plot(f_fit, passa_basso_modolo(f_fit, C_fit_LP), 'b-', label='Fit Passa-Basso')
ax1.errorbar(f_dati, H_HP_exp, yerr=err_H_HP, fmt='s', color='red', label='Dati Passa-Alto (V_B/V_A)', capsize=3)
ax1.plot(f_fit, passa_alto_modolo(f_fit, C_fit_HP), 'r-', label='Fit Passa-Alto')
ax1.set_title("Modulo della Funzione di Trasferimento (Lineare)")
ax1.set_xlabel("Frequenza [Hz]")
ax1.set_ylabel("|H(f)|")
ax1.grid(True, which="both", ls="--")
ax1.legend()

# --- Grafico 2: Modulo Scala Bi-Logaritmica ---
ax2.loglog(f_fit, passa_basso_modolo(f_fit, C_fit_LP), 'b-')
ax2.errorbar(f_dati, H_LP_exp, yerr=err_H_LP, fmt='o', color='blue', capsize=3, label='Dati Passa-Basso')
ax2.loglog(f_fit, passa_alto_modolo(f_fit, C_fit_HP), 'r-')
ax2.errorbar(f_dati, H_HP_exp, yerr=err_H_HP, fmt='s', color='red', capsize=3, label='Dati Passa-Alto')
ax2.set_title("Modulo della Funzione di Trasferimento (Bicellulare/Log)")
ax2.set_xlabel("Frequenza [Hz]")
ax2.set_ylabel("|H(f)|")
ax2.set_xlim(x_min_fissato, x_max_fissato)
y_min_log = min(np.min(H_LP_exp), np.min(H_HP_exp)) * 0.5
ax2.set_ylim(max(1e-3, y_min_log), 1.5)
ax2.grid(True, which="both", ls="--")
ax2.legend()

plt.tight_layout()
plt.savefig('Fit_Modulo_RC.png', dpi=300)
print("Grafico del Modulo salvato come 'Fit_Modulo_RC.png'")
plt.show()

# --- GRAFICO FASE ---
if DATO_FASE_PRESENTE:
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    ax3.errorbar(f_dati, fase_LP_exp, yerr=err_fase_LP, fmt='o', color='blue', label='Fase Passa-Basso', capsize=3)
    ax3.plot(f_fit, passa_basso_fase(f_fit, C_fit_LP), 'b--')
    ax3.errorbar(f_dati, fase_HP_exp, yerr=err_fase_HP, fmt='s', color='red', label='Fase Passa-Alto', capsize=3)
    ax3.plot(f_fit, passa_alto_fase(f_fit, C_fit_HP), 'r--')
    ax3.set_xscale('log')
    ax3.set_title("Fase della Funzione di Trasferimento (Semi-Log)")
    ax3.set_xlabel("Frequenza [Hz]")
    ax3.set_ylabel("Fase [rad]")
    ax3.grid(True, which="both", ls="--")
    ax3.legend()
    plt.tight_layout()
    plt.savefig('Fit_Fase_RC.png', dpi=300)
    print("Grafico della Fase salvato come 'Fit_Fase_RC.png'")
    plt.show()