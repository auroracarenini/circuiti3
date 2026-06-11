import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2  # Importato per il calcolo del p-value

R_valore = 4648  
R_L = 27
sigma_V = 20 * np.sqrt(2) * 1e-3  # 20*sqrt(2) mV convertiti in Volt
sigma_t_str = 700e-9              # Risoluzione intrinseca dello strumento (700 ns)

file_path = 'RL.csv'

col_f = 'f'                # Frequenza
col_Va = 'ch1'             # Tensione Va
col_Vb = 'ch2'             # Tensione Vb
col_V_AB = 'math'          # Tensione Va - Vb
col_t_Va = 'tch1'      # Tempo di Va
col_t_Vb = 'tch2'      # Tempo di Vb
col_t_V_AB = 'tmath'  # Tempo di Va - Vb


def passa_alto_modolo(f, L):
    return np.sqrt(R_L**2 + (2 * np.pi * f * L)**2) / np.sqrt((R_valore + R_L)**2 + (2 * np.pi * f * L)**2)

def passa_basso_modolo(f, L):
    return R_valore / np.sqrt((R_valore + R_L)**2 + (2 * np.pi * f * L)**2)

def passa_alto_fase_rl(f, L):
    return -np.arctan((2 * np.pi * f * L) / (R_valore + R_L))

def passa_basso_fase_rl(f, L):
    return np.arctan((2 * np.pi * f * L) / R_L) - np.arctan((2 * np.pi * f * L) / (R_valore + R_L))

df = pd.read_csv(file_path).sort_values(by=col_f)

f_dati = df[col_f].values
Va = df[col_Va].values
Vb = df[col_Vb].values
V_AB = df[col_V_AB].values * 1.96

# Modulo sperimentale
# Nota: nel circuito RL, V_AB è la tensione sull'induttore (Passa-Alto) e Vb è su R (Passa-Basso)
H_HP_exp = V_AB / Va
H_LP_exp = Vb / Va

# --- NUOVA PROPAGAZIONE ERRORE MODULO (STATISTICO + SISTEMATICO) ---
# k_sys rappresenta l'incertezza di calibrazione del guadagno verticale (es. 1.8%)
k_sys = 0.018  

# Incertezza totale sulle singole tensioni (statica + sistematica proporzionale)
sigma_Va_tot = np.sqrt(sigma_V**2 + (k_sys * Va)**2)
sigma_Vb_tot = np.sqrt(sigma_V**2 + (k_sys * Vb)**2)
sigma_V_AB_tot = np.sqrt(sigma_V**2 + (k_sys * V_AB)**2)

# Propagazione corretta sul modulo H = V_out / V_in
# Passa-Alto (V_AB / Va)
err_H_HP = H_HP_exp * np.sqrt((sigma_V_AB_tot / V_AB)**2 + (sigma_Va_tot / Va)**2)

# Passa-Basso (Vb / Va)
err_H_LP = H_LP_exp * np.sqrt((sigma_Vb_tot / Vb)**2 + (sigma_Va_tot / Va)**2)

if col_t_Va in df.columns and col_t_Vb in df.columns:
    dt_LP = - (df[col_t_V_AB].values - df[col_t_Va].values) * 1e-6 
    dt_HP = (df[col_t_Vb].values - df[col_t_Va].values) * 1e-6
    dt_HP = - dt_HP
    
    fase_LP_exp = 2 * np.pi * f_dati * dt_LP
    fase_HP_exp = 2 * np.pi * f_dati * dt_HP
    sigma_t_str = 0.01 / f_dati
    # --- MODELLO DI ERRORE DINAMICO SULLA FASE ---
    sigma_t_A = np.sqrt((sigma_V / (2 * np.pi * f_dati * Va))**2 + sigma_t_str**2)
    sigma_t_B = np.sqrt((sigma_V / (2 * np.pi * f_dati * Vb))**2 + sigma_t_str**2)
    sigma_t_AB = np.sqrt((sigma_V / (2 * np.pi * f_dati * V_AB))**2 + sigma_t_str**2)
    
    sigma_dt_LP = np.sqrt(sigma_t_A**2 + sigma_t_AB**2)
    sigma_dt_HP = np.sqrt(sigma_t_A**2 + sigma_t_B**2)
    
    err_fase_LP = 2 * np.pi * f_dati * sigma_dt_LP
    err_fase_HP = 2 * np.pi * f_dati * sigma_dt_HP
    DATO_FASE_PRESENTE = True
else:
    DATO_FASE_PRESENTE = False
    print("Colonne del tempo non trovate o non configurate.")


p0_L = 1e-2  # Stima iniziale 10 mH

# Fit Modulo Passa-Alto (V_AB/Va)
popt_HP, pcov_HP = curve_fit(passa_alto_modolo, f_dati, H_HP_exp, p0=[p0_L], sigma=err_H_HP, absolute_sigma=True)
L_fit_HP, L_err_HP = popt_HP[0], np.sqrt(pcov_HP[0][0])

res_HP_mod = H_HP_exp - passa_alto_modolo(f_dati, L_fit_HP)
chi2_HP_mod = np.sum((res_HP_mod / err_H_HP) ** 2)
dof_HP_mod = len(f_dati) - 1
chi2_red_HP_mod = chi2_HP_mod / dof_HP_mod
p_val_HP_mod = 1 - chi2.cdf(chi2_HP_mod, dof_HP_mod)

# Fit Modulo Passa-Basso (Vb/Va)
popt_LP, pcov_LP = curve_fit(passa_basso_modolo, f_dati, H_LP_exp, p0=[p0_L], sigma=err_H_LP, absolute_sigma=True)
L_fit_LP, L_err_LP = popt_LP[0], np.sqrt(pcov_LP[0][0])

res_LP_mod = H_LP_exp - passa_basso_modolo(f_dati, L_fit_LP)
chi2_LP_mod = np.sum((res_LP_mod / err_H_LP) ** 2)
dof_LP_mod = len(f_dati) - 1
chi2_red_LP_mod = chi2_LP_mod / dof_LP_mod
p_val_LP_mod = 1 - chi2.cdf(chi2_LP_mod, dof_LP_mod)

print("--- RISULTATI DEI FIT (MODULO) ---")
print(f"L (Passa-Basso): {L_fit_LP*1e3:.2f} ± {L_err_LP*1e3:.2f} mH | Chi2: {chi2_LP_mod:.2f}, Chi2_red: {chi2_red_LP_mod:.2f}, p-value: {p_val_LP_mod:.4f}")
print(f"L (Passa-Alto):  {L_fit_HP*1e3:.2f} ± {L_err_HP*1e3:.2f} mH | Chi2: {chi2_HP_mod:.2f}, Chi2_red: {chi2_red_HP_mod:.2f}, p-value: {p_val_HP_mod:.4f}\n")

# --- CALCOLO STATISTICHE FASE ---
if DATO_FASE_PRESENTE:
    res_LP_fase = fase_LP_exp - passa_basso_fase_rl(f_dati, L_fit_LP)
    chi2_LP_fase = np.sum((res_LP_fase / err_fase_LP) ** 2)
    dof_LP_fase = len(f_dati)
    chi2_red_LP_fase = chi2_LP_fase / dof_LP_fase
    p_val_LP_fase = 1 - chi2.cdf(chi2_LP_fase, dof_LP_fase)

    res_HP_fase = fase_HP_exp - passa_alto_fase_rl(f_dati, L_fit_HP)
    chi2_HP_fase = np.sum((res_HP_fase / err_fase_HP) ** 2)
    dof_HP_fase = len(f_dati)
    chi2_red_HP_fase = chi2_HP_fase / dof_HP_fase
    p_val_HP_fase = 1 - chi2.cdf(chi2_HP_fase, dof_HP_fase)
    
    print("--- RISULTATI COMPATIBILITÀ FASE ---")
    print(f"Fase Passa-Basso | Chi2: {chi2_LP_fase:.2f}, Chi2_red: {chi2_red_LP_fase:.2f}, p-value: {p_val_LP_fase:.4f}")
    print(f"Fase Passa-Alto  | Chi2: {chi2_HP_fase:.2f}, Chi2_red: {chi2_red_HP_fase:.2f}, p-value: {p_val_HP_fase:.4f}\n")


f_fit = np.logspace(np.log10(min(f_dati)), np.log10(max(f_dati)), 500)

# Grafici Modulo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
x_min_fissato = min(f_dati) * 0.7
x_max_fissato = max(f_dati) * 1.4

# --- Grafico 1: Modulo Scala Lineare ---
ax1.errorbar(f_dati, H_HP_exp, yerr=err_H_HP, fmt='o', color='blue', label='Dati Passa-Alto (V_AB/V_A)', capsize=3)
ax1.plot(f_fit, passa_alto_modolo(f_fit, L_fit_HP), 'b-', label='Fit Passa-Alto')
ax1.errorbar(f_dati, H_LP_exp, yerr=err_H_LP, fmt='s', color='red', label='Dati Passa-Basso (V_B/V_A)', capsize=3)
ax1.plot(f_fit, passa_basso_modolo(f_fit, L_fit_LP), 'r-', label='Fit Passa-Basso')
ax1.set_title("Modulo della Funzione di Trasferimento (Lineare)")
ax1.set_xlabel("Frequenza [Hz]")
ax1.set_ylabel("|H(f)|")
ax1.grid(True, which="both", ls="--")
ax1.legend(loc='lower center')

# --- Grafico 2: Modulo Scala Bi-Logaritmica ---
ax2.loglog(f_fit, passa_alto_modolo(f_fit, L_fit_HP), 'b-')
ax2.errorbar(f_dati, H_HP_exp, yerr=err_H_HP, fmt='o', color='blue', capsize=3, label='Dati Passa-Alto')
ax2.loglog(f_fit, passa_basso_modolo(f_fit, L_fit_LP), 'r-')
ax2.errorbar(f_dati, H_LP_exp, yerr=err_H_LP, fmt='s', color='red', capsize=3, label='Dati Passa-Basso')
ax2.set_title("Modulo della Funzione di Trasferimento (Log-Log)")
ax2.set_xlabel("Frequenza [Hz]")
ax2.set_ylabel("|H(f)|")
ax2.set_xlim(x_min_fissato, x_max_fissato)
y_min_log = min(np.min(H_LP_exp), np.min(H_HP_exp)) * 0.5
ax2.set_ylim(max(1e-3, y_min_log), 1.5)
ax2.grid(True, which="both", ls="--")
ax2.legend(loc='lower left')

# Box Statistiche Orizzontale Elegante
text_stats_mod = (
    f"  Cross-Over Stats (RL Circuit)\n"
    f"  --------------------------------------------------\n"
    f"  Passa-Basso:                     Passa-Alto:\n"
    f"  Chi2 = {chi2_LP_mod:.2f} (dof={dof_LP_mod})      Chi2 = {chi2_HP_mod:.2f} (dof={dof_HP_mod})\n"
    f"  Chi2_red = {chi2_red_LP_mod:.2f}             Chi2_red = {chi2_red_HP_mod:.2f}\n"
    f"  p-value = {p_val_LP_mod:.4f}              p-value = {p_val_HP_mod:.4f}"
)
props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.85)

ax2.text(0.95, 0.05, text_stats_mod, transform=ax2.transAxes, fontsize=9, fontfamily='monospace',
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('Fit_Modulo_RL.png', dpi=300)
print("Grafico del Modulo salvato come 'Fit_Modulo_RL.png'")
plt.show()

# --- GRAFICO FASE (CORRETTO) ---
if DATO_FASE_PRESENTE:
    fig2, ax3 = plt.subplots(figsize=(8, 6.5))
    
    ax3.errorbar(f_dati, fase_HP_exp, yerr=err_fase_HP, fmt='o', color='blue', label='Dati Passa-Alto (V_AB)', capsize=3)
    ax3.plot(f_fit, passa_alto_fase_rl(f_fit, L_fit_HP), 'b--', label='Fit Passa-Alto')
    
    ax3.errorbar(f_dati, fase_LP_exp, yerr=err_fase_LP, fmt='s', color='red', label='Dati Passa-Basso (V_b)', capsize=3)
    ax3.plot(f_fit, passa_basso_fase_rl(f_fit, L_fit_LP), 'r--', label='Fit Passa-Basso')
    
    ax3.set_xscale('log')
    ax3.set_xlim(x_min_fissato, x_max_fissato)
    ax3.set_title("Fase della Funzione di Trasferimento (Semi-Log)")
    ax3.set_xlabel("Frequenza [Hz]")
    ax3.set_ylabel("Fase [rad]")
    ax3.grid(True, which="both", ls="--")
    
    # Spostiamo la legenda in alto a destra per non intralciare
    ax3.legend(loc='upper right')
    
    # Box Statistiche Fase - POSIZIONATO IN BASSO A SINISTRA (0.05, 0.05)
    text_stats_fase = (
        f"  Phase Agreement Stats (Jitter Model)\n"
        f"  --------------------------------------------------\n"
        f"  Passa-Basso:                     Passa-Alto:\n"
        f"  Chi2 = {chi2_LP_fase:.2f} (dof={dof_LP_fase})      Chi2 = {chi2_HP_fase:.2f} (dof={dof_HP_fase})\n"
        f"  Chi2_red = {chi2_red_LP_fase:.2f}             Chi2_red = {chi2_red_HP_fase:.2f}\n"
        f"  p-value = {p_val_LP_fase:.4f}              p-value = {p_val_HP_fase:.4f}"
    )
    
    # Cambiati gli allineamenti in 'bottom' e 'left'
    ax3.text(0.05, 0.05, text_stats_fase, transform=ax3.transAxes, fontsize=9, fontfamily='monospace',
             verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    plt.savefig('Fit_Fase_RL.png', dpi=300)
    plt.show()