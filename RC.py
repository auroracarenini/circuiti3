import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

R_valore = 4648  

# Inserimento degli errori e parametri fisici forniti dal tuo modello
sigma_V = 20 * np.sqrt(2) * 1e-3  # Rumore di ampiezza (Volt)
sigma_t_str = 700e-9              # Risoluzione intrinseca dello strumento (700 ns)

file_path = 'RC.csv'

col_f = 'f'                
col_Va = 'ch1'             
col_Vb = 'ch2'             
col_V_AB = 'math'          
col_t_Va = 'tch1'      
col_t_Vb = 'tch2'      
col_t_V_AB = 'tmath'  

def passa_basso_modolo(f, C):
    return 1 / np.sqrt(1 + (2 * np.pi * f * R_valore * C)**2)

def passa_alto_modolo(f, C):
    omega_RC = 2 * np.pi * f * R_valore * C
    return omega_RC / np.sqrt(1 + omega_RC**2)

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

err_H_LP = H_LP_exp * np.sqrt((sigma_V / V_AB)**2 + (sigma_V / Va)**2)
err_H_HP = H_HP_exp * np.sqrt((sigma_V / Vb)**2 + (sigma_V / Va)**2)

# --- FASE CON IL TUO NUOVO MODELLO DI ERRORE ---
if col_t_Va in df.columns and col_t_Vb in df.columns:
    dt_LP = - (df[col_t_V_AB].values - df[col_t_Va].values) * 1e-6 
    dt_HP = - (df[col_t_Vb].values - df[col_t_Va].values) * 1e-6
    
    fase_LP_exp = 2 * np.pi * f_dati * dt_LP
    fase_HP_exp = 2 * np.pi * f_dati * dt_HP
    
    # 1. Incertezza sui singoli canali basata sulla pendenza dV/dt e sul limite dello strumento
    sigma_t_A = np.sqrt((sigma_V / (2 * np.pi * f_dati * Va))**2 + sigma_t_str**2)
    sigma_t_B = np.sqrt((sigma_V / (2 * np.pi * f_dati * Vb))**2 + sigma_t_str**2)
    sigma_t_AB = np.sqrt((sigma_V / (2 * np.pi * f_dati * V_AB))**2 + sigma_t_str**2) # Per il Passa-Basso (V_AB)
    
    # 2. Propagazione su Delta_t
    sigma_dt_LP = np.sqrt(sigma_t_A**2 + sigma_t_AB**2)
    sigma_dt_HP = np.sqrt(sigma_t_A**2 + sigma_t_B**2)
    
    # 3. Incertezza definitiva sulla fase (in radianti)
    err_fase_LP = 2 * np.pi * f_dati * sigma_dt_LP
    err_fase_HP = 2 * np.pi * f_dati * sigma_dt_HP
    
    DATO_FASE_PRESENTE = True
else:
    DATO_FASE_PRESENTE = False
    print("Colonne del tempo non trovate o non configurate.")

p0_C = 1e-7  

# --- FIT MODULO PASSA-BASSO ---
popt_LP, pcov_LP = curve_fit(passa_basso_modolo, f_dati, H_LP_exp, p0=[p0_C], sigma=err_H_LP, absolute_sigma=True)
C_fit_LP, C_err_LP = popt_LP[0], np.sqrt(pcov_LP[0][0])

res_LP_mod = H_LP_exp - passa_basso_modolo(f_dati, C_fit_LP)
chi2_LP_mod = np.sum((res_LP_mod / err_H_LP) ** 2)
dof_LP_mod = len(f_dati) - 1
chi2_red_LP_mod = chi2_LP_mod / dof_LP_mod
p_val_LP_mod = 1 - chi2.cdf(chi2_LP_mod, dof_LP_mod)

# --- FIT MODULO PASSA-ALTO ---
popt_HP, pcov_HP = curve_fit(passa_alto_modolo, f_dati, H_HP_exp, p0=[p0_C], sigma=err_H_HP, absolute_sigma=True)
C_fit_HP, C_err_HP = popt_HP[0], np.sqrt(pcov_HP[0][0])

res_HP_mod = H_HP_exp - passa_alto_modolo(f_dati, C_fit_HP)
chi2_HP_mod = np.sum((res_HP_mod / err_H_HP) ** 2)
dof_HP_mod = len(f_dati) - 1
chi2_red_HP_mod = chi2_HP_mod / dof_HP_mod
p_val_HP_mod = 1 - chi2.cdf(chi2_HP_mod, dof_HP_mod)

print("--- RISULTATI DEI FIT (MODULO) ---")
print(f"C (Passa-Basso): {C_fit_LP*1e9:.2f} ± {C_err_LP*1e9:.2f} nF | Chi2: {chi2_LP_mod:.2f}, Chi2_red: {chi2_red_LP_mod:.2f}, p-value: {p_val_LP_mod:.4f}")
print(f"C (Passa-Alto):  {C_fit_HP*1e9:.2f} ± {C_err_HP*1e9:.2f} nF | Chi2: {chi2_HP_mod:.2f}, Chi2_red: {chi2_red_HP_mod:.2f}, p-value: {p_val_HP_mod:.4f}\n")

# --- STATISTICHE FASE ---
if DATO_FASE_PRESENTE:
    res_LP_fase = fase_LP_exp - passa_basso_fase(f_dati, C_fit_LP)
    chi2_LP_fase = np.sum((res_LP_fase / err_fase_LP) ** 2)
    dof_LP_fase = len(f_dati)
    chi2_red_LP_fase = chi2_LP_fase / dof_LP_fase
    p_val_LP_fase = 1 - chi2.cdf(chi2_LP_fase, dof_LP_fase)

    res_HP_fase = fase_HP_exp - passa_alto_fase(f_dati, C_fit_HP)
    chi2_HP_fase = np.sum((res_HP_fase / err_fase_HP) ** 2)
    dof_HP_fase = len(f_dati)
    chi2_red_HP_fase = chi2_HP_fase / dof_HP_fase
    p_val_HP_fase = 1 - chi2.cdf(chi2_HP_fase, dof_HP_fase)
    
    print("--- RISULTATI COMPATIBILITÀ FASE ---")
    print(f"Fase Passa-Basso | Chi2: {chi2_LP_fase:.2f}, Chi2_red: {chi2_red_LP_fase:.2f}, p-value: {p_val_LP_fase:.4f}")
    print(f"Fase Passa-Alto  | Chi2: {chi2_HP_fase:.2f}, Chi2_red: {chi2_red_HP_fase:.2f}, p-value: {p_val_HP_fase:.4f}\n")


# --- GRAFICI MODULO ---
f_fit = np.logspace(np.log10(min(f_dati)), np.log10(max(f_dati)), 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

x_min_fissato = min(f_dati) * 0.7
x_max_fissato = max(f_dati) * 1.4

ax1.errorbar(f_dati, H_LP_exp, yerr=err_H_LP, fmt='o', color='blue', label='Dati Passa-Basso (V_AB/V_A)', capsize=3)
ax1.plot(f_fit, passa_basso_modolo(f_fit, C_fit_LP), 'b-', label='Fit Passa-Basso')
ax1.errorbar(f_dati, H_HP_exp, yerr=err_H_HP, fmt='s', color='red', label='Dati Passa-Alto (V_B/V_A)', capsize=3)
ax1.plot(f_fit, passa_alto_modolo(f_fit, C_fit_HP), 'r-', label='Fit Passa-Alto')
ax1.set_title("Modulo della Funzione di Trasferimento (Lineare)")
ax1.set_xlabel("Frequenza [Hz]")
ax1.set_ylabel("|H(f)|")
ax1.grid(True, which="both", ls="--")
ax1.legend(loc='lower center')

ax2.loglog(f_fit, passa_basso_modolo(f_fit, C_fit_LP), 'b-')
ax2.errorbar(f_dati, H_LP_exp, yerr=err_H_LP, fmt='o', color='blue', capsize=3, label='Dati Passa-Basso')
ax2.loglog(f_fit, passa_alto_modolo(f_fit, C_fit_HP), 'r-')
ax2.errorbar(f_dati, H_HP_exp, yerr=err_H_HP, fmt='s', color='red', capsize=3, label='Dati Passa-Alto')
ax2.set_title("Modulo della Funzione di Trasferimento (Log-Log)")
ax2.set_xlabel("Frequenza [Hz]")
ax2.set_ylabel("|H(f)|")
ax2.set_xlim(x_min_fissato, x_max_fissato)
y_min_log = min(np.min(H_LP_exp), np.min(H_HP_exp)) * 0.5
ax2.set_ylim(max(1e-3, y_min_log), 1.5)
ax2.grid(True, which="both", ls="--")
ax2.legend(loc='lower left')

text_stats_mod = (
    f"  Cross-Over Stats\n"
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
plt.savefig('Fit_Modulo_RC.png', dpi=300)
plt.show()

# --- GRAFICO FASE ---
if DATO_FASE_PRESENTE:
    fig2, ax3 = plt.subplots(figsize=(8, 6.5))
    ax3.errorbar(f_dati, fase_LP_exp, yerr=err_fase_LP, fmt='o', color='blue', label='Fase Passa-Basso', capsize=3)
    ax3.plot(f_fit, passa_basso_fase(f_fit, C_fit_LP), 'b--')
    ax3.errorbar(f_dati, fase_HP_exp, yerr=err_fase_HP, fmt='s', color='red', label='Fase Passa-Alto', capsize=3)
    ax3.plot(f_fit, passa_alto_fase(f_fit, C_fit_HP), 'r--')
    ax3.set_xscale('log')
    ax3.set_xlim(x_min_fissato, x_max_fissato)
    ax3.set_title("Fase della Funzione di Trasferimento (Semi-Log)")
    ax3.set_xlabel("Frequenza [Hz]")
    ax3.set_ylabel("Fase [rad]")
    ax3.grid(True, which="both", ls="--")
    ax3.legend(loc='lower left')
    
    text_stats_fase = (
        f"  Phase Agreement Stats (Jitter Model)\n"
        f"  --------------------------------------------------\n"
        f"  Passa-Basso:                     Passa-Alto:\n"
        f"  Chi2 = {chi2_LP_fase:.2f} (dof={dof_LP_fase})      Chi2 = {chi2_HP_fase:.2f} (dof={dof_HP_fase})\n"
        f"  Chi2_red = {chi2_red_LP_fase:.2f}             Chi2_red = {chi2_red_HP_fase:.2f}\n"
        f"  p-value = {p_val_LP_fase:.4f}              p-value = {p_val_HP_fase:.4f}"
    )
    ax3.text(0.95, 0.95, text_stats_fase, transform=ax3.transAxes, fontsize=9, fontfamily='monospace',
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('Fit_Fase_RC.png', dpi=300)
    plt.show()