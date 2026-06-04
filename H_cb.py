import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import pandas as pd 
current_dir = Path(__file__).parent
pi = np.pi



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