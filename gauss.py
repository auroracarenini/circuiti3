import matplotlib.pyplot as plt 
import numpy as np 

L  = [9.46,9.62,10.19,10.44,10.54]
sigma_L = [0.30,0.10,0.15,0.19,0.06]

C = [65.93,69.31,65.11,66.10,72.76]
sigma_C = [0.40,0.92,0.43,0.54,2.07]

L = np.asarray(L)
C = np.asarray(C)
sigma_C = np.asarray(sigma_C)
sigma_L = np.asarray(sigma_L)


peso_C = 1/sigma_C**2
mu_C = np.sum (peso_C*C)/np.sum(peso_C)
sigma_mu_C = 1/np.sqrt(np.sum(peso_C))


peso_L = 1/sigma_L**2
mu_L = np.sum(peso_L*L)/np.sum(peso_L)
sigma_mu_L = 1/np.sqrt(np.sum(peso_L))

print(f'il valore della capacità media è:  {mu_C:.2f} +/- {sigma_mu_C:.2f}')

print(f'il valore dell\'induttanza media è:  {mu_L:.2f} +/- {sigma_mu_L:.2f}')


# ------ SCRITTURA DEI T-TESTS ----------------

CT = 68
LT = 10

tc = np.abs(CT - mu_C)/sigma_mu_C
tl = np.abs(LT - mu_L)/sigma_mu_L

print ("T-TESTS: \n")
print(f'condensatore : {tc:.3f}\n')
print(f'induttore : {tl:.3f}')