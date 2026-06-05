import numpy as np 


#in teoria gli errori della fase di tempo sono tutti uguali poiche oltre che a ipendere dalla stessa arcotangente sono solamente
# traslati si k * pi /2  #
def sigma_fase(f,L,C,R_tot,sigma_l,sigma_r,sigma_c,Dt):
    omega = f * 2 * np.pi
    ind = omega*L   #caratterizzatore dell'induttore nelle formule 
    cond = 1/(omega*C) #caratterizzatore del condensatore nelle formule 
    Dic = ind -cond  #caratterizzatore della differenza nelle formule 
    #sigma _ l 
    dl = R_tot*omega / (R_tot**2 + Dic**2)
    dc = R_tot/((R_tot**2+Dic**2)*(omega *C**2))
    #quest'ultimo va consderato due volte poiche gli errrori su R_ind e R_carico sono gli stessi 
    dr = Dic/(R_tot**2+Dic**2)

    sigma_dt =  np.sqrt((dl*sigma_l)**2+2*(dr*sigma_r)**2+(dc*sigma_c)**2)

    dfase = omega * sigma_dt
    return dfase