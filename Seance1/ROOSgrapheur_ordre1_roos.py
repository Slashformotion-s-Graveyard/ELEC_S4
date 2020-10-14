import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def run(filter_type, k, fc, margin = 5):
    ######################################
    # PROTECTIONS AGAINST WRONGS ARGUMENTS
    ######################################
    if not isinstance(filter_type, str):
        raise TypeError("filter_type is either one of thoses strings: 'HP', 'LP'.")
    if not isinstance(k, (float, int)):
        raise TypeError("k should be a float or an int.")
    if not isinstance(fc, (float, int)):
        raise TypeError("fc should be a float or an int.")
    if not isinstance(margin, (float, int)):
        raise TypeError("margin should be a float or an int.")
    if fc <= 0 :
        raise ValueError("fc must be greater than 0.")
    if margin <= 0 :
        raise ValueError("margin must be greater than 0.")
    


    ###############################
    # SETTING UP FIGURE 
    ###############################
    fig =plt.figure(figsize=(200,200)) # the window will take the entire screen 
    fig.subplots_adjust(bottom=0.025, left=0.050, top = 0.975, right=0.975)
    nrows = 3
    ncols = 2
    X = [ (nrows,ncols,(1,2)), (nrows,ncols,(3,4)), (nrows,ncols,5), (nrows,ncols,6)]
    subs = []
    for nrows, ncols, plot_number in X:
        sub = fig.add_subplot(nrows, ncols, plot_number)
        sub.set_xticks([])
        sub.set_yticks([])
        subs.append(sub)

    # The four plots to use    
    amplification = subs[0]
    dephasage = subs[1]
    indicielle_rep = subs[2]
    impulse_rep = subs[3]

    ###############################
    # CREATING NECESSARY ARRAYS AND THE LTI OBJECT
    ###############################

    
    wc = 2*np.pi*fc


    order_fc = fc//1000
    min_decade = order_fc-margin if order_fc-margin > 0.01 else 0.01
    max_decade = order_fc+margin//3
    
    w = np.logspace(min_decade,max_decade,20000) # permet de répartir les points de calculs régulièrement sur une echelle log. ici : 1000points  de 10^1 à 10^7
    f = w/2/np.pi #génération d'un autre tableur de valeurs en Hz,correleées à w permettant un affichage des Bode en Hz

    den = [1/wc,1] #denominateur sous forme canonique : le 1er terme est d'ordre 1, le 2e d'ordre 0
    
    numHP=[k/wc, 0] #numerateur d'un passe haut
    numLP=[k] #numerateur d'un passe bas
        
    # we choose HP or LP
    if filter_type == 'HP':
        num_chosen = numHP
    elif filter_type == 'LP':
        num_chosen = numLP
    
    H=sig.lti(num_chosen,den) # definition du système linéaire à étudier , vous choisissez votre numérateur
    w, T = H.freqresp(w=w) #T représente la fonction de transfert complexe du systeme linéaire
    amplification.loglog(f, abs(T),color="red", label ="module")
    plt.show()


if __name__ == "__main__":
    run(filter_type='HP', fc=10000, k=1, margin=1) 
    ## filter_type is either 'HP' or 'LP'.
        # Exemple: filter_type='HP'
    ## k is the amplification,  ints or floats. 
        # Exemple: k=1 or k=-2.5
    ## fc is the cuttoff frequency, use ints or floats. 
        # Exemple: fc=100 or f=125,125
        # fc MUST be greater than 0