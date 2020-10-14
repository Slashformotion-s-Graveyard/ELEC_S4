# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:30:12 2020

@author: ANSQUER
@author: ROOS
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


Ks = [-1, 2,0.5]    # coefficient multiplicateur au numérateur sans unité (ce sera votre T0 ou votre Tinfini)
fc = 2000   # fréquence de cassure en Hz

wc = 2*np.pi*fc
w = np.logspace(1,7,1000) # permet de répartir les points de calculs régulièrement sur une echelle log. ici : 1000points  de 10^1 à 10^7
f = w/2/np.pi #génération d'un autre tableur de valeurs en Hz,correleées à w permettant un affichage des Bode en Hz


for K in Ks:
    den = [1/wc,1] #denominateur sous forme canonique : le 1er terme est d'ordre 1, le 2e d'ordre 0
    numLP=[K] #numerateur d'un passe bas
    numHP=[K/wc, 0] #numerateur d'un passe haut
    H=sig.lti(numHP,den) # definition du système linéaire à étudier , vous choisissez votre numérateur
    w, T = H.freqresp(w=w) #T représente la fonction de transfert complexe du systeme linéaire


    # diagrammes de Bode
    plt.figure(1)
    C1 = plt.loglog(f, abs(T), label ="module") #generation de la courbe de module en fonction de f
    C2 = plt.loglog(f, (abs(K)*np.ones(len(f))),color="blue", dashes=[6, 2],label ="THF" ) #asymptote BF (cas du passe bas) de module en pointillés
    C3 = plt.loglog(f, abs(K*f/fc),color="green", dashes=[6, 2],label ="TBF" )
    plt.xlabel("f(Hz)")
    plt.ylabel("amplification")
    plt.grid()
    plt.legend()


    plt.figure(2)
    C4= plt.semilogx(f,(180*np.angle(T))/np.pi, label ="argument") #generation de la courbe d'argument en fonction de f
    plt.xlabel("f(Hz)")
    plt.ylabel("déphasage")
    plt.grid()
    plt.legend()



    # reponse indicielle
    t_e = np.arange(0,10/wc,0.000001) #création du tableau d'instants pour le calcul : de -1/wc à 10/w0 avec des intervalles de 1µs
    [t,s] = H.step(T=t_e) # réponse indicielle du système sur la durée t_e
    #ajout de 2 points avant t=0 pour mieux visualiser les phénomènes à t =0
    t = np.hstack(([-0.001,-0.00001,0],t))
    s = np.hstack(([0,0,0],s))

    e = (t>=0) #création d'un vecteur de même taille que t. le code renvoie 0 si faux, 1 si vrai

    plt.figure(3)
    plt.plot(t*1000,e,label="échelon d'entrée", color='b') #axe en ms
    plt.plot(t*1000,s,label="reponse indicielle", color='r')
    plt.grid()
    plt.xlabel("temps (ms)")
    plt.ylabel('tensions en volts')
    plt.grid(which='both', axis='both')
    plt.legend()
plt.show()

