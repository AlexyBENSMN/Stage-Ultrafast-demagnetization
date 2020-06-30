# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:30:00 2020

@author: alexy
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import math as m

#Paramètres du pulse laser
ts=50E-15
sigma=ts/(2*m.sqrt(2*m.log(2)))
Pave=3.5E-3
frep=5E3
coefabs=8E6
A=(80E-6)**2
#Température initiale
Tinit=300
#Constante qui permet de modéliser les interfaces (une température à calculer de chaque côté très proche de l'interface)
eps=1E-11
#Propriétés du substrat
tausub=1E-9
Tsub=300

#Fonction qui représente le pulse laser, on rajoutera ensuite la fonction plus précise d'absorption
def P(z,t):
    return(Pave*coefabs*m.exp(-z*2.8E7)*m.exp(-t**2/(2*sigma**2))/(frep*A*m.sqrt(2*m.pi)*sigma))


#Fonction principale qui résoud le système d'équation
#Paramètre couche représente les différents paramètres de chaque couches considérées (c'est une liste de liste)
#Les paramètres 4 et 5 correspondent aux propriétés magnétiques, si la couche n'est pas magnétque, on met None
#longueur couche représente la liste des largeurs des couches
#divz correspond à l'echantillonage selon z pour chaque couche
#ttot correspond au temps total de simulation
#divt correspond à l'échantillonage en temps
def resolution(parametre_couche,longueur_couche,divz,ttot,divt):
    #N est le nombre de couches
    N=len(parametre_couche)
    pas=[]
    #deltat est le pas de temps
    deltat=ttot/divt
    #z sera la liste des positions des différents points d'échantillonage
    z=[]
    long=0
    #construction de la liste des pas d'espaces (ce n'est pas le même pour chaque couche) et de la liste des positions z
    #Cela prend en compte les points proches des interfaces (eps+lon et eps-long) avec long la position actuelle de la couche
    for k in range(N):
        pas.append(longueur_couche[k]/divz)
        z.append(eps+long)
        for i in range(1,divz):
            z.append(pas[-1]*i+long)
        long+=longueur_couche[k]
        z.append(long-eps)
    #on augmente divz de 1 car on rajoute un point à cause du dédoublement près des interfaces
    divz+=1
    #Initialisation des matrices pour les 2 températures étudiées
    Te=np.zeros((N*divz,divt))
    Tp=np.zeros((N*divz,divt))
    #Teint et Tpint sont des intermédiaires de calcul pour chaque couche à un temps donné (pas forcément utile mais cela m'a servi au début)
    Teint=np.array([])
    Tpint=np.array([])
    #Mise en place des conditions initiales en température
    for k in range(N*divz):
        Teint=np.append(Teint,Tinit)
        Tpint=np.append(Tpint,Tinit)
        Te[k][0]=Tinit
        Tp[k][0]=Tinit
    #Initialisation de la matrice de l'aimantation mij
    mij=np.zeros((N*divz,divt))
    #Condition initiale de l'aimantation : 1 dans les zones magnétiques et 0 sinon
    for l in range(N):
        for i in range(divz):
            if parametre_couche[l][5]!=None:
                mij[i+l*divz][0]=1.0
            else:
                mij[i+l*divz][0]=0
    #lambd correspond à l'indice de l'échantillonage en temps
    lambd=1
    #Boucle pour la résolution des équations
    while lambd*deltat<ttot:
        #On calcule indépendamment chaque couche
        for l in range(N):
            #On définit les différents paramètres pour la couche considérée
            (gamma,Cp,gep,k0)=(parametre_couche[l][0],parametre_couche[l][1],parametre_couche[l][2],parametre_couche[l][3])
            #Calcul du terme différentiel en z
            # les 3 listes suivantes sont des intermédiaires de calcul
            #deriv2 est la liste qui correspond à l'élement différentiel complet
            deriv=[0 for k in range(divz)]
            aux=[0 for k in range(divz)]
            deriv2=[0 for k in range(divz)]
            #On calcule différement au bord :
            #   Bord gauche : (u(i+1)-u(i))/dz
            deriv[0]=(Teint[divz*l+1]-Teint[divz*l])/pas[l]
            #   Bord droit : (u(i)-u(i-1))/dz
            deriv[-1]=(Teint[divz*(l+1)-1]-Teint[divz*(l+1)-2])/pas[l]
            aux[0]=(k0*Teint[l*divz]*deriv[0]/Tpint[l*divz])
            aux[-1]=(k0*Teint[(l+1)*divz-1]*deriv[-1]/Tpint[(l+1)*divz-1])
            #   Milieu : (u(i+1)-u(i-1))/2dz
            for i in range(1,divz-1):
                deriv[i]=(Teint[i+divz*l+1]-Teint[i+divz*l-1])/(2*pas[l])
                aux[i]=(k0*Teint[i+l*divz]*deriv[i]/Tpint[i+l*divz])
            deriv2[0]=(aux[1]-aux[0])/pas[l]
            deriv2[-1]=(aux[-1]-aux[-2])/pas[l]  
            for i in range(1,divz-1):
                deriv2[i]=(aux[i+1]-aux[i-1])/(2*pas[l])
            for i in range(divz):
                #résolution en différences finies explicite de Te
                Te[i+l*divz][lambd]=Teint[i+l*divz]+deltat*(deriv2[i]+gep*(Tpint[i+l*divz]-Teint[i+l*divz])+P(z[i+l*divz],deltat*lambd))/(gamma*Teint[i+l*divz])
                #résolution en différences finies explicite de Tp
                Tp[i+l*divz][lambd]=Tpint[i+l*divz]+deltat*gep*(Teint[i+l*divz]-Tpint[i+l*divz])/Cp-deltat*(Tpint[i+l*divz]-Tsub)/tausub
                Teint[i+l*divz]=Te[i+l*divz][lambd]
                Tpint[i+l*divz]=Tp[i+l*divz][lambd]
            #conditions de continuité aux interfaces pour Te et Tp
            if l!=0:
                Te[l*divz-1][lambd]=Te[l*divz][lambd]
                Tp[l*divz-1][lambd]=Tp[l*divz][lambd]
                Teint[l*divz-1]=Teint[l*divz]
                Tpint[l*divz-1]=Tpint[l*divz]
        #Dérivée nulle à l'interface avec l'air
        Te[1][lambd]=Te[0][lambd]
        Tp[1][lambd]=Tp[0][lambd]
        Teint[1]=Teint[0]
        Tpint[1]=Tpint[0]
        #Résolution de l'équation de l'aimantation
        for l in range(N):
                #Définition des paramètres pour l'aimantation
                R,Tc=parametre_couche[l][4],parametre_couche[l][5]
                #Résolution en différences finies explicite de mij
                #Aimantation nulle dans les matériaux non magnétiques
                for i in range(divz):
                    if Tc!=None:
                        mij[i+l*divz][lambd]=mij[i+l*divz][lambd-1]+deltat*R*(mij[i+l*divz][lambd-1])*(Tp[i+l*divz][lambd-1])*(1-(mij[i+l*divz][lambd-1])/m.tanh((mij[i+l*divz][lambd-1])*Tc/(Te[i+l*divz][lambd-1])))/Tc
                    else:
                        mij[i+l*divz][lambd]=0
        lambd+=1
    #liste des différents temps considérés
    temps=[deltat*k for k in range(divt)]
    return(Te,Tp,mij,temps,z)


#Définition des paramètres pour la simulation et appel de la fonction resolution
parametre_couche=[[100, 2.63E+6, 7.5E+16, 300, None, None],[300, 2.3E+6, 60E+16, 10.5, 1E13, 300]]
longueur_couche=[10E-9,10E-9]
divz=30
ttot=1000E-15
divt=100000
(Te,Tp,mij,temps,z)=resolution(parametre_couche,longueur_couche,divz,ttot,divt)

#redimensionnement pour avoir le graphe en nanomètre
for i in range(len(z)):
    z[i]=z[i]*1E9

#paramètres pour l'affichage, les deux premiers sont pour l'aimantation et les 2 d'après pour la température
yminm=-1
ymaxm=1
yminT=0
ymaxT=2000
zmin = 0                              
zmax =20

fig = plt.figure() # initialise la figure
#définition de la sous-figure pour l'aimantation
axism = fig.add_subplot(121,xlim =(zmin, zmax), ylim =(yminm,ymaxm),xlabel="Profondeur de l'échantillon(nm)",ylabel='Aimantation(M/Ms)',title="Evolution de l'aimantation sous l'influence d'un pulse laser")
#définition du tracé de l'aimantation
linem, = axism.plot([], [], lw = 3) 

#définition de la sous-figure pour les températures
axis = fig.add_subplot(122,xlim =(zmin, zmax), ylim =(yminT,ymaxT),xlabel="Profondeur de l'échantillon(nm)",ylabel='Température(K)',title="Evolution des températures du système sous l'influence d'un pulse laser")
#définition du tracé de Te
line1, = axis.plot([], [], lw = 3) 
#définition du tracé de Tp
line2, = axis.plot([], [], lw = 3)
axtext = fig.add_axes([0.0,0.95,0.1,0.05])
axtext.axis("off")
#Légende pour Te et Tp
line1.set_label('Température des électrons')
line2.set_label('Température du réseau')
axis.legend([line1,line2], [line1.get_label(),line2.get_label()])


#Définition du temps
time = axtext.text(0.5,0.5, str(0), ha="left", va="top")

def init():
    #Initialisation de l'animation
    linem.set_data([],[])
    line1.set_data([],[])
    line2.set_data([],[])
    return linem, line1, line2

def animate(i): 
    #Tracé des lignes verticales symbolisant les différentes couches
    for l in range(len(longueur_couche)):
        axis.plot([longueur_couche[l]*1E9,longueur_couche[l]*1E9],[yminT,ymaxT],color='black')
        axism.plot([longueur_couche[l]*1E9,longueur_couche[l]*1E9],[yminm,ymaxm],color='black')
    #Tracé pour chaque frame
    line1.set_data(z,Te[:,30*i])
    line2.set_data(z,Tp[:,30*i])
    linem.set_data(z,mij[:,30*i])
    #Affichage du temps
    time.set_text('t='+str(round(30*i*ttot/divt*1E15,3))+'fs')
    return linem, line1, line2, time, 

#Lancement de l'animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=int(divt/30), blit=True, interval=0.01)
ani
