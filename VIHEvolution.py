"""
    Nom du fichier : VIHEvolution.py
    Auteur(s) : Astruc Lélio
    Date de création : 11/05/2023
    Dernière mise à jour : 11/05/2023
    Version : 1

Description : Implémentation en python du modèle de Abba Gumel (https://accromath.uqam.ca/2013/04/propagation-et-contro%cc%82le-du-vih-un-modele-mathematique/) permettant de comprendre les dynamiques d'évolutions
"""


#https://accromath.uqam.ca/2013/04/propagation-et-contro%CC%82le-du-vih-un-modele-mathematique/
import numpy as np
import matplotlib.pyplot as plt


        ### Conditions initiales
        
    ## Populations initiales
# Nombre d'individus sains
S0 = 100
# Nombre d'individus infectés inconscients
I_N0 = 100
# Nombre d'individus infectés conscients mais asymptomatiques
I_C0 = 10
# Nombre d'individus infectés symptomatiques ne suivant pas de sensibilisation
M_N0 = 50
# Nombre d'individus infectés symptomatiques suivant une sensibilisation
M_C0 = 5
# Nombre d'individus infectés symptomatiques sous traitement pharmaceutique (ARV)
M_T0 = 2
# Population initiale
popInitiale = S0 + I_N0 + I_C0 + M_N0 + M_C0 + M_T0

    ## Intervalles de temps et pas de discrétisation h
tempsInitial = 0
tempsFinal = 500
h = 0.01
N = int((tempsFinal - tempsInitial) / h) + 1
t = np.linspace(tempsInitial, tempsFinal, N)

##################################################################################################################################################################################################################

        ### Coefficients initaux

    ## Usage des préservatifs
# Taux d'utilisation des préservatifs
alpha = 0.1
# Efficacité d'un préservatif (100% d'efficacité si le préservatif ne se déchire pas et on considère (d'après ChatGPT) qu'un préservatif a un risque de rupture de 2%)
eps = 0.98

    ## Sensibilisation
# Nombre de personnes participants à des sensibilisations sans pour autant être infectées ou au courant de l'être
gamma = 0.1*popInitiale

    ##Taux de passage d'un cas asymptomatique (I_x) à un cas symptomatique (M_x)
sigma = 0.5

    ## Traitement pharamaceutique
tau_N = 0.05
tau_C = 0.1

    ## Taux d'infection cf. ChatGPT : "Le pourcentage de transmission du VIH lors d'un rapport sexuel non protégé dépend de plusieurs facteurs tels que la charge virale de la personne atteinte du VIH, le type d'activité sexuelle (pénétration vaginale, pénétration anale, sexe oral), la présence d'autres infections sexuellement transmissibles, etc. Cependant, selon les données scientifiques actuelles, le risque de transmission lors d'un rapport sexuel non protégé varie entre 0,04% et 4% par contact, selon ces facteurs."
beta = 0.04

    ## Taux de mortalité naturelle
mu = 0.007

    ## Taux de mortalité pour les classes N et C
delta = 0.7

    ## Efficacité du traitement pour réduire la mortalité
psi = 0.7

    ## Potentiel d'infection de chaque classe
i_C = 0.01
m_N = 0.5
m_C = 0.5
m_T = 0.5

    ## Coefficitents de transitions entre les classes ? 
K_1 = sigma + gamma + mu
K_2 = sigma + mu
K_3 = mu + delta + tau_N
K_4 = mu + delta + tau_C
K_5 = mu + psi*delta


##################################################################################################################################################################################################################

    ## Définition des équations différentielles ordinaires
def SPoint(I_N, I_C, M_N, M_C, M_T,S):
    return popInitiale - ((beta*(1-eps*alpha)*(I_N + i_C*I_C + m_N*M_N + m_C*M_C + m_T*M_T)*S)/popInitiale) - mu*S

def I_NPoint(I_N, I_C, M_N, M_C, M_T,S):
    return ((beta*(1-eps*alpha)*(I_N + i_C*I_C + m_N*M_N + m_C*M_C + m_T*M_T)*S)/popInitiale) - K_1*I_N

def I_CPoint(I_N, I_C):
    return gamma*I_N-K_2*I_C

def M_NPoint(I_C, M_N):
    return sigma*I_C - K_3*M_N
    
def M_CPoint(I_C, M_C):
    return sigma*I_C - K_4*M_C

def M_TPoint(M_N, M_C, M_T): 
    return tau_N*M_N + tau_C*M_C-K_5*M_T

    ## Définition de la méthode de résolution du système d'EDO
#On utilise ici une méthode d'Euler explicite
def eulerSolve(S0, I_N0, I_C0, M_N0, M_C0, M_T0, h, N):
    S = np.zeros(N)
    I_N = np.zeros(N)
    I_C = np.zeros(N)
    M_N = np.zeros(N)
    M_C = np.zeros(N)
    M_T = np.zeros(N)
    S[0] = S0
    I_N[0] = I_N0
    I_C[0] = I_C0
    M_N[0] = M_N0
    M_C[0] = M_C0
    M_T[0] = M_T0
    for n in range(1, N):
        S[n] = S[n-1] + h*SPoint(I_N[n-1], I_C[n-1], M_N[n-1], M_C[n-1], M_T[n-1],S[n-1])
        I_N[n] = I_N[n-1] + h*I_NPoint(I_N[n-1], I_C[n-1], M_N[n-1], M_C[n-1], M_T[n-1],S[n-1])
        I_C[n] = I_C[n-1] + h*I_CPoint(I_N[n-1], I_C[n-1])
        M_N[n] = M_N[n-1] + h*M_NPoint(I_C[n-1], M_N[n-1])
        M_C[n] = M_C[n-1] + h*M_CPoint(I_C[n-1], M_C[n-1])
        M_T[n] = M_T[n-1] + h*M_TPoint(M_N[n-1], M_C[n-1], M_T[n-1])
    survivants = S[n] + I_N[n] + I_C[n] + M_N[n] + M_C[n] + M_T[n]
    return S, I_N, I_C, M_N, M_C, M_T, survivants

S, I_N, I_C, M_N, M_C, M_T, survivants = eulerSolve(S0, I_N0, I_C0, M_N0, M_C0, M_T0, h, N)

    ##Affichage graphique
plt.figure()
plt.plot(t, S, label = 'Individus sains')
plt.plot(t, I_N, label = 'Individus récemment infectés inconscients')
plt.plot(t, I_C, label = 'Individus récemment infectés conscients asymptomatiques')
plt.plot(t, M_N, label = 'Individus sympotmatiques sans sensibilisation')
plt.plot(t, M_C, label = 'Individus symptomatiques avec sensibilisation')
plt.plot(t, M_T, label = 'Individus sous traitement pharmaceutique (ARV)')
plt.xlabel('Temps')
plt.ylabel('Populations')
plt.legend()
plt.title('Evolution des classes de populations en fonction du temps')
plt.show()
