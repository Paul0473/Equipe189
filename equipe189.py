import numpy as np 
import matplotlib.pyplot as plt 
from newton import newton

#Q1b

#Q1c
#figure 1

#Q1d
#figure 2

#Q2context
def vandermonde_modifie(points):
    n = len(points)
    V = np.zeros((n+1, n+1))
    
    for i in range(n+1):
        for j in range(n+1):
            V[i, j] = points[i]**(n-j)
    return V

#Q2a
#figure3
def plot_conditionnement():
    n_values = range(10, 151, 10)
    cond_values = []
    
    for n in n_values:
        points = np.linspace(-1, 1, n+1)
        V = vandermonde_modifie(points)
        cond = np.linalg.cond(V, 'fro')
        cond_values.append(cond)
    
    plt.figure()
    plt.semilogy(n_values, cond_values, label='cond(V(P))')
    plt.semilogy(n_values, 2**np.array(n_values), label='2^n', linestyle='--')
    plt.xlabel('n')
    plt.ylabel('Conditionnement')
    plt.legend()
    plt.title('Conditionnement de la matrice de Vandermonde modifiée')
    plt.show()

#Q2b
#figure4
def regression_conditionnement():
    n_values = range(10, 151, 10)
    cond_values = []
    
    for n in n_values:
        points = np.linspace(-1, 1, n+1)
        V = vandermonde_modifie(points)
        cond = np.linalg.cond(V, 'fro')
        cond_values.append(cond)
    
    log_cond = np.log(cond_values)
    coeffs = np.polyfit(n_values, log_cond, 1)
    
    b = np.exp(coeffs[1])
    print(f"Valeur estimée de b : {b}")
    
    plt.figure()
    plt.semilogy(n_values, cond_values, label='cond(V(P))')
    plt.semilogy(n_values, b**np.array(n_values), label=f'{b}^n', linestyle='--')
    plt.semilogy(n_values, 2**np.array(n_values), label='2^n', linestyle='--')
    plt.xlabel('n')
    plt.ylabel('Conditionnement')
    plt.legend()
    plt.title('Régression du conditionnement de V(P)')
    plt.show()

#Q2c
#figure5
def erreur_relative():
    n_values = range(30, 201)
    erreurs = []
    
    for n in n_values:
        points = np.linspace(-1, 1, n+1)
        V = vandermonde_modifie(points)
        b = np.ones(n+1)
        x_star = np.linalg.solve(V, b)
        x_exact = np.zeros(n+1)
        x_exact[-1] = 1
        
        erreur = np.linalg.norm(x_star - x_exact) / np.linalg.norm(x_exact)
        erreurs.append(erreur)
    
    plt.figure()
    plt.semilogy(n_values, erreurs, label='Erreur relative')
    plt.semilogy(n_values, (np.finfo(np.float64).eps) * (b**np.array(n_values)), label='Erreur machine', linestyle='--')
    plt.xlabel('n')
    plt.ylabel('Erreur relative')
    plt.legend()
    plt.title('Erreur relative pour la résolution du système Vandermonde modifié')
    plt.show()

plot_conditionnement()
regression_conditionnement()
erreur_relative()