import numpy as np 
import matplotlib.pyplot as plt 
from newton import newton

# Partie 1 : Recherche des racines avec Newton
# Q1b

np.set_printoptions(precision=16)

data_A = np.array([(1, 1), (0, 0.73205), (-0.73205, 0)])
data_B = np.array([(-1, 0), (-1.0066, 0.1147), (-1.136, 0.50349)])
data_C = np.array([(0, 1), (-0.0112, 1.149247), (-0.0465, 1.301393)])

datasets = {"A": data_A, "B": data_B, "C": data_C}

def f(x, data):
    h, k, r = x
    return np.array([(p[0] - h)**2 + (p[1] - k)**2 - r**2 for p in data])

def J(x, data):
    h, k, r = x
    return np.array([[-2 * (p[0] - h), -2 * (p[1] - k), -2 * r] for p in data])

x0 = np.array([0, 0, 1])
tol = 1e-10
Nmax = 100

for name, data in datasets.items():
    print(f"\nJeu de données {name}:")
    
    def fonction_racine(x):
        return f(x, data)

    def derivee_fonction_racine(x):
        return J(x, data)

    iterates = newton(fonction_racine, derivee_fonction_racine, x0, tol, Nmax)
    solution = iterates[:, -1]

    det_J = np.linalg.det(J(solution, data))
    
    print(f"h = {solution[0]}, k = {solution[1]}, r = {solution[2]}")
    print(det_J)

# Q1c
err_A = []
err_B = []
err_C = []

for name, data in datasets.items():
    def fonction_racine(x):
        return f(x, data)
    
    def derivee_fonction_racine(x):
        return J(x, data)

    iterates = newton(fonction_racine, derivee_fonction_racine, x0, tol, Nmax)

    err = [np.linalg.norm(iterates[:, i+1] - iterates[:, i]) for i in range(iterates.shape[1] - 1)]
    
    if name == "A":
        err_A = err
    elif name == "B":
        err_B = err
    else:
        err_C = err

plt.figure(figsize=(8, 6))

plt.semilogy(err_A, label='Jeu de données A', color='r')
plt.semilogy(err_B, label='Jeu de données B', color='g')
plt.semilogy(err_C, label='Jeu de données C', color='b')

plt.xlabel('Nombre d\'itérations')
plt.ylabel('Erreur \(E_n\)')
plt.title('Figure 1: Erreur en échelle semilogarithmique pour chaque jeu de données')
plt.legend()

plt.grid(True)
plt.show()


# Q1d
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(data_A[:, 0], data_A[:, 1], label="Jeu A", color='r')
ax.scatter(data_B[:, 0], data_B[:, 1], label="Jeu B", color='g')
ax.scatter(data_C[:, 0], data_C[:, 1], label="Jeu C", color='b')

for name, data in datasets.items():
    def fonction_racine(x):
        return f(x, data)
    
    def derivee_fonction_racine(x):
        return J(x, data)

    iterates = newton(fonction_racine, derivee_fonction_racine, x0, tol, Nmax)
    solution = iterates[:, -1]

    h, k, r = solution
    circle = plt.Circle((h, k), r, color='black', fill=False, linestyle='--', linewidth=1.5)
    ax.add_patch(circle)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Figure 2: Cercles correspondant aux solutions trouvées')
ax.legend()

plt.grid(True)
plt.axis('equal')
plt.show()




# Partie 2 : Matrice de Vandermonde modifiée

def vandermonde_modifie(points):
    n = len(points) - 1  # Correction ici
    V = np.zeros((n+1, n+1))
    
    for i in range(n+1):
        for j in range(n+1):
            V[i, j] = points[i]**(n-j)
    
    return V

# Q2a (Figure 3)
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
    plt.title('Figure 3: Conditionnement de la matrice de Vandermonde modifiée')
    plt.show()

# Q2b (Figure 4)
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
    plt.title('Figure 4: Régression du conditionnement de V(P)')
    plt.show()

# Q2c (Figure 5)
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
    plt.semilogy(n_values, (np.finfo(np.float64).eps) * (2**np.array(n_values)), label='Erreur machine', linestyle='--')
    plt.xlabel('n')
    plt.ylabel('Erreur relative')
    plt.legend()
    plt.title('Figure 5: Erreur relative pour la résolution du système Vandermonde modifié')
    plt.show()

# Exécution des fonctions
plot_conditionnement()
regression_conditionnement()
erreur_relative()
