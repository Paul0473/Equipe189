import numpy as np

def newton(fonction_racine, derivee_fonction_racine, x0, tolr=1e-10, nmax=100):
    length_x0 = len(x0)
    if length_x0 != 1:
        iterations = np.array([x0]).reshape(length_x0,1)
        prochaine_iteration = x0-np.linalg.solve(derivee_fonction_racine(x0),fonction_racine(x0))
        iterations = np.append(iterations,prochaine_iteration.reshape(length_x0,1),1)
        n = 1
        eps = np.finfo(np.float64).eps
        err_rel = np.linalg.norm(iterations[:,n]-iterations[:,n-1])/(np.linalg.norm(iterations[:,n])+eps)
        while (err_rel >= tolr and n < nmax):
            prochaine_iteration = iterations[:,n]-np.linalg.solve(derivee_fonction_racine(iterations[:,n]),
                                                    fonction_racine(iterations[:,n]))
            iterations = np.append(iterations,prochaine_iteration.reshape(length_x0,1),1)
            err_rel = np.linalg.norm(iterations[:,n+1]-iterations[:,n])/(np.linalg.norm(iterations[:,n+1])+eps)
            n += 1
        return iterations
    elif length_x0 == 1:
        iterations = np.array([x0]).reshape(length_x0,1)
        prochaine_iteration = x0-fonction_racine(x0)/derivee_fonction_racine(x0)
        iterations = np.append(iterations,prochaine_iteration.reshape(length_x0,1),1)
        n = 1
        eps = np.finfo(np.float64).eps
        err_rel = np.linalg.norm(iterations[:,n]-iterations[:,n-1])/(np.linalg.norm(iterations[:,n])+eps)
        while (err_rel >= tolr and n < nmax):
            prochaine_iteration = iterations[:,n]-fonction_racine(iterations[:,n])/derivee_fonction_racine(iterations[:,n])
            iterations = np.append(iterations,prochaine_iteration.reshape(length_x0,1),1)
            err_rel = np.linalg.norm(iterations[:,n+1]-iterations[:,n])/(np.linalg.norm(iterations[:,n+1])+eps)
            n += 1
        return iterations