import numpy as np
import sklearn as sk
from sklearn import preprocessing
from F_soft_impute import SoftImpute as FSoftImpute
from prime_pca import *



def Cov_reconstruction(Y, V_k, Omega, K, trace=False): 
    print("---------------------------------")
    print('\t Fancy Soft Impute')
    clf = FSoftImpute(max_rank=20, shrinkage_value=0.0, verbose=trace)
    clf.solve(Y, np.logical_not(Omega))
    print(sin_theta_distance(clf.V, V_k))
    print("----------")
    
    print('\t PrimePca')
    clf_p = PrimePca(K=K, trace=trace, max_iter=2000, thresh_convergence=10**(-5))
    clf_p.fit(Y)
    print(sin_theta_distance(clf_p.v, V_k))
    print("---------------------------------")

def H_One_missingness(Z=0, U=None, n = 2000, d = 500, K = 2):
    print('==> H1 missingness ')
    p = 0.05
    sig_u = np.zeros((2, 2)) 
    sig_u[0, 0] = sig_u[1, 1] = 100
    V_k = np.ones((d, K)) 
    V_k[250:, 1] = -1
    V_k /= np.sqrt(500)
    if(U is None):
        U = np.random.multivariate_normal([0, 0], sig_u, size=n)
    Omega = np.random.binomial(1, p, (n, d))

    X = U @ (V_k).T
    Y = X + Z
    Y_omega = Y.copy()
    Y_omega[np.logical_not(Omega)] = np.nan

    Cov_reconstruction(Y_omega, V_k, Omega, K)


def H_Two_missingness(Z = 0, U=None, n = 2000, d = 500, K = 2):
    print('==> H2 missingness ')

    P = np.random.uniform(low = 0, high = 0.2, size=n)
    Q = np.random.uniform(low = 0.05, high = 0.95, size=d) 
    Omega = np.zeros((n, d))
    Omega_proba = np.zeros((n, d))

    for i in range(n):
        for j in range(d):
            Omega_proba[i, j] = P[i]*Q[j]
            Omega[i, j] = np.random.binomial(size=1, n=1, p=Omega_proba[i, j])
            #np.random.binomial(n=1, prob=Omega_proba[i, j], size=1)
       

    sig_u = np.zeros((K, K)) 
    np.fill_diagonal(sig_u, 100)
    if(U is None):
        U = np.random.multivariate_normal([0, 0], sig_u, size=n)

    V_k = np.ones((d, K))
    V_k[250:, 1] = -1
    V_k = V_k * 1/np.sqrt(500)

    X = U @ V_k.T
    Y = X + Z

    Y_omega = Y.copy()
    Y_omega[np.logical_not(Omega)] = np.nan

    Cov_reconstruction(Y_omega, V_k, Omega, K)


def H_Three_missingness(Z = 0, U = None, n = 2000, d = 500, K = 2):
    print('==> H3 missingness ')

    p_e = 0.01
    p_o = 0.19
    sig_u = np.zeros((2, 2)) 
    sig_u[0, 0] = sig_u[1, 1] = 100
    V_k = np.ones((d, K)) 
    V_k[250:, 1] = -1
    V_k /= np.sqrt(500)
    if(U is None):
        U = np.random.multivariate_normal([0, 0], sig_u, size=n)
    Omega = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            p = p_e
            if(j %2 ==1): p = p_o
            Omega[i, j] = np.random.binomial(size=1, n=1, p=p)

    # Omega = np.random.binomial(1, p_e, (n, d))
    # Omega[:, 1::2] = np.random.binomial(1, p_o, (n, d//2))

    X = U @ V_k.T
    Y = X + Z

    Y_omega = Y.copy()
    Y_omega[np.logical_not(Omega)] = np.nan

    Cov_reconstruction(Y_omega, V_k, Omega, K)

def H_Four_missingness(Z = 0, U=None, n = 2000, d = 500, K = 2):
    print('==> H4 missingness ')

    p_e = 0.02
    p_o = 0.18
    sig_u = np.zeros((2, 2)) 
    sig_u[0, 0] = sig_u[1, 1] = 100
    V_k = np.ones((d, K)) 
    V_k[250:, 1] = -1
    V_k /= np.sqrt(500)
    if(U is None):
        U = np.random.multivariate_normal([0, 0], sig_u, size=n)
    Omega = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            p = p_e
            if(i%2 ==1): p = p_o
            Omega[i, j] = np.random.binomial(size=1, n=1, p=p)

    X = U @ V_k.T
    Y = X + Z

    Y_omega = Y.copy()
    Y_omega[np.logical_not(Omega)] = np.nan

    Cov_reconstruction(Y_omega, V_k, Omega, K)


def main():
    print("--- No Noise Case ---")
    H_One_missingness()
    H_Two_missingness()
    H_Three_missingness()
    H_Four_missingness()

    d, n, K, nu = 500, 2000, 2, 20
    print("--- Noisy Case with nu = ", nu, " ---")
    Z = np.random.multivariate_normal(np.zeros(d), np.identity(d), size=n)
    U = np.random.multivariate_normal(np.zeros(K), nu**2*np.identity(K), size=n)
    H_One_missingness(Z=Z, U=U, d=d, n=n, K=K)
    H_Two_missingness(Z=Z, U=U, d=d, n=n, K=K)
    H_Three_missingness(Z=Z, U=U, d=d, n=n, K=K)
    H_Four_missingness(Z=Z, U=U, d=d, n=n, K=K)


    d, n, K, nu = 500, 2000, 2, 40
    print("--- Noisy Case with nu = ", nu, " ---")
    Z = np.random.multivariate_normal(np.zeros(d), np.identity(d), size=n)
    U = np.random.multivariate_normal(np.zeros(K), nu**2*np.identity(K), size=n)
    H_One_missingness(Z=Z, U=U, d=d, n=n, K=K)
    H_Two_missingness(Z=Z, U=U, d=d, n=n, K=K)
    H_Three_missingness(Z=Z, U=U, d=d, n=n, K=K)
    H_Four_missingness(Z=Z, U=U, d=d, n=n, K=K)

 
    d, n, K, nu = 500, 2000, 2, 60
    print("--- Noisy Case with nu = ", nu, " ---")
    Z = np.random.multivariate_normal(np.zeros(d), np.identity(d), size=n)
    U = np.random.multivariate_normal(np.zeros(K), nu**2*np.identity(K), size=n)
    H_One_missingness(Z=Z, U=U, d=d, n=n, K=K)
    H_Two_missingness(Z=Z, U=U, d=d, n=n, K=K)
    H_Three_missingness(Z=Z, U=U, d=d, n=n, K=K)
    H_Four_missingness(Z=Z, U=U, d=d, n=n, K=K)

if __name__ == '__main__':
    main()