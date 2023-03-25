'''
   No Copyright, feel free to use this code

   Implemention of the prime pca algorythme from the paper : 
   Lacking values need to be set at np.nan !
   The code is optimized using either sklearn functions or numba to speed up the numpy calls. We end up with a solution which 
   is in average two to three times faster then the original R implementation

'''
from __future__ import print_function
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from numba import njit
from sklearn.utils.extmath import randomized_svd

from numba.core.errors import NumbaPerformanceWarning, NumbaPendingDeprecationWarning
import warnings

from testing import *

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@njit
def get_omega(X, thresh=10**(-6)): 
    return np.abs(X)>thresh 


@njit
def filter_row(row, V, thresh, prob): 
    d = (V.shape)[0]
    K = (V.shape)[1]
    obs_num = np.sum(row)
    if (obs_num <= K):
        return(False)

    V_select = V[row, :]
    _, sigmas, _ = np.linalg.svd(V_select)
    
    sigmas = sigmas.flatten()

    if(sigmas[-1] >= np.sqrt(obs_num/d)/thresh):
        rv = np.random.uniform(0, 1)
        if(rv < prob): return True
    return False

@njit
def sample_filter_op(thresh, V, Omega, prob = 1):
    n = Omega.shape[0]
    rows = np.zeros(n)
    for i in range(n) : 
        rows[i] = filter_row(row = Omega[i, :], V=V, thresh=thresh, prob=prob)
    return rows

@njit
def select_rows(X, rows): 
    return X[np.where(rows)]

@njit
def mse_eval(V_new, Xs, Os): 
  K = V_new.shape[1]
  residuals = (complete(V_new, Xs, Os, K)[1]).flatten()
  return(np.mean(np.square(residuals)))

@njit
def complete(V_cur, X, Os, K, trace = False): 
    n = X.shape[0]
    d = X.shape[1]
    X_complete = X.copy()
    residuals = np.zeros_like(X)
    
    U_hat = np.zeros((n, K))
    for i in range(n): 
        omega_row = Os[i, :]
        x = X[i]
        x = x[omega_row].reshape(np.sum(omega_row), 1)
        V_part  = V_cur[omega_row, :]
        u_hat = np.linalg.pinv(V_part) @ x
        x_hat = (V_cur @ u_hat).flatten()
        residual = np.zeros(d)
        residual[omega_row] = (x.flatten() - x_hat[omega_row])
        residuals[i] = residual
        x_temp = X_complete[i]
        x_temp[np.logical_not(omega_row)] = x_hat[np.logical_not(omega_row)]
        X_complete[i] = x_temp
        U_hat[i] = u_hat.flatten()
        
    return X_complete, residuals, U_hat

def svd_step(X, K, n_power_iterations=2):
        """
        Returns the V max ranked
        """
        #  Perform the faster randomized SVD
        (U, s, V) = randomized_svd(
            X,
            K,
            n_iter=n_power_iterations,
            random_state=None)
        return V.T

def eigen_refine(V_cur, X, Os, trace = False, thresh = 1e-10)  : 
    K = (V_cur.shape)[1]
    X_completed = complete(V_cur, X, Os, K, trace = trace)[0]
    V_new = svd_step(X_completed, K)
    return V_new


def col_scale(X, center = True, normalize = False):
    mask = np.isnan(X)
    X_center = preprocessing.scale(X, axis=0, with_mean=center, with_std=normalize, copy=True)
    X_center[mask] = 0
    return(X_center)


def inverse_prob_method(X, K, trace = False, center = True, normalize = False, thresh=10^(-5)):
    n, d=X.shape
    if (np.any(np.sum(np.isnan(X), axis=1) == n) or np.any(np.sum(np.abs(X)<thresh, axis=1) == n)):
        print("There exists at least one all-NA column. Please screen this out first.\n")
        return(None)
    
    if ((np.any(np.sum(np.logical_not(np.isnan(X)) , axis=1) == 1) or np.any(np.sum(np.abs(X)>thresh, axis=1) == 1)) and (normalize)):
        print("There exists at least one column with only one NA, so that the normalisation cannot be applied. Please screen this out first.\n")
        return(None)
    
    X_center = col_scale(X, center, normalize)
    Omega = get_omega(X_center)
    Sigma_x = X.T @ X
    Sigma_o = Omega.T @ Omega
    Sigma_o[np.abs(Sigma_o) < 10**(-10)] = np.Inf
    weight = 1/Sigma_o
    weight[weight == np.Inf] = 0
    Sigma_tilde = Sigma_x
    V_sigma = svd_step(Sigma_tilde, K)
    return V_sigma

@njit
def sin_theta_distance (V1, V2) :
    K = min((V1.shape)[1], (V2.shape)[1])
    sigmas = (np.linalg.svd(V1.T@V2)[1]).flatten()
    return(np.sqrt(K - min(np.sum(np.square(sigmas)), K)))


class PrimePca:
    def __init__(self,  K, V_init = None, thresh_sigma = 10, max_iter = 1000, 
  thresh_convergence = 1e-05, thresh_als = 1e-10, trace = False, 
  prob = 1, center = True, normalize = False):
        self.K = K
        self.V_init = V_init
        self.thresh_sigma = thresh_sigma
        self.max_iter = max_iter
        self.thresh_convergence = thresh_convergence
        self.thresh_als = thresh_als
        self.trace = trace
        self.prob = prob
        self.center = center
        self.normalize = normalize
        self.v = None
        self.loss_all = None
        self.step_cur = 0

    def fit(self, X, Omega=None):
        n,m = X.shape
        if (np.any(np.sum(np.isnan(X), axis=1) == n)):
            print("There exists at least one all-NA column. Please screen this out first.\n")
            return(None)
    
        if (np.any(np.sum(np.logical_not(np.isnan(X)), axis=1) == 1) and (self.normalize)):
            print("There exists at least one column with only one NA, so that the normalisation cannot be applied. Please screen this out first.\n")
            return(None)

        X_center = col_scale(X, self.center, self.normalize)
        loss_all = []
        if(self.V_init == None): 
            V_cur = inverse_prob_method(X_center, self.K, center = self.center, normalize = self.normalize)
        else : 
            V_cur = self.V_init
        
        Omega_temp = get_omega(X_center)
        
        if(Omega == None): Omega = Omega_temp

        i = 0
        step_cur = 0
        while(i<self.max_iter): 
            i += 1
            rows = sample_filter_op(self.thresh_sigma, V_cur, Omega, prob = self.prob)
            Xs = select_rows(X_center, rows=rows)
            Os = select_rows(Omega, rows=rows)
            svd_new_V = eigen_refine(V_cur, Xs, Os, thresh = self.thresh_als)
            V_new = svd_new_V
            difference = sin_theta_distance(V_new, V_cur)
            loss_cur = mse_eval(V_new, Xs, Os)
            loss_all.append(loss_cur)

            if(self.trace):
                print("Step ", i, "\n", (Xs.shape)[0], " rows selected\n", 
                    "MSE: ", loss_cur, "\n")
                print("F-norm sine theta difference: ", difference, 
                    "\n")

            V_cur = V_new
            step_cur = i
            if (difference < self.thresh_convergence):
                # print("Convergence threshold is hit.")
                break

            if (i >= self.max_iter):
                print("Max iteration number is hit.")
              
        self.v = V_cur
        self.step_cur = step_cur
        self.loss_all = loss_all

def main():
    print("--- No Noise Case ---")
    H_One_missingness()


if __name__ == '__main__':
    main()
