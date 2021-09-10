import numpy as np
from numpy import linalg as LA
from random import gauss

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return np.array([x/mag for x in vec])
def compute_Psi(z):
    
    psi = 2*np.log(np.exp(z)+1) - z
    return psi 
def accuracy_function(X,Y,theta,zeta,strat_features):
    hi_theta =  evaluate_hi(X, Y, theta, strat_features = strat_features, zeta = zeta)
    acc = (np.multiply(Y,hi_theta) >= 0)
    return acc.mean()
def gradient_Psi(z):
    # z_half = z/2
    # psi_grad_p1 = np.exp(z_half)-np.exp(-z_half)
    # psi_grad_p2 = np.exp(z_half)+np.exp(-z_half)
    # psi_grad = psi_grad_p1/psi_grad_p2
    psi_grad = (2*np.exp(z))/(1+np.exp(z)) - 1
    return psi_grad 

def evaluate_logreg_loss(X,y,theta,strat_features,zeta,reg_factor = 0):
    #Compute the  best response 
    hi_theta =  evaluate_hi(X, y, theta, strat_features = strat_features, zeta = zeta)
    
    loss_1 = (1.0 / X.shape[0]) * np.sum(compute_Psi(hi_theta))
    factor = y*hi_theta
    loss_2 = (1.0 / X.shape[0])* np.sum(factor)
    loss_reg = reg_factor*np.linalg.norm(theta)**2
    return loss_1-loss_2+loss_reg

def dro_lg_loss(X, y, theta, alpha=0.0, epsilon=0.1, kappa=2.0):
    loss_1 = alpha * epsilon
    loss_2 = (1.0 / X.shape[0]) * np.log(1 + np.exp(-1.0 * np.multiply(y, (X @ theta)))).sum()
    loss_3 = (1.0 / X.shape[0]) * np.maximum(np.multiply(y, (X @ theta)) - alpha * kappa, 0).sum()
    dro_loss = loss_1 + loss_2 + loss_3 
    return dro_loss

# def evaluate_loss(X,y,alpha,theta,gamma,epsilon,kappa):
#    loss_1 = alpha * epsilon
#    loss_2 = (1.0 / X.shape[0]) * np.log(1 + np.exp(X@theta)).sum()
#    factor = np.multiply(y,(X@theta))-alpha*kappa
#    loss_3 = (1.0 / X.shape[0])* (gamma*factor).sum() #   return loss_1+loss_2+loss_3

def evaluate_loss_PP(X,y,alpha,theta,gamma,delta,kappa,strat_features,zeta=0.001,reg_factor=0):
    n,d = X.shape
    hi_theta =  evaluate_hi(X, y, theta, strat_features = strat_features, zeta = zeta)
    loss_1 = alpha*(delta-kappa)
    Y_indicator = (1+y)/2
    gam_Y = gamma*Y_indicator
    one_vec = np.ones((n,1))
    one_Y = one_vec*(1-Y_indicator)
    #loss_2 = (1.0 / X.shape[0]) * np.log(1 + np.exp(hi_theta)).sum()
    loss_2 = (1.0 / n) * np.sum(compute_Psi(hi_theta))
    factor_gam = hi_theta-alpha*kappa
    factor_one = hi_theta 
    loss_gam = gam_Y*factor_gam
    loss_one = one_Y*factor_one
    loss_3 = (1.0 / n)* np.sum(loss_gam+loss_one)
    loss_reg = reg_factor*np.linalg.norm(theta)**2
    return loss_1+loss_2+loss_3 + loss_reg





        

    

    



def best_response(X,Y, theta, strat_features=[], noise=0, scale=0, zeta = 0.001):
    
    n,d = X.shape
    X_strat = np.copy(X)
    
    theta_vec = theta.reshape((1,d))
    theta_strat = theta_vec[0, strat_features]
    for j in range(n):    
        if Y[j] == -1:
            X_strat[j, strat_features] += zeta * theta_strat
    return X_strat

def evaluate_hi(X, Y, theta, strat_features = [], noise = 0, scale = 0, zeta= 0.001):
    # Evaluate hi function, which returns (best response features)^T theta.

    X_perf = best_response(X,Y, theta, strat_features=strat_features, noise=noise, scale=scale, zeta=zeta)
    return X_perf @ theta



def getTruegrad(X,Y,theta, gamma_, strat_features = [],zeta=0.001,d=10):
    # X, Y are specific to data set 
    #n, d = X.shape
    n = X.shape[0]
    hi_theta = evaluate_hi(X, Y, theta, strat_features=strat_features,zeta=zeta)
    Y_indicator  = (1+Y)/2
    gam_Y = Y_indicator*gamma_
    one_vec = np.ones((n,1))
    one_Y = (1-Y_indicator)*one_vec
    X_strat = np.copy(X)
    theta_vec = theta.reshape((1,d))
    
    inter_1 = theta_vec[0, strat_features].reshape((len(strat_features),))
    for j in range(n):
        if Y[j] == -1: 
            #print(X_strat[j, strat_features].shape)
            X_strat[j, strat_features] += 2*zeta * inter_1
            
    #inter_2 = np.exp(hi_theta)/(1 + np.exp(hi_theta))
    inter_2 = gradient_Psi(hi_theta)
    
    X_true_grad = (1/n)*X_strat.transpose() @ (inter_2 + gam_Y+one_Y)
    return X_true_grad


    

    

    
def OperatorSC(X_, y_,theta,strat_features = [],zeta=0.001,reg_factor = 0, dynamic=False, mixing=0, n_epoch=0, X_base=None):
    # Always input Y as swapped 
    n,d = X_.shape
    #Pseudo Code : 
        # Compute the gradient of logistic loss with respect to theta:
            # Important to note that if Y=
    ######## Note that I am putting negative y here because of labelling issues 
    hi_theta = evaluate_hi(X_, y_, theta, strat_features=strat_features,zeta=zeta)
    X_strat = np.copy(X_)
    theta_vec = theta.reshape((1,d))
    
    #if len(strat_features)==1:
    inter_1 = theta_vec[0,strat_features].reshape((len(strat_features),))
    #print(inter_1)
    
    #else:
    #    inter_1 = theta_vec[strat_features].reshape((len(strat_features),))

    if dynamic:
        for jj in range(n):
            if y_[jj] == -1:
                #print("neg label : ", jj)
                #X_strat[jj, strat_features] += 2*zeta * inter_1
                ##X_strat[jj, strat_features]= np.copy(1*X_strat[jj, strat_features])
                X_strat[jj, strat_features]=mixing**n_epoch*X_base[jj,strat_features]+(1-mixing**n_epoch)*X_strat[jj, strat_features]
                
                for ij,strat in enumerate(strat_features):
                    X_strat[jj, strat] += 2*zeta * inter_1[ij]
                    #X_strat[jj, strat_features]= np.copy(1*X_strat[jj, strat_features])
                    X_strat[jj, strat]=mixing**n_epoch*X_base[jj,strat]+(1-mixing**n_epoch)*X_strat[jj, strat]
    else:
        for j in range(n):
            if y_[j] == -1:
                X_strat[j, strat_features] += 2*zeta * inter_1
    
    #inter_2 = np.exp(hi_theta)/(1 + np.exp(hi_theta))
    inter_2 = gradient_Psi(hi_theta)
    F_theta_reg = 2*reg_factor*theta
    inter_3 = y_
    X_true_grad = (1/n)*X_strat.transpose() @ (inter_2 - inter_3) + F_theta_reg
    if dynamic:
        return X_true_grad, X_strat
    else:
        return X_true_grad
    
    

    
