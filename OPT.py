import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import argparse
import random
import copy 
from utils_ import *


def SC_solver(X_train, y_train,zeta=0.01, eta=0.001, EPOCH=100, strat_features = [],
              u_init=[],divide=1,factor_eta= 1,avg_factor=0.5,reg_factor = 0, dynamic=False, mixing=0, n_epoch=0, loc=1.0, scale=1.0, seed=0, 
              theta_gt=None):
    #  Using GD method to solve for PP problem of strategic classificaiton 
    #print('X_train shape: ', X_train.shape)
    #print('y_train shape: ', y_train.shape)
    #print('y values: ', np.unique(y_train))
    np.random.seed(seed)
    #print('====================================')
    eta_init = np.copy(eta)
    n,d = X_train.shape
    theta_lst  = []
    loss_lst = []
    X_train_dynamic_all=[]
    if len(u_init) == 0:
        theta_init = 0*np.ones((d,1))
    else: 
        theta_init = u_init[0]
    theta_lst.append(theta_init)
    
    loss_init = evaluate_logreg_loss(X_train,y_train,theta_init,strat_features,zeta,reg_factor=reg_factor)
    loss_lst.append(loss_init)    
    if dynamic:
        X_base=np.random.normal(loc=loc,scale=scale, size=(n,d)) #np.random.rand(np.shape(X_train)[0],np.shape(X_train)[1])
        X_train_dynamic=np.random.normal(loc=loc,scale=scale, size=(n,d)) #np.copy(X_train)
        X_train_dynamic=np.copy(X_train)
        inter_1 = theta_gt[0,strat_features].reshape((len(strat_features),))
        for jj in range(n):
            if y_train[jj] == -1:
                for ij,strat in enumerate(strat_features):
                    X_train_dynamic[jj, strat] -= 2*zeta * inter_1[ij]
                    #X_strat[jj, strat_features]= np.copy(1*X_strat[jj, strat_features])
                    #X_train_dynamic[jj, strat]=mixing**n_epoch*X_train_dynamic[jj,strat]+(1-mixing**n_epoch)*X_train_dynamic[jj, strat]
            else:
                for ij,strat in enumerate(strat_features):
                    X_train_dynamic[jj, strat] += 2*zeta * inter_1[ij]
                    #X_strat[jj, strat_features]= np.copy(1*X_strat[jj, strat_features])
                    #X_train_dynamic[jj, strat]=mixing**n_epoch*X_train_dynamic[jj,strat]+(1-mixing**n_epoch)*X_train_dynamic[jj, strat]  
        X_train_dynamic=np.random.normal(loc=loc,scale=scale, size=(n,d))
    u = np.copy(theta_init)
    loss_prev = np.copy(loss_init)
    for j in range(EPOCH):
        if dynamic:
            grad_theta, X_train_dynamic = OperatorSC(X_train, y_train,u,strat_features = strat_features,zeta=zeta,reg_factor=reg_factor, 
                                dynamic=dynamic, mixing=mixing, n_epoch=n_epoch, X_base=X_train_dynamic) #, X_train_dynamic
            X_train_dynamic_all.append(X_train_dynamic)
        else:
            grad_theta = OperatorSC(X_train, y_train,u,strat_features = strat_features,zeta=zeta,reg_factor=reg_factor, 
                                dynamic=dynamic)
        u = np.copy(u - eta*grad_theta)
        theta_lst.append(u)
        loss = evaluate_logreg_loss(X_train,y_train,u,strat_features,zeta,reg_factor=reg_factor)
        loss_lst.append(loss)
        #if j%100 == 1:
        #    print('PP Problem eta ', eta)
        #if j>0.5*EPOCH :
        #    eta = eta_init*0.5
        loss_prev = np.copy(loss)
    
    theta_final = np.copy(u)
    loss_final = np.copy(loss)
    if dynamic:
        return theta_final,loss_final,theta_lst,loss_lst,  X_train_dynamic_all
    else:
        return theta_final,loss_final,theta_lst,loss_lst



    
