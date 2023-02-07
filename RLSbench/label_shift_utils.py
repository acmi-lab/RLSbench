import math
import sys
from ctypes import c_short

import cvxpy as cp
import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger("label_shift")

def get_dirichlet_marginal(alpha, seed): 
    
    np.random.seed(seed)

    return np.random.dirichlet(alpha)

def get_resampled_indices(y, num_labels, Py, seed):

    np.random.seed(seed)
    # get indices for each label
    indices_by_label = [(y==k).nonzero()[0] for k in range(num_labels)]
    num_samples = int(min([len(indices_by_label[i])/Py[i] for i in range(num_labels)]))

    agg_idx = []        
    for i in range(num_labels):
        # sample an example from X with replacement
        idx = np.random.choice(indices_by_label[i], size = int(num_samples* Py[i]), replace = False)
        agg_idx.append(idx)

    return np.concatenate(agg_idx)

def tweak_dist_idx(y, num_labels, n, Py, seed):

    np.random.seed(seed)
    # get indices for each label
    indices_by_label = [(y==k).nonzero()[0] for k in range(num_labels)]
    
    labels = np.argmax(
        np.random.multinomial(1, Py, n), axis=1)

    agg_idx = []        
    for i in range(n):
        # sample an example from X with replacement
        idx = np.random.choice(indices_by_label[labels[i]])
        agg_idx.append(idx)

    return agg_idx

def compute_w_opt(C_yy,mu_y,mu_train_y, rho):
    n = C_yy.shape[0]
    theta = cp.Variable(n)
    b = mu_y - mu_train_y
    objective = cp.Minimize(cp.pnorm(C_yy @ theta - b) + rho* cp.pnorm(theta))
    constraints = [-1 <= theta]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    # print(theta.value)
    w = 1 + theta.value
#     print('Estimated w is', w)
    #print(constraints[0].dual_value)
    return w

def im_weights_update(source_y, target_y, cov, im_weights=None, ma = 0.5):
    """
    Solve a Quadratic Program to compute the optimal importance weight under the generalized label shift assumption.
    :param source_y:    The marginal label distribution of the source domain.
    :param target_y:    The marginal pseudo-label distribution of the target domain from the current classifier.
    :param cov:         The covariance matrix of predicted-label and true label of the source domain.
    :return:
    """
    # Convert all the vectors to column vectors.
    dim = cov.shape[0]
    source_y = source_y.reshape(-1, 1).astype(np.double)
    target_y = target_y.reshape(-1, 1).astype(np.double)
    cov = cov.astype(np.double)

    P = matrix(np.dot(cov.T, cov), tc="d")
    q = -matrix(np.dot(cov.T, target_y), tc="d")
    G = matrix(-np.eye(dim), tc="d")
    h = matrix(np.zeros(dim), tc="d")
    A = matrix(source_y.reshape(1, -1), tc="d")
    b = matrix([1.0], tc="d")
    sol = solvers.qp(P, q, G, h, A, b)
    new_im_weights = np.array(sol["x"])
    
    # import pdb; pdb.set_trace()

    # EMA for the weights
    im_weights = (1 - ma) * new_im_weights + ma * im_weights

    return im_weights

def compute_3deltaC(n_class, n_train, delta):
    rho = 3*(2*np.log(2*n_class/delta)/(3*n_train) + np.sqrt(2*np.log(2*n_class/delta)/n_train))
    return rho


def EM(p_base,soft_probs, nclass): 
#   Initialization
    q_prior = np.ones(nclass)
    q_posterior = np.copy(soft_probs)
#     print (Q_func(q_posterior,q_prior))
    curr_q_prior = np.average(soft_probs,axis=0)
#     print q_prior
#     print curr_q_prior
    iter = 0
    while abs(np.sum(abs(q_prior - curr_q_prior))) >= 1e-6 and iter < 10000:
#         print iter
        q_prior = np.copy( curr_q_prior)
#         print curr_q_prior
#         print np.divide(curr_q_prior, p_base)
        temp = np.multiply(np.divide(curr_q_prior, p_base), soft_probs)
#         print temp
        q_posterior = np.divide(temp, np.expand_dims(np.sum(temp,1),axis=1))
#         print q_posterior
        curr_q_prior = np.average(q_posterior, axis = 0)
#         print curr_q_prior
        iter +=1 
#     print q_prior
#     print curr_q_prior
#     print (Q_func(q_posterior,curr_q_prior))
#     print iter
    return curr_q_prior 


def get_fisher(py_x, py, w):
    
    dims = py.shape[0] -1 
    temp = np.divide(py_x,py)
#     print temp[:,-1].shape
#     print (temp[:,:dims] - temp[:,-1]).shape
    score = np.divide(temp[:,:dims] - np.expand_dims(temp[:,-1], axis=1),np.expand_dims(np.matmul(py_x,w),axis=1))
#     print score.shape
    fisher = np.matmul(score.T,score)
    
    return fisher

def idx2onehot(a,k):
    a=a.astype(int)
    b = np.zeros((a.size, k))
    b[np.arange(a.size), a] = 1
    return b

def confusion_matrix(ytrue, ypred,k):
    # C[i,j] denotes the frequency of ypred = i, ytrue = j.
    n = ytrue.size
    C = np.dot(idx2onehot(ypred,k).T,idx2onehot(ytrue,k))
    return C/n

def confusion_matrix_probabilistic(ytrue, ypred,k):
    # Input is probabilistic classifiers in forms of n by k matrices
    n,d = np.shape(ypred)
    C = np.dot(ypred.T, idx2onehot(ytrue,k))
    return C/n

def calculate_marginal(y,k):
    mu = np.zeros(shape=(k))
    for i in range(k):
        mu[i] = np.count_nonzero(y == i)
    return mu/len(y)

def calculate_marginal_probabilistic(y,k):
    return np.mean(y,axis=0)

def estimate_labelshift_ratio(ytrue_s, ypred_s, ypred_t,k):
    if ypred_s.ndim == 2: # this indicates that it is probabilistic
        C = confusion_matrix_probabilistic(ytrue_s,ypred_s,k)
        mu_t = calculate_marginal_probabilistic(ypred_t, k)
    else:
        C = confusion_matrix(ytrue_s, ypred_s,k)
        mu_t = calculate_marginal(ypred_t, k)
    lamb = 1e-8
    wt = np.linalg.solve(np.dot(C.T, C)+lamb*np.eye(k), np.dot(C.T, mu_t))
    return wt

def estimate_labelshift_ratio_direct(ytrue_s, ypred_s, ypred_t,k):
    if ypred_s.ndim == 2: # this indicates that it is probabilistic
        C = confusion_matrix_probabilistic(ytrue_s,ypred_s,k)
        mu_t = calculate_marginal_probabilistic(ypred_t, k)
    else:
        C = confusion_matrix(ytrue_s, ypred_s,k)
        mu_t = calculate_marginal(ypred_t, k)
    # lamb = (1/min(len(ypred_s),len(ypred_t)))
    wt = np.linalg.solve(C,mu_t)
    return wt

def estimate_target_dist(wt, ytrue_s,k):
    ''' Input:
    - wt:    This is the output of estimate_labelshift_ratio)
    - ytrue_s:      This is the list of true labels from validation set

    Output:
    - An estimation of the true marginal distribution of the target set.
    '''
    mu_t = calculate_marginal(ytrue_s,k)
    return wt*mu_t

# functions that convert beta to w and converge w to a corresponding weight function.
def beta_to_w(beta, y, k):
    w = []
    for i in range(k):
        w.append(np.mean(beta[y.astype(int) == i]))
    w = np.array(w)
    return w

# a function that converts w to beta.
def w_to_beta(w,y):
    return w[y.astype(int)]

def w_to_weightfunc(w):
    return lambda x, y: w[y.astype(int)]


def BBSE(ypred_source, ytrue_source, ypred_target, numClasses): 

    py_true_source = calculate_marginal(ytrue_source, numClasses)

    ypred_hard_source = np.argmax(ypred_source, 1)
    ypred_hard_target = np.argmax(ypred_target, axis=1)

    wt_hard = estimate_labelshift_ratio(ytrue_source, ypred_hard_source, ypred_hard_target, numClasses).reshape((numClasses))
    wt_hard[wt_hard <0] = 0.0

    # C_soft = confusion_matrix_probabilistic(ytrue_source,ypred_source,numClasses)
    # C_hard = confusion_matrix(ytrue_source,ypred_hard_source,numClasses)

    wt_soft = estimate_labelshift_ratio(ytrue_source, ypred_source, ypred_target, numClasses).reshape((numClasses))
    wt_soft[wt_soft <0] = 0.0


    return np.multiply(wt_soft, py_true_source), np.multiply(wt_hard, py_true_source)
    

def MLLS(ypred_source, ytrue_source, ypred_target, numClasses): 

    # ypred_hard_source = np.argmax(ypred_source, 1)

    # ypred_marginal =  calculate_marginal(ypred_hard_source,numClasses)
    ypred_marginal = np.average(ypred_source, axis=0)
    # logger.debug(f"{ypred_marginal}")
    # logger.debug(f"{ypred_target}") 

    py_target = EM(ypred_marginal, ypred_target, numClasses)

    return py_target


def RLLS(ypred_source, ytrue_source, ypred_target, numClasses):

    n_train = ypred_source.shape[0]
    rho =  0.01*compute_3deltaC(numClasses, n_train, 0.05)

    ypred_hard_source = np.argmax(ypred_source,axis=1)
    ypred_hard_target = np.argmax(ypred_target, axis=1)
    mu_train_y = calculate_marginal(ytrue_source,numClasses)

    C_hard = confusion_matrix(ytrue_source,ypred_hard_source,numClasses)
    mu_y = np.average(idx2onehot(ypred_hard_target, numClasses), axis=0)

    wt_hard = compute_w_opt(C_hard, mu_y,mu_train_y, rho)

    C_soft = confusion_matrix_probabilistic(ytrue_source,ypred_source,numClasses)

    mu_y = np.average(ypred_target, axis=0)

    wt_soft = compute_w_opt(C_soft, mu_y,mu_train_y, rho)

    return np.multiply(wt_soft, mu_train_y), np.multiply(wt_hard, mu_train_y)

def gan_loss(epoch, disc_net, gen_params, opt_d, opt_g, valloader, val_labels, testloader, val_target_dist, device):
    disc_net.train()
#     gen_params.train()

    val_inputs = torch.tensor(valloader).float().to(device)
    test_inputs = torch.tensor(testloader).float().to(device)

    real_labels = torch.ones(valloader.shape[0]).long().to(device)
    fake_labels = torch.zeros(testloader.shape[0]).long().to(device)

    
    
    im_weights =  torch.divide(F.softmax(gen_params, dim = -1), torch.tensor(val_target_dist).float().to(device))
#     print(im_weights)
#     print(val_inputs[:10])
    
#     val_inputs = torch.mul(val_inputs, im_weights) 
#     print(torch.sum(test_inputs,dim=-1).shape)
#     val_inputs = val_inputs/ torch.sum(val_inputs,dim=-1, keepdim=True)
#     print(val_inputs[:10])
    
    for i in range(1):
        opt_d.zero_grad()

        inputs = torch.cat((val_inputs.detach(), test_inputs), 0)
        labels = torch.cat((real_labels, fake_labels), 0)
        weights = torch.cat([im_weights[val_labels].detach(), torch.ones_like(fake_labels)],0)
        xx = disc_net(inputs)
        d_loss = torch.mean(nn.CrossEntropyLoss(reduction="none")(xx, labels)*weights)
        
#         if epoch %50 ==0 : 
#             print(xx)
        d_loss.backward()
        opt_d.step()
    
    opt_g.zero_grad()
    
#     print(type(real_labels))
#     print(type(test_inputs))
#     print(test_inputs.shape)
    
    g_loss = -1.0*torch.mean(nn.CrossEntropyLoss(reduction="none")(disc_net(val_inputs), real_labels)*im_weights[val_labels])

    g_loss.backward()
    opt_g.step()

    return g_loss.item(), d_loss.item()


def gan_target_marginal(epochs, ypred_source, ytrue_val, ypred_target, numClasses, device="cpu"): 

    py_true_source = calculate_marginal(ytrue_val, numClasses).reshape((numClasses))
    d_net = nn.Sequential(nn.Linear(numClasses, 2, bias=False) ) 
    d_net = d_net.to(device)
    opt_d = optim.SGD(d_net.parameters(), lr=.1, momentum= 0.0, weight_decay=0.000)

    g_params = torch.Tensor([ 0.0]* numClasses).requires_grad_()
    opt_g = optim.SGD([g_params], lr=.10, momentum= 0.0, weight_decay=0.000)

    for epoch in range(epochs):
        
        g_loss, d_loss = gan_loss(epoch, d_net, g_params, opt_d, opt_g, ypred_source, ytrue_val, ypred_target, py_true_source, device)
        
        # if epoch % 50 == 0: 
            # print(f"Epoch: {epoch:.2f} G Loss: {g_loss:.5f}, D Loss: {d_loss:.5f}")
            # print (F.softmax(g_params, dim = -1))
            
    return F.softmax(g_params, dim = -1).detach().numpy()


#----------------------------------------------------------------------------

def im_reweight_acc(im_weights, probs, targets): 

    new_probs = im_weights[None]*probs

    preds = np.argmax(new_probs, axis=-1)

    return np.mean(preds == targets)*100


def get_acc(probs, labels): 
    preds = np.argmax(probs, axis=-1)
    return np.mean(preds == labels)*100
