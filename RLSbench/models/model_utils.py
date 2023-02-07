import logging
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("label_shift")


def PCA_whitener(centered):
    n, p = centered.shape
    cov = torch.matmul(centered.T, centered) / (n - 1)
    # torch.symeig much less stable since the small eigenvalues are very close together, often returns negatives.
    U, S, _ = torch.svd(cov)

    rho = .01
    return U, (1. - rho) * S + rho * S.mean()

def linear_probe(model, dataloader, device, lambda_param=1e-6, progress_bar=True): 

    logger.info("Linear probing ... ")
    
    model[0].to(device)
    model[0].eval()
    data_features = []
    data_labels = []
    iterator = dataloader
    if progress_bar: 
        iterator = tqdm(iterator)
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        for batch in iterator:
            # import pdb; pdb.set_trace()
            
            x, y = batch[:2]
            x = x.to(device)
            y = y.to(device)
            features= model[0](x)
            data_features.append(features.cpu().numpy())
            data_labels.append(y.cpu().numpy())


    data_features = torch.tensor(np.concatenate(data_features, axis=0))
    data_labels = torch.tensor(np.concatenate(data_labels, axis=0))

    optimizer = torch.optim.LBFGS(model[1].parameters(), history_size=100, max_iter=100, lr=0.1)

    model[0].to(torch.device("cpu"))

    model[1].to(device)
    model[1].train()

    new_loss = -200.0 
    loss = -100.0

    iteration = 0
    logger.info("Got features, now training the linear layer ...")
    while (np.abs(new_loss - loss) > 1e-6):
        
        logger.info(f"Linear probing iteration {iteration+1} ... ")
        loss = new_loss
        data_features, data_labels = data_features.to(device), data_labels.to(device)

        def closure_fn():
            out = model[1](data_features)

            l2_norm = sum(p.pow(2.0).sum() for p in model[1].parameters())
            loss = F.cross_entropy(out, data_labels) + l2_norm*lambda_param

            optimizer.zero_grad()
            loss.backward()
            
            return loss

        optimizer.step(closure_fn)
        iteration = iteration + 1
        with torch.no_grad():
            new_loss = F.cross_entropy(model[1](data_features), data_labels).cpu().numpy()

    model[1].to(torch.device("cpu"))

    return model

def train_CORAL(model, dataloader, im_weights, device, lambda_param=1e-6):

    logger.info("Getting features ... ")
    
    model.featurizer.eval()
    data_features = []
    data_labels = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[:2]
            x = x.to(device)
            y = y.to(device)
            features= model.featurizer(x)
            data_features.append(features.cpu().numpy())
            data_labels.append(y.cpu().numpy())
    

    data_features = torch.tensor(np.concatenate(data_features, axis=0))
    data_labels = torch.tensor(np.concatenate(data_labels, axis=0))

    # centered = data_features - torch.mean(data_features, 0, True)

    U, S = PCA_whitener(data_features)
    W = U @ torch.diag_embed(torch.reciprocal(torch.sqrt(S))) @ U.T       
    # features = torch.mm(data_features, W)
    W = W.to(device)

    logger.info("Linear probing ... ")

    # optimizer = torch.optim.LBFGS(model.classifier.parameters(), history_size=100, max_iter=100, lr=0.1)
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    model.classifier.train()

    for batch in dataloader:
        x, y = batch[:2]
        data_features, data_labels = x.to(device), y.to(device)

        with torch.no_grad():
            data_features = model.featurizer(data_features)
            data_features = torch.mm(data_features, W)
        
        out = model.classifier(data_features)

        loss = F.cross_entropy(out, data_labels)

        optimizer.zero_grad()            
        loss.backward()
            
        optimizer.step()

    return model    

def test_CORAL_params(model, dataloader, device): 

    logger.info("Getting mean and variance of target ... ")
    
    model.featurizer.eval()
    data_features = []
    data_labels = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[:2]
            x = x.to(device)
            # y = y.to(device)
            features= model.featurizer(x)
            data_features.append(features.cpu().numpy())
            # data_labels.append(y.cpu().numpy())
    

    data_features = torch.tensor(np.concatenate(data_features, axis=0))
    # data_labels = torch.tensor(np.concatenate(data_labels, axis=0))

    # mean = torch.mean(data_features, 0, True)
    # centered = data_features - mean 

    U, S = PCA_whitener(data_features)
    cov_inv = U @ torch.diag_embed(torch.reciprocal(torch.sqrt(S))) @ U.T       

    logger.info("Done.")

    return cov_inv 

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    
    # TODO: Check what if we combine this with BN adapt
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            # m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
    return model

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names
