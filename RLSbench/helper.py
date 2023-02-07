import logging
import os
import copy 
import torch
import torch.nn.functional as F
from calibration import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from RLSbench.transforms import initialize_transform

from RLSbench.label_shift_utils import *
from RLSbench.utils import (InfiniteDataIterator, collate_list,
                                     detach_and_clone, load, save_model, move_to)

from RLSbench.collate_functions import initialize_collate_function

logger = logging.getLogger("label_shift")

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def target_eval(model_preds, results, estimated_marginal, config, ytrue, cal=None):
    
    prefix = ""
    if cal is not None:
        prefix = "cal_"

        model_preds["source"] = cal.calibrate(model_preds["source"])
        model_preds["target"] = cal.calibrate(model_preds["target"])
    
    source_ydist = calculate_marginal(ytrue["source"], config.num_classes)
    target_ydist = calculate_marginal(ytrue["target"], config.num_classes)

    source_model_probs  = softmax(model_preds["source"], axis=-1)
    target_model_probs  = softmax(model_preds["target"], axis=-1)

    pred_ydist = calculate_marginal_probabilistic(target_model_probs, config.num_classes)
    try: 
        MLLS_ydist = MLLS(source_model_probs, ytrue["source"], target_model_probs, config.num_classes)
    except:
        MLLS_ydist = pred_ydist

    try:
        RLLS_ydist, _ = RLLS(source_model_probs, ytrue["source"],target_model_probs, config.num_classes)
    except:
        RLLS_ydist = pred_ydist
        # import pdb; pdb.set_trace()

    estimated_marginal[f"{prefix}source"] = source_ydist
    estimated_marginal[f"{prefix}baseline"] = pred_ydist 
    estimated_marginal[f"{prefix}MLLS"] = MLLS_ydist
    estimated_marginal[f"{prefix}RLLS"] = RLLS_ydist
    estimated_marginal[f"{prefix}true"] = target_ydist

    logger.debug(f"Baseline prediction {pred_ydist}")
    logger.debug(f"MLLS prediction {MLLS_ydist}")
    logger.debug(f"RLLS prediction {RLLS_ydist}")
    logger.debug(f"True prediction {target_ydist}")
    
    # if config.source_balanced: 
    #     # import pdb; pdb.set_trace()


    # else: 
    results[f"{prefix}target_acc_no_im"] = get_acc(target_model_probs, ytrue["target"])
    results[f"{prefix}target_acc_oracle"] = im_reweight_acc( target_ydist/ source_ydist,  target_model_probs, ytrue["target"])
    results[f"{prefix}target_acc_baseline"] = im_reweight_acc( pred_ydist/ source_ydist,  target_model_probs, ytrue["target"])
    results[f"{prefix}target_acc_MLLS"] = im_reweight_acc( MLLS_ydist/ source_ydist,  target_model_probs, ytrue["target"])
    results[f"{prefix}target_acc_RLLS"] = im_reweight_acc( RLLS_ydist/ source_ydist,  target_model_probs, ytrue["target"])

    results[f"{prefix}baseline_AE"] = np.sum(np.abs( pred_ydist - target_ydist))
    results[f"{prefix}MLLS_AE"] = np.sum(np.abs( MLLS_ydist - target_ydist))
    results[f"{prefix}RLLS_AE"] = np.sum(np.abs( RLLS_ydist - target_ydist))

    prefix = "mul"
    results[f"{prefix}target_acc_oracle"] = im_reweight_acc( target_ydist,  target_model_probs, ytrue["target"])
    results[f"{prefix}target_acc_baseline"] = im_reweight_acc( pred_ydist,  target_model_probs, ytrue["target"])
    results[f"{prefix}target_acc_MLLS"] = im_reweight_acc( MLLS_ydist,  target_model_probs, ytrue["target"])
    results[f"{prefix}target_acc_RLLS"] = im_reweight_acc( RLLS_ydist,  target_model_probs, ytrue["target"])

    return results, estimated_marginal


def initialize_marginal(dataloaders, config):
    estimated_marginal = {}
    ytrue = {}

    logger.info("Initializing marginal distribution...")
    for dataset in dataloaders:
        if dataset.endswith('test'):
            
            dataset_name = dataset.split('_')[0]

            epoch_y_true = []
            iterator = tqdm(dataloaders[dataset]) if config.progress_bar else dataloaders[dataset]

            for batch in iterator:
                epoch_y_true.append(detach_and_clone(batch[1]))

            epoch_y_true = collate_list(epoch_y_true).cpu().numpy()

            ytrue[dataset_name] = epoch_y_true
    

    source_ydist = calculate_marginal(ytrue["source"], config.num_classes)
    target_ydist = calculate_marginal(ytrue["target"], config.num_classes)
    
    estimated_marginal[f"source"] = source_ydist
    estimated_marginal[f"baseline"] = source_ydist 
    estimated_marginal[f"MLLS"] = source_ydist
    estimated_marginal[f"RLLS"] = source_ydist
    estimated_marginal[f"true"] = target_ydist

    # import pdb; pdb.set_trace()
    return estimated_marginal

def run_epoch(algorithm, dataloader, config, train, unlabeled_dataloader=None, estimated_marginal = None):

    if train:
        algorithm.train()
        torch.set_grad_enabled(True)
    else:
        algorithm.eval()
        torch.set_grad_enabled(False)

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    iterator = dataloader

    if config.progress_bar:
        iterator = tqdm(iterator)

    last_batch_idx = len(iterator)-1
    
    if unlabeled_dataloader:
        unlabeled_data_iterator = InfiniteDataIterator(unlabeled_dataloader)

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    train_y_pred = []
    for labeled_batch in iterator:
        if unlabeled_dataloader:
            unlabeled_batch = next(unlabeled_data_iterator)

            results = algorithm.update(labeled_batch, unlabeled_batch, \
                target_marginal = estimated_marginal[f"{config.estimation_method}"], \
                source_marginal = estimated_marginal["source"], \
                target_average= estimated_marginal["target_average"], \
                is_epoch_end=(batch_idx==last_batch_idx))

            if config.algorithm in ('SENTRY', 'IW-FixMatch', 'IW-PseudoLabel'): 
                train_y_pred.append(results['y_weak_pseudo_label'])

        else:
            algorithm.update(labeled_batch, is_epoch_end=(batch_idx==last_batch_idx))

        batch_idx += 1

    if len(train_y_pred) > 0:
        train_y_pred = collate_list(train_y_pred).detach().cpu().numpy()
        # train_y_pred = np.argmax(train_y_pred, axis=1)
        target_average = calculate_marginal(train_y_pred, config.num_classes)

        return target_average

    else: 
        return None


def train(algorithm, dataloaders, results_logger, config, epoch_offset, datasets = None):
    """
    Train loop that, each epoch:
        - Steps an algorithm on the datasets['train'] split and the unlabeled split
        - Evaluates the algorithm on the datasets['val'] split
        - Saves models / preds with frequency according to the configs
        - Evaluates on any other specified splits in the configs
    Assumes that the datasets dict contains labeled data.
    """
    logger.info("Training model ...")

    # _, estimated_marginal = evaluate(algorithm, dataloaders, 0, results_logger, config)
    estimated_marginal = {}
    if config.use_target:
        estimated_marginal = initialize_marginal(dataloaders, config)

    save_model_if_needed(algorithm, 0, config)

    if "SENTRY" in config.algorithm or "IS-FixMatch" in config.algorithm  or "IS-CDANN" in config.algorithm or "IS-ERM" in config.algorithm \
        or "IS-DANN" in config.algorithm or "IS-PseudoLabel" in config.algorithm:  
        # import pdb; pdb.set_trace() 
        dataloaders['source_train'] =  rebalance_loader(datasets["source_train"] , config, model = None, use_true_target = True) 

    estimated_marginal["target_average"]  = None
    for epoch in range(epoch_offset, config.n_epochs):
        logger.info(f'\nEpoch {epoch+1}:\n')
        
        # import pdb; pdb.set_trace()
        # First run training
        if "target_train" in dataloaders:
            target_average = run_epoch(algorithm, dataloaders['source_train'], config, train=True, unlabeled_dataloader=dataloaders['target_train'], estimated_marginal = estimated_marginal)
        else: 
            target_average = run_epoch(algorithm, dataloaders['source_train'], config, train=True, estimated_marginal = None)

        if (epoch+1) % config.evaluate_every == 0:
            _, estimated_marginal = evaluate(algorithm, dataloaders, epoch+1, results_logger, config) 

        estimated_marginal["target_average"] = target_average

        save_model_if_needed(algorithm, epoch+1, config)

        if "SENTRY" in config.algorithm or "IS-FixMatch" in config.algorithm or "IS-CDANN" in config.algorithm \
            or "IS-DANN" in config.algorithm or "IS-PseudoLabel" in config.algorithm:  
            dataloaders['target_train'] =  rebalance_loader(datasets["target_train"] , config, model = algorithm.model, use_true_target = False)
            
        logger.info(f"Epoch {epoch+1} done.")

def evaluate(algorithm, dataloaders, epoch, results_logger, config, log=True):
    algorithm.eval()
    torch.set_grad_enabled(False)

    logger.info(f"Evaluating for epoch {epoch}..." )
    
    results = {}
    model_preds = {}
    ytrue = {}
    estimated_marginal = {}

    for dataset in dataloaders:
        if dataset.endswith('test'):
            
            dataset_name = dataset.split('_')[0]
            logger.info(f"Evaluating on {dataset_name}...")

            epoch_y_true = []
            epoch_y_preds = []

            iterator = tqdm(dataloaders[dataset]) if config.progress_bar else dataloaders[dataset]

            for batch in iterator:

                batch_results = algorithm.evaluate(batch)
                epoch_y_true.append(detach_and_clone(batch_results['y_true']))
                y_preds = detach_and_clone(batch_results['y_pred'])

                epoch_y_preds.append(y_preds)

            epoch_y_preds = collate_list(epoch_y_preds).cpu().numpy()
            epoch_y_true = collate_list(epoch_y_true).cpu().numpy()

            ytrue[dataset_name] = epoch_y_true
            model_preds[dataset_name] = epoch_y_preds
    
    results['epoch'] = epoch
    results["source_acc"] = get_acc(model_preds["source"], ytrue["source"])

    # import pdb; pdb.set_trace()
    if config.use_target:
        assert "source" in ytrue.keys() and "target" in ytrue.keys(), "ytrue must contain source and target"

        results, estimated_marginal = target_eval(model_preds, results, estimated_marginal, config, ytrue)

        # logger.info("Calibrating the model on source ... ")

        # import pdb; pdb.set_trace()

        # cal = VectorScaling(config.num_classes, bias=True, device=config.device)
        # cal.fit(model_preds["source"], ytrue["source"])

        # logger.info("Calibration done ....")

        # results, estimated_marginal = target_eval(model_preds, results, estimated_marginal, config, ytrue, cal)

    # results["cal_source_acc"] = get_acc(model_preds["source"], ytrue["source"])


    if log:
        results_logger.log(results)

    logger.info(f"Evaluation results for epoch {epoch}:\n{results}")
    
    logger.info("Evaluation complete.")

    return results, estimated_marginal


def adapt(algorithm, dataloaders, results_logger, config, datasets = None):

    if "BN_adapt" in config.algorithm or "TENT" in config.algorithm: 
        source_model_path = config.source_model_path

        model_paths = []
        for root, dirs, files in os.walk(source_model_path):

            for file in files:
                if file.endswith(".pth"):
                    model_path = os.path.join(root, file)
                    model_paths.append(model_path)
        
        
        for model_path in model_paths:
            logger.info(f"Loading model from {model_path}")

            # import pdb; pdb.set_trace()
            
            epoch = load(algorithm, model_path, device = config.device)

            _, estimated_marginal = evaluate(algorithm, dataloaders, epoch, results_logger, config, log=False) 

            logger.info(f"Adapting model from {model_path} at epoch {epoch}")
        
            if "IS-BN_adapt" in config.algorithm or "IS-TENT" in config.algorithm: 
                dataloaders['target_train'] =  rebalance_loader(datasets["target_train"] , config, model = algorithm.model, use_true_target = False)
        
            algorithm.train()
            torch.set_grad_enabled(True)

            algorithm.adapt(source_loader=dataloaders["source_train"], 
                    target_loader=dataloaders["target_train"],
                    target_marginal = estimated_marginal[f"{config.estimation_method}"],
                    source_marginal = estimated_marginal["source"]) 
            
            logger.info(f"Adapting complete. Evaluating on test set...")

            # if (epoch) % config.evaluate_every == 0:
            _, estimated_marginal = evaluate(algorithm, dataloaders, epoch, results_logger, config) 

            algorithm.reset()
            
            logger.info("Epoch %d done." % epoch)

    elif "CORAL" in config.algorithm: 
        # For best model evaluation
        
        source_model_path = config.source_model_path

        epoch = None 
        for root, dirs, files in os.walk(source_model_path):

            for file in files:
                if file.endswith("eval.csv"):
                    import pandas as pd 

                    df = pd.read_csv(os.path.join(root, file))
                    epoch = int(df [df['source_acc'] == df['source_acc'].max()]["epoch"].min())
                    # model_path = os.path.join(root, file)
                    # model_paths.append(model_path)
        
        model_path = None 
        for root, dirs, files in os.walk(source_model_path):

            for file in files:
                if file.endswith(f"epoch:{epoch}_model.pth"):
                    model_path = os.path.join(root, file)

        if model_path is None:
            raise ValueError("No model found")


        logger.info(f"Loading model from {model_path}")

        # import pdb; pdb.set_trace()
        
        epoch = load(algorithm, model_path, device = config.device)

        # _, estimated_marginal = evaluate(algorithm, dataloaders, epoch, results_logger, config, log=False) 

        # logger.info(f"Adapting model from {model_path} at epoch {epoch}")
        if "IS-CORAL" in config.algorithm: 
            logger.info("Getting balanced and pseudo balanced dataloaders... ")
            dataloaders['source_train'] =  rebalance_loader(datasets["source_train"] , config, model = None, use_true_target = True) 
            dataloaders['target_train'] =  rebalance_loader(datasets["target_train"] , config, model = algorithm.model, use_true_target = False)

        for epoch in range(3):
            algorithm.train()
            torch.set_grad_enabled(True)
            
            algorithm.adapt(source_loader=dataloaders["source_train"], 
                    target_loader=dataloaders["target_train"])
                    # target_marginal = estimated_marginal[f"{config.estimation_method}"],
                    # source_marginal = estimated_marginal["source"]) 
            
            logger.info(f"Adapting complete. Evaluating on test set...")

            # if (epoch) % config.evaluate_every == 0:
            _, estimated_marginal = evaluate(algorithm, dataloaders, epoch, results_logger, config) 

        algorithm.reset()
        
        logger.info("Epoch %d done." % epoch)


def eval_models(algorithm, dataloaders, results_logger, config):

    if "adv" not in config.algorithm:
        logger.info("Evaluating all models ... ")
        source_model_path = config.source_model_path

        model_paths = []
        for root, dirs, files in os.walk(source_model_path):

            for file in files:
                if file.endswith(".pth"):
                    model_path = os.path.join(root, file)
                    model_paths.append(model_path)
        
        for model_path in model_paths:
            logger.info(f"Loading model from {model_path}")

            epoch = load(algorithm, model_path, device = config.device)

            logger.info(f"Evaluating model at epoch {epoch}...")

            _, estimated_marginal = evaluate(algorithm, dataloaders, epoch, results_logger, config) 

            logger.info("Epoch %d done." % epoch)

    else: 
        logger.info("Evaluating best model ... ")
        # For best model evaluation
        source_model_path = config.source_model_path

        # model_paths = []
        epoch = None 
        for root, dirs, files in os.walk(source_model_path):

            for file in files:
                if file.endswith("eval.csv"):
                    import pandas as pd 

                    df = pd.read_csv(os.path.join(root, file))
                    epoch = int(df [df['source_acc'] == df['source_acc'].max()]["epoch"].min())
                    # model_path = os.path.join(root, file)
                    # model_paths.append(model_path)
        
        model_path = None 
        for root, dirs, files in os.walk(source_model_path):

            for file in files:
                if file.endswith(f"epoch:{epoch}_model.pth"):
                    model_path = os.path.join(root, file)
        
        
        if model_path is None:
            raise ValueError("No model found")

        logger.info(f"Loading model from {model_path}")

        epoch = load(algorithm, model_path, device = config.device)

        logger.info(f"Evaluating model at epoch {epoch}...")

        _, estimated_marginal = evaluate(algorithm, dataloaders, epoch, results_logger, config) 

        logger.info("Epoch %d done." % epoch)

def infer_predictions(model, loader, config):
    """
    Simple inference loop that performs inference using a model (not algorithm) and returns model outputs.
    Compatible with both labeled and unlabeled WILDS datasets.
    """
    model.eval()
    y_pred = []
    iterator = tqdm(loader) if config.progress_bar else loader
    
    with torch.no_grad(): 
        for batch in iterator:
            x = batch[0]
            x = move_to(x, config.device)
            # x = x.to(config.device)
            output = model(x)
            pseudo_labels = output.argmax(dim=-1)
            y_pred.append(detach_and_clone(pseudo_labels).cpu().numpy())

    return np.concatenate(y_pred)

def save_model_if_needed(algorithm, epoch, config):

    if config.save_every is not None and (epoch) % config.save_every == 0:
        save_model(algorithm, epoch, f'{config.log_dir}/epoch:{epoch}_model.pth')

    if config.save_last:
        save_model(algorithm, epoch, f'{config.log_dir}/epoch:last_model.pth')



def rebalance_loader(dataset, config, model = None, use_true_target = False): 

    collate_function = initialize_collate_function(config.collate_function)
    
    # import pdb; pdb.set_trace()

    if not use_true_target: 
        assert model is not None, "If use_true_target is False, model must be provided"

    if use_true_target:
        target = np.array(dataset.y_array)
    
    else:

        # if "bert" in config.transform:
        #     weak_data_transforms = initialize_transform(
        #         transform_name=config.transform,
        #         config=config,
        #         additional_transform_name=None,
        #         is_training=True, 
        #         model_name=config.model)

        # elif config.transform.lower() == 'none' or config.transform == None:
        #      weak_data_transforms = initialize_transform(
        #         transform_name=config.transform,
        #         config=config,
        #         additional_transform_name=None,
        #         is_training=True, 
        #         model_name=config.model)

        # else:
        weak_data_transforms = initialize_transform(
            transform_name=config.transform,
            config=config,
            additional_transform_name="weak",
            is_training=True, 
            model_name=config.model)

        dataset_copy = copy.deepcopy(dataset)

        if config.transform is not None and config.transform.lower() != 'none':
            if dataset.transform is not None: 
                dataset_copy.transform = weak_data_transforms
            else: 
                dataset_copy.dataset.transform = weak_data_transforms  
            
        
        dataloader =  DataLoader(
            dataset_copy,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers, 
            pin_memory=True, 
            collate_fn=collate_function
        )
        target = infer_predictions(model, dataloader, config)

    # import pdb; pdb.set_trace()
    class_counts = np.bincount(target)
    max_count = max(class_counts)
    class_counts_proxy = class_counts +  1e-8
    class_weights = max_count / class_counts_proxy

    class_weights[class_counts == 0] = 0

    weights = class_weights[target]
    sampler = WeightedRandomSampler(weights, len(weights))

    loader = DataLoader(dataset, 
        batch_size = config.batch_size, 
        sampler = sampler, 
        num_workers = config.num_workers, 
        pin_memory = True, 
        collate_fn=collate_function
    )

    return loader
