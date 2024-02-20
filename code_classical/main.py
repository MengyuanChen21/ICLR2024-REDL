import os
import logging
import torch
from seml.utils import flatten
import wandb
import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from pathlib import Path
import itertools
import random
from pprint import pprint

from train import train
from dataset import get_dataset

from models.model_loader import load_model
from models.ModifiedEvidentialN import ModifiedEvidentialNet

from utils.io_utils import DataWriter
from utils.metrics import accuracy, confidence, anomaly_detection, our_confidence, our_anomaly_detection
from utils.metrics import compute_X_Y_alpha, name2abbrv

create_model = {'menet': ModifiedEvidentialNet}
logging.getLogger().setLevel(logging.INFO)


def main(config_dict):
    config_id = config_dict['config_id']
    suffix = config_dict['suffix']

    seeds = config_dict['seeds']

    dataset_name = config_dict['dataset_name']
    ood_dataset_names = config_dict['ood_dataset_names']
    split = config_dict['split']

    # Model parameters
    model_type = config_dict['model_type']
    name_model_list = config_dict['name_model']

    # Architecture parameters
    directory_model = config_dict['directory_model']
    architecture = config_dict['architecture']
    input_dims = config_dict['input_dims']
    output_dim = config_dict['output_dim']
    hidden_dims = config_dict['hidden_dims']
    kernel_dim = config_dict['kernel_dim']
    k_lipschitz = config_dict['k_lipschitz']

    # Training parameters
    max_epochs = config_dict['max_epochs']
    patience = config_dict['patience']
    frequency = config_dict['frequency']
    batch_size = config_dict['batch_size']
    lr_list = config_dict['lr']
    loss = config_dict['loss']
    lamb1_list = config_dict['lamb1_list']
    lamb2_list = config_dict['lamb2_list']

    clf_type = config_dict['clf_type']
    fisher_c_list = config_dict['fisher_c']
    noise_epsilon = config_dict['noise_epsilon']

    model_dir = config_dict['model_dir']
    results_dir = config_dict['results_dir']
    stat_dir = config_dict['stat_dir']
    store_results = config_dict['store_results']
    store_stat = config_dict['store_stat']

    use_wandb = config_dict['use_wandb']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for setting in itertools.product(seeds, lr_list, fisher_c_list, name_model_list, lamb1_list, lamb2_list):
        (seed, lr, fisher_c, name_model, lamb1, lamb2) = setting

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        ## Load dataset
        train_loader, val_loader, test_loader, N, output_dim = get_dataset(dataset_name, batch_size=batch_size, split=split, seed=seed)

        logging.info(f'Received the following configuration: seed {seed}')
        logging.info(f'DATASET | '
                     f'dataset_name {dataset_name} - '
                     f'ood_dataset_names {ood_dataset_names} - '
                     f'split {split}')

        ## Train or Load a pre-trained model
        if name_model is not None:
            logging.info(f'MODEL: {name_model}')
            config_dict = OrderedDict(name_model=name_model, model_type=model_type, seed=seed,
                                      dataset_name=dataset_name, split=split, loss=loss, epsilon=noise_epsilon)

            if use_wandb:
                run = wandb.init(project='IEDL', reinit=True,
                                 group=f'{dataset_name}_{ood_dataset_names}',
                                 name=f'{model_type}_{loss}_ep{noise_epsilon}_{seed}')

            model = load_model(directory_model=directory_model, name_model=name_model, model_type=model_type)
            stat_dir = stat_dir + f'{name_model}'

        else:
            logging.info(f'ARCHITECTURE | '
                         f' model_type {model_type} - '
                         f' architecture {architecture} - '
                         f' input_dims {input_dims} - '
                         f' output_dim {output_dim} - '
                         f' hidden_dims {hidden_dims} - '
                         f' kernel_dim {kernel_dim} - '
                         f' k_lipschitz {k_lipschitz}')
            logging.info(f'TRAINING | '
                         f' max_epochs {max_epochs} - '
                         f' patience {patience} - '
                         f' frequency {frequency} - '
                         f' batch_size {batch_size} - '
                         f' lr {lr} - '
                         f' loss {loss}')
            logging.info(f'MODEL PARAMETERS | '
                         f' clf_type {clf_type} - '
                         f' fisher_c {fisher_c} - '
                         f' lamb1 {lamb1} -'
                         f' lamb2 {lamb2}')

            config_dict = OrderedDict(model_type=model_type, seed=seed, dataset_name=dataset_name, split=split,
                                      architecture=architecture, input_dims=input_dims, output_dim=output_dim,
                                      hidden_dims=hidden_dims, kernel_dim=kernel_dim, k_lipschitz=k_lipschitz,
                                      max_epochs=max_epochs, patience=patience, frequency=frequency,
                                      batch_size=batch_size, clf_type=clf_type, lr=lr, loss=loss, fisher_c=fisher_c,
                                      lamb1=lamb1, lamb2=lamb2)

            if use_wandb:
                run = wandb.init(project='IEDL', reinit=True,
                                 group=f'{__file__}_{dataset_name}_{architecture}_{suffix}',
                                 name=f'{model_type}_{seed}_{loss}_lr{lr}_f{fisher_c}_{clf_type}')
                wandb.config.update(config_dict)

            filtered_config_dict = {'seed': seed,
                                    'architecture': architecture,
                                    'input_dims': input_dims,
                                    'output_dim': output_dim,
                                    'hidden_dims': hidden_dims,
                                    'kernel_dim': kernel_dim,
                                    'k_lipschitz': k_lipschitz,
                                    'batch_size': batch_size,
                                    'lr': lr,
                                    'loss': loss,
                                    'clf_type': clf_type,
                                    'fisher_c': fisher_c,
                                    'lamb1': lamb1,
                                    'lamb2': lamb2,
                                    }

            model = create_model[model_type](**filtered_config_dict)

            if torch.cuda.is_available():
                # torch.backends.cudnn.benchmark = True
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
                    model = torch.nn.DataParallel(model)
                    model = model.module

            full_config_name = ''
            for k, v in config_dict.items():
                if isinstance(v, dict):
                    v = flatten(v)
                    v = [str(val) for key, val in v.items()]
                    v = "-".join(v)
                if k != 'name_model':
                    full_config_name += str(v) + '-'
            full_config_name = full_config_name[:-1]

            # model_path = model_dir + f'model-{full_config_name}'
            model_path = model_dir + f'{seed}'
            stat_dir = stat_dir + f'model-{full_config_name}'

            Path(model_dir).mkdir(parents=True, exist_ok=True)

            model.to(device)
            train(model, train_loader, val_loader, max_epochs=max_epochs, frequency=frequency, patience=patience,
                  model_path=model_path, full_config_dict=config_dict, use_wandb=use_wandb, device=device, 
                  output_dim=output_dim)

            model.load_state_dict(torch.load(model_path + '_best')['model_state_dict'])

        ## Test model
        model.to(device)
        model.eval()

        with torch.no_grad():
            id_Y_all, id_X_all, id_alpha_pred_all = compute_X_Y_alpha(model, test_loader, device)

            # Save metrics
            metrics = {}
            scores = {}
            ood_scores = {}
            metrics['id_accuracy'] = accuracy(Y=id_Y_all, alpha=id_alpha_pred_all).tolist()

            for name in ['max_prob', 'max_modified_prob', 'max_alpha', 'alpha0', 'differential_entropy', 'mutual_information']:
                if model_type == "duq" and name != 'max_alpha':
                    continue
                if name == 'max_modified_prob' and model_type != 'menet':
                    continue
                abb_name = name2abbrv[name]
                save_path = None
                if store_stat:
                    save_path = f'{stat_dir}/{config_id}_id_{abb_name}.csv'
                    Path(stat_dir).mkdir(parents=True, exist_ok=True)

                if model_type == 'evnet' or model_type == 'duq':
                    aupr, auroc, score = confidence(Y=id_Y_all, alpha=id_alpha_pred_all, uncertainty_type=name,
                                                    save_path=save_path, return_scores=True)
                elif model_type == 'menet' or model_type == 'ablation':
                    aupr, auroc, score = our_confidence(Y=id_Y_all, alpha=id_alpha_pred_all, uncertainty_type=name,
                                                        save_path=save_path, return_scores=True)
                else:
                    raise NotImplementedError
                metrics[f'id_{abb_name}_apr'], metrics[f'id_{abb_name}_auroc'] = aupr, auroc
                
                scores[f'{abb_name}'] = score

            ood_dataset_loaders = {}
            for ood_dataset_name in ood_dataset_names:
                config_dict['ood_dataset_name'] = ood_dataset_name
                _, _, ood_test_loader, _, _ = get_dataset(ood_dataset_name, batch_size=batch_size,
                                                            split=split, seed=seed)
                ood_dataset_loaders[ood_dataset_name] = ood_test_loader

                ood_Y_all, ood_X_all, ood_alpha_pred_all = compute_X_Y_alpha(model, ood_test_loader, device,
                                                                             noise_epsilon=noise_epsilon)

                if ood_dataset_name == dataset_name and noise_epsilon != 0:
                    metrics['ood_accuracy'] = accuracy(Y=ood_Y_all, alpha=ood_alpha_pred_all).tolist()

                for name in ['max_prob', 'max_modified_prob', 'max_alpha', 'alpha0', 'differential_entropy', 'mutual_information']:
                    if model_type == "duq" and name != 'max_alpha':
                        continue
                    if name == 'max_modified_prob' and model_type != 'menet':
                        continue
                    abb_name = name2abbrv[name]
                    save_path = None
                    if store_stat:
                        save_path = f'{stat_dir}/{config_id}_ood_{abb_name}.csv'
                    if model_type == 'evnet' or model_type == 'duq':
                        aupr, auroc, _, ood_score = anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all,
                                                                      uncertainty_type=name, save_path=save_path, return_scores=True)
                    elif model_type == 'menet' or model_type == 'ablation':
                        aupr, auroc, _, ood_score = our_anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all,
                                                                          uncertainty_type=name, save_path=save_path, return_scores=True)
                    else:
                        raise NotImplementedError
                    metrics[f'ood_{abb_name}_apr'], metrics[f'ood_{abb_name}_auroc'] = aupr, auroc
                    ood_scores[f'{abb_name}'] = ood_score

                print("Metrics: ")
                pprint(metrics)

                if use_wandb:
                    data_df = pd.DataFrame(data=[metrics])
                    wandb_table = wandb.Table(dataframe=data_df)
                    wandb.log({'{}'.format(ood_dataset_name): wandb_table})

                if store_results:
                    row_dict = config_dict.copy()
                    for k, v in config_dict.items():
                        if isinstance(v, list):
                            row_dict[k] = str(v)

                    row_dict.update(metrics)  # shallow copy

                    Path(results_dir).mkdir(parents=True, exist_ok=True)
                    data_writer = DataWriter(dump_period=1)
                    csv_file = f'{results_dir}/{config_id}.csv'
                    data_writer.add(row_dict, csv_file)

        if use_wandb:
            run.finish()

    return


if __name__ == '__main__':
    use_argparse = True

    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('--configid', action='store', type=str, required=True)
        my_parser.add_argument('--suffix', type=str, default='debug', required=False)
        args = my_parser.parse_args()
        args_configid = args.configid
        args_suffix = args.suffix
    else:
        args_configid = 'test'
        args_suffix = 'debug'

    if '/' in args_configid:
        args_configid_split = args_configid.split('/')
        my_config_id = args_configid_split[-1]
        config_tree = '/'.join(args_configid_split[:-1])
    else:
        my_config_id = args_configid
        config_tree = ''

    PROJPATH = os.getcwd()
    cfg_dir = f'{PROJPATH}/configs'
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = f'{PROJPATH}/configs/{config_tree}/{my_config_id}.json'
    logging.info(f'Reading Configuration from {cfg_path}')

    with open(cfg_path) as f:
        proced_config_dict = json.load(f)

    proced_config_dict['config_id'] = my_config_id
    proced_config_dict['suffix'] = args_suffix

    proced_config_dict['model_dir'] = f'{PROJPATH}/saved_models/{my_config_id}/'
    proced_config_dict['results_dir'] = f'{PROJPATH}/saved_models/{my_config_id}/'
    proced_config_dict['stat_dir'] = f'{PROJPATH}/results/{config_tree}_stat/'

    main(proced_config_dict)

