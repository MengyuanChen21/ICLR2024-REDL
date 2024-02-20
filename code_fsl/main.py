import os
import time
import json
import itertools
from collections import OrderedDict
from pathlib import Path
import wandb
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn

from FSLTask import FSLTaskMaker
from evaluation import test_misclassication, test_ood_uncertainty
from our_evaluation import our_test_misclassication, our_test_ood_uncertainty
from train import train_iedl, train_medl
from utils.io_utils import DataWriter, logger

def main(config_dict):
    config_id = config_dict['config_id']
    suffix = config_dict['suffix']

    rng_seed = config_dict['rng_seed']
    n_tasks = config_dict['n_tasks']
    source_dataset = config_dict['source_dataset']
    target_dataset = config_dict['target_dataset']
    ood_dataset = config_dict['ood_dataset']
    backbone_arch = config_dict['backbone_arch']
    backbone_method = config_dict['backbone_method']

    n_shots_list = config_dict['n_shots_list']
    n_ways_list = config_dict['n_ways_list']
    split_name_list = config_dict['split_list']

    model_type = config_dict['model_type']

    lbfgs_iters = config_dict['lbfgs_iters']
    store_results = config_dict['store_results']

    dump_period = config_dict['dump_period']
    torch_threads = config_dict['torch_threads']

    results_dir = config_dict['results_dir']
    features_dir = config_dict['features_dir']
    cache_dir = config_dict['cache_dir']
    use_wandb = config_dict['use_wandb']
    print_freq = config_dict['print_freq']

    loss_type = config_dict['loss_type']  # IEDL or EDL
    act_type = config_dict['act_type']  # softplus or exp or softmax
    fisher_coeff_list = config_dict['fisher_coeff_list']
    lamb1_list = config_dict['lamb1_list']
    lamb2_list = config_dict['lamb2_list']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_bs = config_dict['task_bs']  # The number of tasks to stack to each other for parallel optimization

    dsname2abbrv = {'miniImagenet': 'mini', 'tieredImagenet': 'tiered', 'CUB': 'CUB'}

    data_writer = None
    if store_results:
        assert results_dir is not None, 'Please provide results_dir in the config_dict.'
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        data_writer = DataWriter(dump_period=min(dump_period, n_tasks))

    tch_dtype = torch.float32
    untouched_torch_thread = torch.get_num_threads()
    if torch_threads:
        torch.set_num_threads(torch_threads)

    for setting in itertools.product(n_ways_list, n_shots_list, split_name_list, fisher_coeff_list, lamb1_list, lamb2_list):

        (n_ways, n_shots, split, fisher_coeff, lamb1, lamb2) = setting

        if use_wandb:
            name = f'{model_type}_{loss_type}_f{fisher_coeff}_{act_type}_{lamb1}_{lamb2}'

            run = wandb.init(project='IEDL-FSL', reinit=True,
                             group=f'{__file__}_{n_ways}way_{n_shots}shot_{target_dataset}_{source_dataset}_{ood_dataset}',
                             name=f'{name}_{suffix}')
            wandb.define_metric("Test/*", step_metric="Test/n_ep")

            config_dict['n_ways'] = n_ways
            config_dict['n_shots'] = n_shots
            config_dict['split'] = split
            config_dict['fisher_coeff'] = fisher_coeff
            config_dict['lamb1'] = lamb1
            config_dict['lamb2'] = lamb2

            wandb.config.update(config_dict)

        os.makedirs(results_dir, exist_ok=True)
        np.random.seed(rng_seed+12345)
        torch.manual_seed(rng_seed+12345)

        n_query = max(15, n_shots)
        src_ds_abbrv = dsname2abbrv.get(source_dataset, source_dataset)
        trg_ds_abbrv = dsname2abbrv.get(target_dataset, target_dataset)

        config_cols_dict = OrderedDict(n_shots=n_shots, n_ways=n_ways, n_query=n_query, source_dataset=source_dataset,
                                       target_dataset=target_dataset, ood_dataset=ood_dataset, split=split, loss_type=loss_type, 
                                       fisher_coeff=fisher_coeff, act_type=act_type, lbfgs_iters=lbfgs_iters, 
                                       backbone_arch=backbone_arch, backbone_method=backbone_method, rng_seed=rng_seed,
                                       lamb1=lamb1, lamb2=lamb2)

        print('-'*80)
        logger('Current configuration:')
        for (cfg_key_, cfg_val_) in config_cols_dict.items():
            logger(f"  --> {cfg_key_}: {cfg_val_}", flush=True)

        #### get ID dataset
        task_maker = FSLTaskMaker()
        task_maker.reset_global_vars()

        features_bb_dir = f"{features_dir}/{backbone_arch}_{backbone_method}"
        Path(features_bb_dir).mkdir(parents=True, exist_ok=True)
        task_maker.loadDataSet(f'{src_ds_abbrv}2{trg_ds_abbrv}_{split}', features_dir=features_bb_dir)
        logger("* Target Dataset loaded", flush=True)

        n_lsamples = n_ways * n_shots
        n_usamples = n_ways * n_query
        n_samples = n_lsamples + n_usamples

        cfg = {'n_shots': n_shots, 'n_ways': n_ways, 'n_query': n_query, 'seed': rng_seed}
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        task_maker.setRandomStates(cfg, cache_dir=cache_dir)
        ndatas = task_maker.GenerateRunSet(end=n_tasks, cfg=cfg)
        ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_tasks, n_samples, -1)
        labels = torch.arange(n_ways).view(1, 1, n_ways)
        labels = labels.expand(n_tasks, n_shots + n_query, n_ways)
        labels = labels.clone().view(n_tasks, n_samples)

        #### get OOD dataset
        ood_task_maker = FSLTaskMaker()
        ood_task_maker.reset_global_vars()

        ood_ds_abbrv = dsname2abbrv.get(ood_dataset, ood_dataset)
        ood_task_maker.loadDataSet(f'{src_ds_abbrv}2{ood_ds_abbrv}_{split}', features_dir=features_bb_dir)
        logger("* OOD Dataset loaded", flush=True)

        ood_n_query = min(ood_task_maker._min_examples, n_query)
        ood_n_ways = min(n_query * n_ways // ood_n_query, ood_task_maker.data.shape[0])
        ood_cfg = {'n_shots': 0, 'n_ways': ood_n_ways, 'n_query': ood_n_query, 'seed': rng_seed}

        ood_n_samples = ood_n_ways * ood_n_query

        ood_task_maker.setRandomStates(ood_cfg, cache_dir=cache_dir)
        ood_ndatas = ood_task_maker.GenerateRunSet(end=n_tasks, cfg=ood_cfg)
        ood_ndatas = ood_ndatas.permute(0, 2, 1, 3).reshape(n_tasks, ood_n_samples, -1)
        ood_labels = torch.arange(ood_n_ways).view(1, 1, ood_n_ways)
        ood_labels = ood_labels.expand(n_tasks, ood_n_query, ood_n_ways)
        ood_labels = ood_labels.clone().view(n_tasks, ood_n_samples)

        # ---- classification for each task
        metrics = {}

        logger(f'* Starting Classification for {n_tasks} Tasks...')

        all_run_idxs = np.arange(n_tasks)
        all_run_idxs = all_run_idxs.reshape(-1, task_bs)

        n_dim = ndatas.shape[-1]

        for ii, run_idxs in enumerate(tqdm(all_run_idxs)):
            run_idxs = run_idxs.astype(int).tolist()
            batch_dim = len(run_idxs)

            support_data = ndatas[run_idxs][:, :n_lsamples, :].to(device=device, dtype=tch_dtype)
            assert support_data.shape == (batch_dim, n_lsamples, n_dim)

            support_label = labels[run_idxs][:, :n_lsamples].to(device=device, dtype=torch.int64)
            assert support_label.shape == (batch_dim, n_lsamples)

            query_data = ndatas[run_idxs][:, n_lsamples:, :].to(device=device, dtype=tch_dtype)
            assert query_data.shape == (batch_dim, n_usamples, n_dim)

            query_label = labels[run_idxs][:, n_lsamples:].to(device=device, dtype=torch.int64)
            assert query_label.shape == (batch_dim, n_usamples)

            ood_query_data = ood_ndatas[run_idxs].to(device=device, dtype=tch_dtype)
            # assert ood_query_data.shape == (batch_dim, n_usamples, n_dim)

            ood_query_label = ood_labels[run_idxs].to(device=device, dtype=torch.int64)
            # assert ood_query_label.shape == (batch_dim, n_usamples)

            # ---- train classifier
            if model_type == 'evnet':
                classifier = train_iedl(support_data, support_label, loss_type, act_type, fisher_coeff,
                                        max_iter=lbfgs_iters, verbose=False, use_wandb=use_wandb, n_ep=ii)
            elif model_type == 'menet':
                classifier = train_medl(support_data, support_label, loss_type, act_type, lamb1, lamb2, fisher_coeff,
                                        max_iter=lbfgs_iters, verbose=False, use_wandb=use_wandb, n_ep=ii)
            else:
                raise NotImplementedError

            with torch.no_grad():
                if model_type == 'evnet':
                    id_misclassification = test_misclassication(classifier, act_type, query_data, query_label)
                    ood_detect = test_ood_uncertainty(classifier, act_type, query_data, ood_query_data, ood_query_label)
                elif model_type == 'menet':
                    id_misclassification = \
                        our_test_misclassication(classifier, act_type, query_data, query_label, lamb1, lamb2)
                    ood_detect = \
                        our_test_ood_uncertainty(classifier, act_type, query_data, ood_query_data, ood_query_label, lamb1, lamb2)
                else:
                    raise NotImplementedError

            if len(metrics) == 0:
                for k, v in id_misclassification.items():
                    metrics[k] = v
                for k, v in ood_detect.items():
                    metrics[k] = v
            else:
                for k, v in id_misclassification.items():
                    metrics[k] += id_misclassification[k]
                for k, v in ood_detect.items():
                    metrics[k] += ood_detect[k]

            if use_wandb and ((ii == len(all_run_idxs) - 1) or ((ii + 1) % print_freq == 0)):
                for k, v in metrics.items():
                    # wandb.log({'Test/{}'.format(k): 100 * np.mean(v), 'Test/n_ep': (ii + 1) * task_bs})
                    wandb.log({'Test/{}'.format(k): 100 * np.mean(v)})

        # if store_results:
        #     csv_path = f'{results_dir}/{config_id}.csv'
        #     for key, value in metrics.items():
        #         assert len(value) == n_tasks, f'The length of {key} is {len(value)}, not {n_tasks}'
        #     for task_id in range(n_tasks):
        #         row_dict = config_cols_dict.copy()  # shallow copy
        #         row_dict['task_id'] = task_id
        #         for key, value in metrics.items():
        #             row_dict[key] = value[task_id]
        #         data_writer.add(row_dict, csv_path)

        if store_results:
            csv_path = f'{results_dir}/{config_id}.csv'
            for key, value in metrics.items():
                assert len(value) == n_tasks, f'The length of {key} is {len(value)}, not {n_tasks}'
            row_dict = config_cols_dict.copy()  # shallow copy
            for key, value in metrics.items():
                row_dict[key] = 100 * np.mean(value)
                row_dict[f'{key}_std'] = np.std(value)
            data_writer.add(row_dict, csv_path)

        if use_wandb:
            run.finish()

    if store_results:
        # We need to make a final dump before exiting to make sure all data is stored
        data_writer.dump()

    torch.set_num_threads(untouched_torch_thread)

def random_noise_dataset(n_ways, n_shots, batch_size=10, dims=640, mean=0, sigma=1, bounds=None):
    if bounds is None:
        bounds = torch.tensor([0.0, 1.0], dtype=torch.float32)
    clip_lower, clip_upper = bounds
    random_ixs = list(range(n_ways * n_shots))
    random.shuffle(random_ixs)

    generate_dims = (batch_size, n_ways*n_shots, dims)
    X = torch.randn(generate_dims) * sigma + mean
    X = X.clamp(clip_lower, clip_upper)

    return X[:, random_ixs, :]


if __name__ == '__main__':
    torch.set_num_threads(12)
    use_argparse = True
    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('--configid', action='store', type=str, default='1_mini/5w-iedl', required=True)
        my_parser.add_argument('--suffix', type=str, default='debug', required=False)
        args = my_parser.parse_args()
        args_configid = args.configid
        args_suffix = args.suffix
    else:
        args_configid = '1_mini/5w-iedl'
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
    logger(f'Reading Configuration from {cfg_path}', flush=True)

    with open(cfg_path) as f:
        proced_config_dict = json.load(f)

    proced_config_dict['config_id'] = my_config_id
    proced_config_dict['suffix'] = args_suffix
    proced_config_dict['results_dir'] = f'{PROJPATH}/results/{config_tree}_{args_suffix}'
    proced_config_dict['cache_dir'] = f'{PROJPATH}/cache'
    proced_config_dict['features_dir'] = f'{PROJPATH}/features'

    main(proced_config_dict)
