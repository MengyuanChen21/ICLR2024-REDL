from os.path import dirname

####################################################
########          global configs           #########
####################################################

acc_items = ['id_accuracy', 'id_max_p_apr', 'id_max_p_auroc', 'id_max_alp_apr', 'id_max_alp_auroc', 'id_diff_ent_apr', 'id_diff_ent_auroc', 'id_mi_apr', 'id_mi_auroc', 'ood_max_p_apr', 'ood_max_p_auroc', 'ood_alpha0_apr', 'ood_alpha0_auroc', 'ood_diff_entropy_apr', 'ood_mi_apr', 'ood_diff_entropy_auroc', 'ood_mi_auroc']

PROJPATH = f'{dirname(dirname(__file__))}'
smry_tbls_dir = f'{PROJPATH}/summary'

####################################################
########        csv2summ configs           #########
####################################################

results_csv_dir = f'{PROJPATH}/results'
# generating the path to the files to be read
# provided by a seperate .py file

# (folders in results_csv_dir, expname_regcol)
csvdir_expname = [('1_mini_debug', 'iedl')]

expname_regcol = {'iedl': ['fisher_coeff']}
deprecated_cols = []

summ_cond_vars = ['source_dataset', 'target_dataset', 'ood_dataset', 'n_ways', 'n_shots', 'n_query', 'experiment',
                  'loss_type']

prop_cols = ['n_shots', 'n_ways', 'n_query', 'source_dataset', 'target_dataset', 'ood_dataset', 'split', 'loss_type', 
             'act_type', 'lbfgs_iters', 'backbone_method', 'backbone_arch', 'experiment']

prop_cols = prop_cols + deprecated_cols
crn_cols = ['rng_seed', 'task_id']
dfltvals_dict = dict()

scale_percent = 100
