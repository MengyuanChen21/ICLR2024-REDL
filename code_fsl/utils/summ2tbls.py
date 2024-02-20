import numpy as np
import pandas as pd
from collections import defaultdict
from summ_utils import gen_table
from pathlib import Path
from cfg import  paper_tbls_dir, smry_tbls_dir, reg_sources
from cfg import table_sep_cols, main_acc, scale_percent, row_tree, col_tree

# Generating the Summarized DataFrame
summ_tbl= defaultdict(list)

# Some utility functions
def convert_before_after_delta_to_3_rows(df_summ_tbl, keep_cols, main_acc):
    """
    There are 3 accuracy columns that we care about:
      1. 'base_test_acc' otherwise known as 'Before'
      2. 'test_acc' otherwise known as 'After'
      2. 'delta_test_acc' otherwise known as 'Improvement'
      
    This function takes a dataframe which has property columns 
    along with the 3 accuracy columns above, and converts each 
    row into 3 separate rows. For example, it takes
    
      n_ways | n_shots | base_test_acc | test_acc | delta_test_acc
      5      | 5       | 0.50          | 0.55     | 0.05
    
    and then converts it into 
    
      n_ways | n_shots | print_name  | print_val
      5      | 5       | Before      | 0.50     
      5      | 5       | After       | 0.55    
      5      | 5       | Improvement | 0.05    
    
    Note that the ci columns are also considered, but emitted 
    above for simplicity in explanation.
    
    This function makes the table useful for later application
    of the gen_table function with row_tree and col_tree lists.
    """
    rows_list = []
    for row_idx, row in df_summ_tbl.iterrows():
        row_props = {col: row[col] for col in keep_cols}

        for ycol, formal_name in [(f'base_{main_acc}', 'Before'), 
                                  (main_acc, 'After'), 
                                  (f'delta_{main_acc}', 'Improvement')]:
            row_dict = row_props.copy()
            row_dict['print_val'] = row[ycol] 
            row_dict['print_val_ci'] = row[f'{ycol}_ci']
            row_dict['print_name'] = formal_name
            rows_list.append(row_dict)    
    df_comp = pd.DataFrame(rows_list)
    return df_comp

no_ci= False
enable_bf= True
if no_ci:
    def str_maker(flt_mean, flt_ci=None, is_bold=False):
        return f'%0.2f' % (scale_percent*flt_mean) + '%'
else:
    def str_maker(flt_mean, flt_ci=None, is_bold=False):
        pm = '+/-'
        bf_st = 'BFS' if is_bold and enable_bf else ''
        bf_end = 'BFE' if is_bold and enable_bf else ''
        if flt_ci is not None:
            out_str = f'{bf_st}%0.2f{bf_end} {pm} %.2f' % (scale_percent*flt_mean, 
                                                           scale_percent*flt_ci)
        else:
            out_str = f'{bf_st}%0.2f{bf_end}' % (scale_percent*flt_mean)
        return out_str

####################################################################################
####################################################################################
####################################################################################
# -> Generating the cross-domain table in the main paper

reg_src = reg_sources[0]
summary_df = pd.read_csv(f'{smry_tbls_dir}/{reg_src}2test.csv')
crossdomain_smry_df = summary_df[summary_df['target_dataset'] == 'CUB']

print(f'  Generating the cross-domain table in the main paper')
for tbl_id, summary_table in crossdomain_smry_df.groupby(table_sep_cols):
    for i, col in enumerate(table_sep_cols):
        summ_tbl[col].append(tbl_id[i])

    # get the max base acc (0, 750)
    row_ind= summary_table[f'base_{main_acc}'].idxmax()
    max_base= summary_table.loc[row_ind, f'base_{main_acc}']
    summ_tbl[f'base_{main_acc}'].append(max_base)

    max_= summary_table.loc[row_ind, f'base_{main_acc}_ci']
    summ_tbl[f'base_{main_acc}_ci'].append(max_)

    # get the max test acc after firth
    row_ind= summary_table[main_acc].idxmax()
    max_test= summary_table.loc[row_ind, main_acc]
    summ_tbl[main_acc].append(max_test)

    test_acc_ci= summary_table.loc[row_ind, f'{main_acc}_ci']
    delta_test_acc_ci= summary_table.loc[row_ind, f'delta_{main_acc}_ci']
    delta_test_acc= max_test - max_base

    summ_tbl[f'{main_acc}_ci'].append(test_acc_ci)
    summ_tbl[f'delta_{main_acc}'].append(delta_test_acc)
    summ_tbl[f'delta_{main_acc}_ci'].append(delta_test_acc_ci)

df_summ_tbl= pd.DataFrame.from_dict(summ_tbl, orient= 'columns')
df_summ_tbl.sort_values(by= table_sep_cols)
    
keep_cols = ['source_dataset', 'target_dataset', 'n_ways', 'n_shots']
df_comp = convert_before_after_delta_to_3_rows(df_summ_tbl, keep_cols, main_acc)

df_comp["source2target"] = [f'{src} -> {trg}' for src, trg in 
                            zip(df_comp['source_dataset'], 
                                df_comp['target_dataset'])]
out_ycol= "print_val"
tbl = gen_table(df_comp, row_tree, col_tree, out_ycol,
                y_col_ci=out_ycol+'_ci', str_maker=str_maker,
                is_bold_maker=None)

tbl = tbl.reset_index(col_level= 1)  
tbl = tbl.rename(columns={'n_ways': 'Ways', 'n_shots': 'Shot'})

ltx_tbl_str = tbl.to_latex(multicolumn=True, escape=True,
                           multicolumn_format='c|',
                           column_format='|c'*20)
ltx_tbl_str = ltx_tbl_str.replace('+/-', '$\pm$')
ltx_tbl_str = ltx_tbl_str.replace('BFS', '')
ltx_tbl_str = ltx_tbl_str.replace('BFE', '')
ltx_tbl_str = ltx_tbl_str.replace('\\\n', '\\\midrule\n')

if isinstance(tbl_id, (list, tuple)):
    list_tbl_id = tbl_id
else:
    list_tbl_id = [tbl_id]

tbl_name = 'cross_domain'
Path(paper_tbls_dir).mkdir(parents=True, exist_ok=True)
ltx_save_path = f'{paper_tbls_dir}/{tbl_name}.tex'
with open(ltx_save_path, 'w') as f_ptr:
    f_ptr.write(ltx_tbl_str)
    print(f'  *   --> Latex table saved at {ltx_save_path}.')

csv_save_path = f'{paper_tbls_dir}/{tbl_name}.csv'
tbl.to_csv(csv_save_path, index=False)
print(f'  *   --> CSV table saved at {csv_save_path}.\n  ' + '-' * 80)


####################################################################################
####################################################################################
####################################################################################
# -> Generating the tiered-Imagnet table w/w-o augmentation (Table A5 of appendix)
print(f'  Generating the tiered-Imagnet table with and without augmentation')

tiered_df = summary_df[(summary_df['source_dataset'] == 'tieredImagenet') &
                       (summary_df['target_dataset'] == 'tieredImagenet') &
                       (summary_df['n_ways'] >= 10) ]

keep_cols = ['source_dataset', 'target_dataset', 'n_ways', 'n_shots', 'n_aug']
tiered_df_comp = convert_before_after_delta_to_3_rows(tiered_df, keep_cols, main_acc)

naug2str = lambda n_aug: ('No ' if n_aug==0 else f'{int(n_aug)}-') + 'Artificial Samples' 
tiered_df_comp["augmentation"] = [naug2str(n_aug) for n_aug in tiered_df_comp['n_aug']]
out_ycol= "print_val"
tbl_tiered = gen_table(tiered_df_comp, row_tree=row_tree, 
                       col_tree=['augmentation', 'print_name'], 
                       y_col=out_ycol, y_col_ci=out_ycol+'_ci', 
                       str_maker=str_maker, is_bold_maker=None)

tbl_tiered = tbl_tiered.reset_index(col_level=1)
tbl_tiered = tbl_tiered.rename(columns={'n_ways': 'Ways', 'n_shots': 'Shot'})

ltx_tbl_str = tbl_tiered.to_latex(multicolumn=True, escape=True,
                                  multicolumn_format='c|',
                                  column_format='|c'*20)
ltx_tbl_str = ltx_tbl_str.replace('+/-', '$\pm$')
ltx_tbl_str = ltx_tbl_str.replace('BFS', '')
ltx_tbl_str = ltx_tbl_str.replace('BFE', '')
ltx_tbl_str = ltx_tbl_str.replace('\\\n', '\\\midrule\n')

tbl_name = 'tiered'
Path(paper_tbls_dir).mkdir(parents=True, exist_ok=True)
ltx_save_path = f'{paper_tbls_dir}/{tbl_name}.tex'
with open(ltx_save_path, 'w') as f_ptr:
    f_ptr.write(ltx_tbl_str)
    print(f'  *   --> Latex table saved at {ltx_save_path}.')

csv_save_path = f'{paper_tbls_dir}/{tbl_name}.csv'
tbl_tiered.to_csv(csv_save_path, index=False)
print(f'  *   --> CSV table saved at {csv_save_path}.\n  ' + '-' * 80)


####################################################################################
####################################################################################
####################################################################################
# -> Generating 5-ways, 1- and 5-shot, base accuracy table for mini and tiered
print('  Generating the 5-ways 1-shot and 5-ways 5-shot ' + 
      'base accuracy table for mini- and tiered-Imagnet ')

w5_df = summary_df[(summary_df['target_dataset'] == summary_df['source_dataset']) &
                   (summary_df['n_ways'] == 5) ]
tbl_w5 = gen_table(w5_df, row_tree=['n_ways', 'n_shots'], 
                   col_tree=['target_dataset'], y_col=main_acc,
                   y_col_ci=main_acc+'_ci', str_maker=str_maker,
                   is_bold_maker=None)
tbl_w5 = tbl_w5.reset_index(col_level= 0)
tbl_w5 = tbl_w5.rename(columns={'n_ways': 'Ways', 'n_shots': 'Shot'})

tbl_name = '5ways_mini_tiered'
csv_save_path = f'{paper_tbls_dir}/{tbl_name}.csv'
tbl_w5.to_csv(csv_save_path, index=False)
print(f'  *   --> CSV table saved at {csv_save_path}.\n  ' + '-' * 80)
