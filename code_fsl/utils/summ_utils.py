import os
import pandas as pd
import numpy as np
from itertools import product
from pandas.errors import EmptyDataError
from sklearn.metrics import precision_recall_fscore_support
from os.path import abspath, dirname
import h5py
import tables.atom
# import pickle5
from cfg import crn_cols, scale_percent, acc_items
from cfg import prop_cols as proj_prop_cols

try:
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    use_mpi = True
except ImportError:
    mpi_rank, mpi_size, mpi_comm, use_mpi = 0, 1, None, False

# tables.atom.pickle = pickle5

def add_missing_dflt_vals(df, def_dict):
    for colname, defval in def_dict.items():
        if colname in df.columns:
            df[colname].fillna(defval, inplace=True)
        else:
            df[colname] = defval
    return df


def default_str_maker(flt_mean, flt_ci=None):
    if flt_ci is not None:
        pm = 'PM'  # '+/-'
        out_str = f'%+0.2f {pm} %.2f' % (scale_percent * flt_mean, scale_percent * flt_ci)
    else:
        out_str = f'%+0.2f' % scale_percent * flt_mean
    return out_str + '%'


def gen_table(summary_df, row_tree, col_tree, y_col, y_col_ci=None, str_maker=None):
    if str_maker is None:
        str_maker = default_str_maker
    x_cols = row_tree + col_tree
    my_summary_df = summary_df.copy(deep=False)

    row_tree_uniques = [my_summary_df[col].unique().tolist() for col in row_tree]
    col_tree_uniques = [my_summary_df[col].unique().tolist() for col in col_tree]
    x_col_uniques = row_tree_uniques + col_tree_uniques
    np_ndarr = np.full(tuple(len(a) for a in x_col_uniques), np.nan)
    np_ndarr_raveled = np.empty_like(np_ndarr.reshape(-1), dtype=object)
    for i, x_tup in enumerate(product(*x_col_uniques)):
        df = my_summary_df
        for x_col, x_val in zip(x_cols, x_tup):
            df = df[df[x_col] == x_val]
        assert len(df) == 1, f'The combination {x_cols}={x_tup} has {len(df)} rows in it instead of 1: \n{df}'

        entry = df[y_col].values.item()
        if y_col_ci is not None:
            entry_ci = df[y_col_ci].values.item()
        else:
            entry_ci = None
        np_ndarr_raveled[i] = str_maker(entry, entry_ci)

    np_ndarr = np_ndarr_raveled.reshape(*np_ndarr.shape)

    nrows = np.prod(tuple(len(a) for a in row_tree_uniques))
    ncols = np.prod(tuple(len(a) for a in col_tree_uniques))
    out_df = pd.DataFrame(np_ndarr.reshape(nrows, ncols),
                          columns=pd.MultiIndex.from_product(col_tree_uniques),
                          index=pd.MultiIndex.from_product(row_tree_uniques))
    return out_df


# getting the files in each folder
def get_csvh5files(fldr_mini, results_dir):
    fldr_mini_files = []
    for fldr in fldr_mini:
        for file in os.listdir(f'{results_dir}/{fldr}'):
            if file.endswith(".csv") or file.endswith(".h5"):
                fldr_mini_files.append(f'{results_dir}/{fldr}/{file}')
    return fldr_mini_files


def read_csvh5(filename):
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(filename, sep=',')
        except EmptyDataError:
            if mpi_rank == 0:
                print(f'WARNING: {filename} seems to be empty. I will ignore it and move on', flush=True)
            df = None
    elif filename.endswith('.h5'):
        if mpi_rank == 0:
            print(f'    -> Reading {filename}', flush=True)

        with h5py.File(filename, mode='r') as hdf_obj:
            parts = sorted(hdf_obj['main'].keys(), key=lambda x: int(x.split('part')[-1]))

        with pd.HDFStore(filename, mode='r') as hdftbl:
            raw_dfs = [pd.read_hdf(hdftbl, key=f'/main/{part}') for part in parts]

        raw_np_dicts = []
        with h5py.File(filename, mode='r') as hdf_obj:
            for part in parts:
                raw_np_dicts.append({key: val[()] for key, val in
                                     hdf_obj['attachments'][part].items()})
        supplemented_df_list = []
        if mpi_rank == 0:
            print(f'    -> Supplementing {filename}', flush=True)
        for raw_df, raw_np_dict in zip(raw_dfs, raw_np_dicts):
            df_supplemented = raw_df
            supplemented_df_list.append(df_supplemented)

        df = pd.concat(supplemented_df_list, axis=0, ignore_index=True)
    else:
        raise ValueError('extension not implemented.')

    return df


def summarizer(df, prop_cols=None, y_col=None, reg_column=None, cond=None):
    if prop_cols is None:
        prop_cols = proj_prop_cols
    if y_col is None:
        y_col = acc_items

    assert 'split' in df.columns, 'Are you sure you are using the right naming scheme?'
    for i in range(len(crn_cols)):
        assert crn_cols[i] in df.columns, f'Are you sure you are using the right naming scheme {crn_cols[i]}?'

    assert reg_column is not None

    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'novel']

    out_dict = dict()

    if isinstance(reg_column, list):
        all_cols = prop_cols + reg_column
    else:
        all_cols = prop_cols + [reg_column]

    msg_ = 'A prop col may be missing in narrowing down the data. its not safe to keep going.'
    if len(df_val) > 0:
        if len(df_val.groupby(prop_cols)) != 1:
            trash_dir = f'{dirname(dirname(abspath(__file__)))}/trash'
            os.makedirs(trash_dir, exist_ok=True)
            fp = f'{trash_dir}/df_val_dbg_rank{mpi_rank}.csv'
            df_val.to_csv(fp)
            msg_ += f'\nHere is the df_val to look at: {fp}'
            msg_ += f'\nHere is the conditioning: {cond}'
        assert len(df_val.groupby(prop_cols)) == 1, msg_

        df_val_mean = df_val.groupby(all_cols).mean()

        for key in y_col:
            if key in df_val_mean.head():
                stats_test = df_val.groupby(all_cols)[key].agg(['mean', 'count', 'std'])
                df_val_mean[f'{key}_ci'] = 1.96 * stats_test['std'] / np.sqrt(stats_test['count'])

        df_val_mean.reset_index(inplace=True)

        # df_val_mean[f'delta_{y_col}'] = df_val_mean[y_col].max() - df_val_mean[y_col]

        out_dict['val'] = df_val_mean

    if len(df_test) > 0:
        if len(df_test.groupby(prop_cols)) != 1:
            trash_dir = f'{dirname(dirname(abspath(__file__)))}/trash'
            os.makedirs(trash_dir, exist_ok=True)
            fp = f'{trash_dir}/df_val_dbg_rank{mpi_rank}.csv'
            df_test.to_csv(fp)
            msg_ += f'\nHere is the df_val to look at: {fp}'
            msg_ += f'\nHere is the conditioning: {cond}'
        assert len(df_test.groupby(prop_cols)) == 1, msg_

        df_test_mean = df_test.groupby(all_cols).mean()

        for key in y_col:
            if key in df_test_mean.head():
                stats_test = df_test.groupby(all_cols)[key].agg(['mean', 'count', 'std'])
                df_test_mean[f'{key}_ci'] = 1.96 * stats_test['std'] / np.sqrt(stats_test['count'])

        df_test_mean.reset_index(inplace=True)
        out_dict['test'] = df_test_mean

    return out_dict

