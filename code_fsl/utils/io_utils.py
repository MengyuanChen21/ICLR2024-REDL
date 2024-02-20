import os
import io
import json
import time
import hashlib
import tarfile
import subprocess
import h5py
from datetime import datetime
from collections import OrderedDict
from typing import Dict, Any
import torch
import numpy as np
import pandas as pd


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


class DataWriter:
    def __init__(self, dump_period=10):
        self.dump_period = dump_period
        self.file_path = None
        self.data = None

    @property
    def data_len(self):
        dlen = 0
        if self.data is not None:
            dlen = len(list(self.data.values())[0])
        return dlen

    def set_path(self, file_path):
        if file_path is not None:
            if file_path != self.file_path:
                self.dump()
        self.file_path = file_path

    @property
    def file_ext(self):
        assert self.file_path is not None
        if self.file_path.endswith('.csv'):
            return 'csv'
        elif self.file_path.endswith('.h5'):
            return 'h5'
        else:
            raise ValueError(f'Unknown extension for {self.file_path}')

    def add(self, row_dict, file_path):
        self.set_path(file_path)
        assert isinstance(row_dict, OrderedDict)
        if self.data is None:
            self.data = OrderedDict()
        else:
            msg_assert = 'input keys and my columns are different:\n'
            msg_assert = msg_assert + f'  input keys: {set(row_dict.keys())}\n'
            msg_assert = msg_assert + f'  my columns: {set(self.data.keys())}\n'
            assert list(self.data.keys()) == list(row_dict.keys()), msg_assert

        # Checking if any of the numpy arrays have changed shape
        for i, (key, val) in enumerate(row_dict.items()):
            if key not in self.data:
                self.data[key] = []
            if (self.file_ext == 'h5') and self.is_attachment(val):
                if len(self.data[key]) > 0:
                    if val.shape != self.data[key][0].shape:
                        self.dump()

        for i, (key, val) in enumerate(row_dict.items()):
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(val)

        if (self.data_len % self.dump_period == 0) and (self.data_len > 0):
            self.dump()

    def is_attachment(self, var):
        use_np_protocol = isinstance(var, np.ndarray)
        if use_np_protocol:
            use_np_protocol = var.size > 1
        return use_np_protocol

    def dump(self):
        if self.data_len == 0:
            return None

        assert self.file_path is not None

        if self.file_ext == 'csv':
            data_df = pd.DataFrame(self.data)
            columns = list(self.data.keys())
            # Appending the latest row to file_path
            if not os.path.exists(self.file_path):
                data_df.to_csv(self.file_path, mode='w', header=True, index=False, columns=columns)
            else:
                # First, check if we have the same columns
                old_cols = pd.read_csv(self.file_path, nrows=1).columns.tolist()
                old_cols_set = set(old_cols)
                my_cols_set = set(columns)
                msg_assert = 'file columns and my columns are different:\n'
                msg_assert = msg_assert + f'  file cols: {old_cols_set}\n'
                msg_assert = msg_assert + f'  my columns: {my_cols_set}\n'
                assert old_cols_set == my_cols_set, msg_assert
                data_df.to_csv(self.file_path, mode='a', header=False, index=False, columns=old_cols)
        elif self.file_ext == 'h5':
            np_data = {}
            pd_data = {}
            for key, valslist in self.data.items():
                if self.is_attachment(valslist[0]):
                    np_data[key] = np.stack(valslist, axis=0)
                else:
                    pd_data[key] = valslist

            # Writing the main table
            data_df = pd.DataFrame(pd_data)
            nextpart = 0
            if os.path.exists(self.file_path):
                with pd.HDFStore(self.file_path) as hdf:
                    nextpart = len(tuple(x for x in hdf.keys() if x.startswith('/main/part')))
                    assert nextpart not in hdf.keys()
            data_df.to_hdf(self.file_path, key=f'/main/part{nextpart}',
                           mode='a', index=False, append=False)

            # Writing the numpy attachments
            hdf_obj = h5py.File(self.file_path, mode='a', driver='core')
            for key, np_arr in np_data.items():
                hdf_obj.create_dataset(f'/attachments/part{nextpart}/{key}',
                                       shape=np_arr.shape, dtype=np_arr.dtype, data=np_arr,
                                       compression="gzip", compression_opts=9)
            hdf_obj.close()
        else:
            raise ValueError(f'file extension not implemented for {self.file_path}.')

        for key in self.data.keys():
            self.data[key] = []


def append_to_tar(tar_path, file_name, file_like_obj):
    archive = tarfile.open(tar_path, "a")
    info = tarfile.TarInfo(name=file_name)
    file_like_obj.seek(0, io.SEEK_END)
    info.size = file_like_obj.tell()
    info.mtime = time.time()
    file_like_obj.seek(0, io.SEEK_SET)
    archive.addfile(info, file_like_obj)
    archive.close()
    file_like_obj.close()


def logger(*args, **kwargs):
    now = datetime.now()
    dt_string = now.strftime("%H:%M:%S")
    print(f'[{dt_string}] ', end='')
    print(*args, **kwargs)
