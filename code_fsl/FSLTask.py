import os
import pickle
import numpy as np
import torch
from pathlib import Path


class FSLTaskMaker:
    def __init__(self):
        # ========================================================
        #   Module internal functions and variables
        self._min_examples = -1
        self._randStates = None
        self._rsCfg = None
        self.data = None
        self.labels = None
        self.dsName = None
        self.np_random = None

        self._maxRuns = 10000

    def reset_global_vars(self):
        self._min_examples = -1
        self._randStates = None
        self._rsCfg = None
        self.data = None
        self.labels = None
        self.dsName = None
        self.np_random = np.random.RandomState(seed=0)

        # Note: The seed here does not matter for reproducibility, because the object
        #       calls self.np_random.set_state() before using self.np_random in every
        #       method. There is only one exception: If you call
        #       self.GenerateRun(iRun, cfg, regenRState=True) without setting
        #       the np_random state, you may get non-deterministic behavior!
        #       self.setRandomStates calls self.GenerateRun(iRun, cfg, regenRState=True).
        #       However, it sets the self.np_random state carefully before
        #       making this call to ensure reproducibility.

    def _load_pickle(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            labels = [np.full(shape=len(data[key]), fill_value=key)
                      for key in data]
            data = [features for key in data for features in data[key]]
            dataset = dict()
            dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
            dataset['labels'] = torch.LongTensor(np.concatenate(labels))
            return dataset

    # =========================================================
    #    Callable variables and functions from outside the module
    def loadDataSet(self, dsname, features_dir):
        self.dsName = dsname  # Example: self.dsName = 'mini2CUB_novel'

        self._randStates = None
        self._rsCfg = None
        assert os.path.exists(features_dir), f'{features_dir} does not exist'

        # Loading data from files on computer
        dataset = self._load_pickle(f"{features_dir}/{self.dsName}.pkl")

        # Computing the number of items per class in the dataset
        self._min_examples = dataset["labels"].shape[0]
        for i in range(dataset["labels"].shape[0]):
            if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
                self._min_examples = min(self._min_examples, torch.where(
                    dataset["labels"] == dataset["labels"][i])[0].shape[0])
        print("Guaranteed number of items per class: {:d}\n".format(self._min_examples))

        # Generating data tensors
        self.data = torch.zeros((0, self._min_examples, dataset["data"].shape[1]))
        self.labels = dataset["labels"].clone()
        while self.labels.shape[0] > 0:
            indices = torch.where(dataset["labels"] == self.labels[0])[0]
            self.data = torch.cat([self.data, dataset["data"][indices, :]
                                  [:self._min_examples].view(1, self._min_examples, -1)], dim=0)
            indices = torch.where(self.labels != self.labels[0])[0]
            self.labels = self.labels[indices]
        print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
            self.data.shape[0], self.data.shape[1], self.data.shape[2]))

    def GenerateRun(self, iRun, cfg, regenRState=False, generate=True):
        if not regenRState:
            self.np_random.set_state(self._randStates[iRun])

        classes = self.np_random.permutation(np.arange(self.data.shape[0]))[:cfg["n_ways"]]
        shuffle_indices = np.arange(self._min_examples)
        dataset = None
        if generate:
            dataset = torch.zeros(
                (cfg['n_ways'], cfg['n_shots']+cfg['n_query'], self.data.shape[2]))
        for i in range(cfg['n_ways']):
            shuffle_indices = self.np_random.permutation(shuffle_indices)
            if generate:
                dataset[i] = self.data[classes[i], shuffle_indices, :][:cfg['n_shots']+cfg['n_query']]

        return dataset

    def ClassesInRun(self, iRun, cfg):
        self.np_random.set_state(self._randStates[iRun])
        classes = self.np_random.permutation(np.arange(self.data.shape[0]))[:cfg["n_ways"]]
        return classes

    def setRandomStates(self, cfg, cache_dir):
        if self._rsCfg == cfg:
            return

        assert os.path.exists(cache_dir), f'{cache_dir} does not exist'
        rsFile = os.path.join(cache_dir, "RandStates_{}_s{}_q{}_w{}_s{}".format(
            self.dsName, cfg['n_shots'], cfg['n_query'], cfg['n_ways'], cfg['seed']))
        if not os.path.exists(rsFile):
            print("{} does not exist, regenerating it...".format(rsFile))
            self.np_random.seed(cfg['seed'])
            self._randStates = []
            for iRun in range(self._maxRuns):
                self._randStates.append(self.np_random.get_state())
                self.GenerateRun(iRun, cfg, regenRState=True, generate=False)
            torch.save(self._randStates, rsFile)
        else:
            print("reloading random states from file....")
            self._randStates = torch.load(rsFile)
        self._rsCfg = cfg

    def GenerateRunSet(self, start=None, end=None, cfg=None, cache_dir=None):
        if start is None:
            start = 0
        if end is None:
            end = self._maxRuns
        if cfg is None:
            cfg = {"n_shots": 1, "n_ways": 5, "n_query": 15, "seed": 0}

        self.setRandomStates(cfg, cache_dir=cache_dir)
        print("generating task from {} to {}".format(start, end))

        dataset = torch.zeros((end-start, cfg['n_ways'], cfg['n_shots']+cfg['n_query'], self.data.shape[2]))
        for iRun in range(end-start):
            dataset[iRun] = self.GenerateRun(start+iRun, cfg)

        return dataset


# define a main code to test this module
if __name__ == "__main__":

    taskmaker = FSLTaskMaker()
    print("Testing Task loader for Few Shot Learning")

    features_dir = './features/WideResNet_28_10_S2M2_R'
    cache_dir = './cache/WideResNet_28_10_S2M2_R'
    Path(features_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    taskmaker.loadDataSet('mini2CUB_novel', features_dir=features_dir)

    cfg = {"n_shots": 1, "n_ways": 5, "n_query": 15, "seed": 0}
    taskmaker.setRandomStates(cfg, cache_dir=cache_dir)

    run10 = taskmaker.GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])

    run10 = taskmaker.GenerateRun(10, cfg)
    print("Second call:", run10[:2, :2, :2])

    ds = taskmaker.GenerateRunSet(start=2, end=12, cfg=cfg, cache_dir=cache_dir)
    print("Third call:", ds[8, :2, :2, :2])
    print(ds.size())
