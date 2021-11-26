import bisect
import numpy as np
from collections import OrderedDict


class DataSource:
    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def __add__(self, other):
        if isinstance(other, DataSourceEmpty):
            return self
        return CompositeDataSource([self, other])


class DataSourceEmpty(DataSource):
    def __getitem__(self, item):
        raise ValueError("Empty data source cannot be indexed.")

    def __len__(self):
        return 0

    def __add__(self, other):
        return CompositeDataSource([other])


class NdArrayDataSource(DataSource):
    def __init__(self, variables, preprocess=None):
        self.preprocess = preprocess
        if isinstance(variables, list):
            for v in variables:
                if v.shape[0] != variables[0].shape[0]:
                    raise ValueError("Input variables should have same batches")
            self.variable_list = variables
        elif isinstance(variables, OrderedDict):
            batch_size = -1
            self.variable_list = []
            for k, v in variables.items():
                if batch_size < 0:
                    batch_size = v.shape[0]
                else:
                    if v.shape[0] != batch_size:
                        raise ValueError("Input variables should have same batches")
                self.variable_list.append(v)
                print(k)
        else:
            raise ValueError("Input variables should be list or OrderedDict")

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, np.ndarray):
            variables = tuple(v[item] for v in self.variable_list)
        else:
            variables = tuple(v[item:item+1] for v in self.variable_list)
        if callable(self.preprocess):
            variables = self.preprocess(variables)
        return variables

    def __len__(self):
        return self.variable_list[0].shape[0]


class CompositeDataSource(DataSource):
    def __init__(self, other):
        super(CompositeDataSource, self).__init__()
        self.dataset_list = list(other)
        self.num_of_samples = 0
        self.partition_list = [0]
        for d in self.dataset_list:
            self.num_of_samples += len(d)
            self.partition_list.append(self.num_of_samples)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, item):
        idx = bisect.bisect_right(self.partition_list, item)
        if idx == 0 or idx == len(self.partition_list):
            raise ValueError("The index {} exceed the bound ({}) of dataset.".format(item, len(self)))
        #print("item={}, idx={}, part={}".format(item, idx, self.partition_list[idx-1]))
        dataset_index = item - self.partition_list[idx-1]
        return self.dataset_list[idx-1][dataset_index]


class SelectedDataSource(DataSource):
    def __init__(self, dataset, selection):
        self.dataset = dataset
        self.selection = selection

    def __getitem__(self, item):
        return self.dataset[self.selection[item]]

    def __len__(self):
        return len(self.selection)



if __name__ == '__main__':
    import numpy as np
    a = np.arange(10)
    b = np.arange(10)
    dataset1 = NdArrayDataSource([a, b])
    selection = np.argwhere(a > 5)
    dataset2 = SelectedDataSource(dataset1, selection)
    print(len(dataset2))
    print(dataset2[0:3])
    dataset2 += dataset1
    print(len(dataset2))
    for i in range(len(dataset2)):
        print(dataset2[i])

    x=dataset2[13]

