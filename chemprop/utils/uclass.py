import numpy as np


class uncertainty:
    def __init__(self):
        pass


class uncertainties:

    def __init__(self):
        self.uncertainty_collection = {}
        self.uncertainty_count = {}
        self.norm_func = {'MinMax' : lambda x: (x-np.min(x)) / (np.max(x)-np.min(x)),
                          'Zscore' : lambda x: (x-np.mean(x)) / np.std(x),
                          'Simple' : lambda x: x,
                          'Argsort': lambda x: np.argsort(np.argsort(x))}

    def add_unc(self,
                uc_name,
                uc_value):

        if uc_name not in self.uncertainty_collection:
            self.uncertainty_count[uc_name] = 1
            self.uncertainty_collection[uc_name] = uc_value
        else:
            self.uncertainty_count[uc_name] += 1
            self.uncertainty_collection[uc_name] += uc_value

    def mean(self):
        for name in self.uncertainty_collection:
            self.uncertainty_collection[name] = self.uncertainty_collection[name] / self.uncertainty_count[name]

    def get_dict(self):
        return self.uncertainty_collection

    def get_names(self):
        return set(self.uncertainty_collection.keys())

    def simply_add(self, name_list):
        return np.sum([self.uncertainty_collection[name] for name in name_list], axis=0)

    def ensemble(self, weights=None, norm_methods=None):
        if norm_methods is None:
            norm_methods = self.norm_func

        output = {}
        for weight_name in weights:
            if np.sum(list(weights[weight_name].values())) == 0:
                weights[weight_name] = {k:1.0 for k in weights[weight_name].keys()}

            for norm_name in norm_methods:
                output[f'{weight_name}_{norm_name}'] = np.mean([self.norm_func[norm_name](self.uncertainty_collection[uc_name])\
                                                                *weights[weight_name].get(uc_name, 0.) for uc_name in self.get_names()], axis=0)
        return output


