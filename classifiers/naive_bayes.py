import numpy as np


def is_float(s):
    return s.replace('.', '', 1).isdigit()


class NaiveBayes(object):
    def __init__(self):
        self.probabilities = {}
        self.gassian_data = {}
        self.dataset_divided_by_class = {}
        self.trained = False

    def train(self, dataset, target):
        self._divide_by_classes(dataset, target)
        self._calculate_gassians_params()
        self._build_matrice_for_nominal_values()
        self.trained = True

    def predict(self, dataset): #untested
        if not self.trained:
            raise Exception('Model not trained yet')
        return [self.predict_class(data_row) for data_row in dataset]

    def _divide_by_classes(self, dataset, target):
        for row_data, row_class in zip(dataset, target):
            if row_class not in self.dataset_divided_by_class.keys():
                self.dataset_divided_by_class[row_class] = []
            self.dataset_divided_by_class[row_class].append(row_data)

    def _calculate_gassians_params(self):
        for key, data in self.dataset_divided_by_class.items():
            self.gassian_data[key] = self._summarize_class_elements(data)

    def _summarize_class_elements(self, data):
        attributes_stats = []
        for attr in np.array(data).T:
            if attr[0] is not str:
                attributes_stats.append((np.mean(attr),np.std(attr)))
        return attributes_stats

    def _build_matrice_for_nominal_values(self): #untested
        for key, data in self.dataset_divided_by_class.items():
            self.probabilities[key] = self._summarize_class_nominal_elements(data)

    def _summarize_class_nominal_elements(self, data):
        attributes_stats = []
        for attr in np.array(data).T:
            if attr[0] is str:
                attributes_stats.append(self._stats_for_specific_attr(attr))
        return attributes_stats

    def _stats_for_specific_attr(self, attr): #untested
        stats = {}
        for a in attr:
            stats[a] = stats.get(a, 1.) + 1. # załozenie ze kazdy element jest 1
        for key, value in stats:
            stats[key] = value / len(attr)
        return stats

    # def predict_class(self, data_row): #untested
    #     probabilities = {}
    #     for key,value in self.dataset_divided_by_class.iteritems():
    #         probabilities[key] = 1
    #         if key in self.dataset_divided_by_class:
    #             for i in range(len(self.dataset_divided_by_class[key]).iteritems()):
    #                 probabilities[key] = probabilities[key]*self.dataset_divided_by_class[key]
    #         else:
    #             probabilities[key] = 1/len(self.dataset_divided_by_class)


