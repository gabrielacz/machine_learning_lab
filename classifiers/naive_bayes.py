import numpy as np
import math


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

    def predict(self, dataset):  # untested
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
            # print(type(attr[0]))
            # if attr[0] is not str:
            if attr[0].dtype.type is not np.str_:
                attributes_stats.append((np.mean(attr), np.std(attr)))
            else:
                attributes_stats.append((None, None))
        return attributes_stats

    def _build_matrice_for_nominal_values(self):
        for key, data in self.dataset_divided_by_class.items():
            self.probabilities[key] = self._summarize_class_nominal_elements(data)

    def _summarize_class_nominal_elements(self, data):
        attributes_stats = []
        for attr in np.array(data).T:
            if attr[0].dtype.type is np.str_:
                attributes_stats.append(self._stats_for_specific_attr(attr))
            else:
                attributes_stats.append({})  # empy in case of mixed args
        return attributes_stats

    def _stats_for_specific_attr(self, attr):
        # TODO P(Y)!  P(y|X) = P(y).P(X|y)/P(X)
        stats = {}
        sum_of_element_apperance = len(attr) + len(set(attr))  # because +1 to avoid 0 prob
        for a in attr:
            stats[a] = stats.get(a, 1.) + 1.  # every elem is min 1 time
        for key, value in stats.items():
            stats[key] = value / sum_of_element_apperance
        return stats

    def predict_class(self, data_row):  # untested
        probabilities = {}
        for classValue, examples in self.dataset_divided_by_class.items():
            probabilities[classValue] = self._calc_prob_for_class(data_row, classValue)
        # print(probabilities)
        return self._choose_best_class(probabilities)

    def _calc_prob_for_class(self, examples, classValue):
        tmp_prob = 1
        for index, example in enumerate(examples):
            # print('class'+classValue)
            # print(example)
            # print(index)
            tmp_prob *= self._calc_prob_for_attribute(classValue, example, index)
        return tmp_prob

    def _calc_prob_for_attribute(self, classValue, example, index):
        if self._is_nominal(classValue, index):
            return self._get_probability_for_nominal_attr(classValue, example, index)
        else:
            # print(self._get_prob_for_continuous_values(classValue, index, example))
            return self._get_prob_for_continuous_values(classValue, index, example)

    def _get_prob_for_continuous_values(self, classValue, index, x):
        mean = self.gassian_data[classValue][index][0]
        std = self.gassian_data[classValue][index][1]
        # TODO Dystrybuanta nie rozkład! Jak jest u wazniaka
        # print('mean')
        # print(mean)
        # print(std)
        # print(x)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
        # print('result')
        # print((1 / (math.sqrt(2*math.pi) * std)) * exponent)
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

    def _get_probability_for_nominal_attr(self, classValue, example, index):
        if example not in self.probabilities[classValue][index]:
            attr = self.probabilities[classValue][index]
            sum_of_element_apperance = len(attr) + len(set(attr))
            return 1 / sum_of_element_apperance
        else:
            return self.probabilities[classValue][index][example]

    def _is_nominal(self, classValue, index):
        return self.gassian_data[classValue][index] == (None, None)

    def _choose_best_class(self, probabilities):
        max_class = None
        max_prob = 0
        for classValue, prob in probabilities.items():
            if max_prob < prob:
                max_class = classValue
                max_prob = prob
        return max_class
