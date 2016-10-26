import numpy as np
import math


def is_float(s):
    return s.replace('.', '', 1).isdigit()


class NaiveBayes(object):
    def __init__(self):
        self.__probabilities = {}
        self.__gassian_data = {}
        self.__dataset_divided_by_class = {}
        self.__trained = False

    def train(self, dataset, target):
        self._divide_by_classes(dataset, target)
        self._calculate_gassians_params()
        self._build_matrice_for_nominal_values()
        self.__trained = True

    def predict(self, dataset):
        if not self.__trained:
            raise Exception('Model not trained yet')
        return [self._predict_class(data_row) for data_row in dataset]

    def _divide_by_classes(self, dataset, target):
        for row_data, row_class in zip(dataset, target):
            if row_class not in self.__dataset_divided_by_class.keys():
                self.__dataset_divided_by_class[row_class] = []
            self.__dataset_divided_by_class[row_class].append(row_data)

    def _calculate_gassians_params(self):
        for key, data in self.__dataset_divided_by_class.items():
            self.__gassian_data[key] = self._summarize_class_elements(data)

    def _summarize_class_elements(self, data):
        attributes_stats = []
        for attr in np.array(data).T:
            if attr[0].dtype.type is not np.str_:
                attributes_stats.append((np.mean(attr), np.std(attr)))
            else:
                attributes_stats.append((None, None))
        return attributes_stats

    def _build_matrice_for_nominal_values(self):
        for key, data in self.__dataset_divided_by_class.items():
            self.__probabilities[key] = self._summarize_class_nominal_elements(data)

    def _summarize_class_nominal_elements(self, data):
        attributes_stats = []
        for attr in np.array(data).T:
            if attr[0].dtype.type is np.str_:
                attributes_stats.append(self._stats_for_specific_attr(attr))
            else:
                attributes_stats.append({})  # empty in case of mixed args
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

    def _predict_class(self, data_row):
        probabilities = {}
        for classValue, examples in self.__dataset_divided_by_class.items():
            probabilities[classValue] = self._calc_prob_for_class(data_row, classValue)
        return self._choose_best_class(probabilities)

    def _calc_prob_for_class(self, examples, classValue):
        tmp_prob = 1
        count_all_classes_elements = sum([len(v) for k, v in self.__dataset_divided_by_class.items()])
        count_this_class_elements = len(self.__dataset_divided_by_class[classValue])
        class_probability = count_this_class_elements / count_all_classes_elements
        for index, example in enumerate(examples):
            tmp_prob *= self._calc_prob_for_attribute(classValue, example, index)
        return tmp_prob * class_probability

    def _calc_prob_for_attribute(self, classValue, example, index):
        if self._is_nominal(classValue, index):
            return self._get_probability_for_nominal_attr(classValue, example, index)
        else:
            return self._get_prob_for_continuous_values(classValue, index, example)

    def _get_prob_for_continuous_values(self, classValue, index, x):
        mean = self.__gassian_data[classValue][index][0]
        std = self.__gassian_data[classValue][index][1]
        if std == 0:
            return 1
        # TODO Dystrybuanta nie rozkład! Jak jest u wazniaka?
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

    def _get_probability_for_nominal_attr(self, classValue, example, index):
        if example not in self.__probabilities[classValue][index]:
            attr = self.__probabilities[classValue][index]
            sum_of_element_apperance = len(attr) + len(set(attr))
            return 1 / sum_of_element_apperance
        else:
            return self.__probabilities[classValue][index][example]

    def _is_nominal(self, classValue, index):
        return self.__gassian_data[classValue][index] == (None, None)

    def _choose_best_class(self, probabilities):
        max_class = None
        max_prob = 0
        for classValue, prob in probabilities.items():
            if max_prob < prob:
                max_class = classValue
                max_prob = prob
        return max_class
