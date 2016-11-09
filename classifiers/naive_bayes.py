import numpy as np
import math


def is_float(s):
    return s.replace('.', '', 1).isdigit()


class NaiveBayes(object):
    def __init__(self):
        self._probabilities = {}
        self._gassian_data = {}
        self._dataset_divided_by_class = {}
        self._trained = False

    def train(self, dataset, target):
        self._divide_by_classes(dataset, target)
        self._calculate_gassians_params()
        self._build_matrice_for_nominal_values()
        self._trained = True

    def predict2(self, dataset):
        if not self._trained:
            raise Exception('Model not trained yet')
        predictions = [self._predict_class2(data_row) for data_row in dataset]
        return predictions

    def _predict_class2(self, data_row):
        probabilities = {}
        for classValue, examples in self._dataset_divided_by_class.items():
            probabilities[classValue] = self._calc_prob_of_class2(classValue, data_row, examples)
        # print(data_row)
        # print(probabilities)
        return self._choose_best_class2(probabilities)

    def _calc_prob_of_class2(self, classValue, data_row, examples):
        # TODO dodany +len(set(list)) <---pomyśleć czy ok
        # P(Y)
        count_all_classes_elements = \
            sum([len(v) + len(np.unique(v)) for k, v in self._dataset_divided_by_class.items()])
        count_this_class_elements = len(examples) + len(np.unique(examples))
        class_probability = count_this_class_elements / count_all_classes_elements

        # P(x|Y)
        tmp_prob = 1
        for index, attrb_value in enumerate(data_row):
            tmp_prob *= self._calc_prob_for_attribute2(classValue, index, attrb_value)
        return tmp_prob * class_probability

    def _calc_prob_for_attribute2(self, class_value, index, attrb_value):
        if self._is_nominal(class_value, index):
            return self._get_probability_for_nominal_attr2(class_value, index, attrb_value)
        else:
            return self._get_prob_for_continuous_values2(class_value, index, attrb_value)

    def _get_prob_for_continuous_values2(self, class_value, index, x):
        mean = self._gassian_data[class_value][index][0]
        std = self._gassian_data[class_value][index][1]
        if std == 0.:
            return 1.
        # TODO Dystrybuanta nie rozkład! Jak jest u wazniaka?
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
        return (1. / (math.sqrt(2 * math.pi) * std)) * exponent

    def _get_probability_for_nominal_attr2(self, class_value, index, attrb_value):
        if attrb_value not in self._probabilities[class_value][index]:
            attributes_from_same_class = self._probabilities[class_value][index]
            sum_of_element_apperance = len(attributes_from_same_class) + len(set(attributes_from_same_class))
            return 1. / sum_of_element_apperance
        else:
            return self._probabilities[class_value][index][attrb_value]

    def _choose_best_class2(self, probabilities):
        max_class = None
        max_prob = 0
        for classValue, prob in probabilities.items():
            if max_prob < prob:
                max_class = classValue
                max_prob = prob
        return max_class

    #old method
    def predict(self, dataset):
        if not self._trained:
            raise Exception('Model not trained yet')
        return [self._predict_class(data_row) for data_row in dataset]

    def _predict_class(self, data_row):
        probabilities = {}
        for classValue, examples in self._dataset_divided_by_class.items():
            probabilities[classValue] = self._calc_prob_for_class(data_row, classValue)
        # print(probabilities)

        return self._choose_best_class(probabilities)

    def _calc_prob_for_class(self, examples, classValue):
        tmp_prob = 1
        # TODO dodany +len(set(list)) <---pomyśleć czy ok
        count_all_classes_elements = \
            sum([len(v) + len(np.unique(v)) for k, v in self._dataset_divided_by_class.items()])
        count_this_class_elements = \
            len(self._dataset_divided_by_class[classValue]) \
            + len(np.unique(self._dataset_divided_by_class[classValue]))
        class_probability = count_this_class_elements / count_all_classes_elements
        for index, example in enumerate(examples):
            tmp_prob *= self._calc_prob_for_attribute(classValue, example, index)
        return tmp_prob * class_probability

    def _choose_best_class(self, probabilities):
        max_class = None
        max_prob = 0
        for classValue, prob in probabilities.items():
            if max_prob < prob:
                max_class = classValue
                max_prob = prob
        return max_class

    def _divide_by_classes(self, dataset, target):
        for row_data, row_class in zip(dataset, target):
            if row_class not in self._dataset_divided_by_class.keys():
                self._dataset_divided_by_class[row_class] = []
            self._dataset_divided_by_class[row_class].append(row_data)

    def _calculate_gassians_params(self):
        for key, data in self._dataset_divided_by_class.items():
            self._gassian_data[key] = self._summarize_class_elements(data)

    def _summarize_class_elements(self, data):
        attributes_stats = []
        for attr in np.array(data).T:
            if attr[0].dtype.type is not np.str_:
                attributes_stats.append((np.mean(attr), np.std(attr)))
            else:
                attributes_stats.append((None, None))
        return attributes_stats

    def _build_matrice_for_nominal_values(self):
        for key, data in self._dataset_divided_by_class.items():
            self._probabilities[key] = self._summarize_class_nominal_elements(data)

    def _summarize_class_nominal_elements(self, data):
        attributes_stats = []
        for attr in np.array(data).T:
            if attr[0].dtype.type is np.str_:
                attributes_stats.append(self._stats_for_specific_attr(attr))
            else:
                attributes_stats.append({})  # empty in case of mixed args (one nominal one continuous)
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

    def _calc_prob_for_attribute(self, classValue, example, index):
        if self._is_nominal(classValue, index):
            return self._get_probability_for_nominal_attr(classValue, example, index)
        else:
            return self._get_prob_for_continuous_values(classValue, index, example)

    def _get_prob_for_continuous_values(self, classValue, index, x):
        mean = self._gassian_data[classValue][index][0]
        std = self._gassian_data[classValue][index][1]
        if std == 0.:
            return 1.
        # TODO Dystrybuanta nie rozkład! Jak jest u wazniaka?
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
        return (1. / (math.sqrt(2 * math.pi) * std)) * exponent

    def _get_probability_for_nominal_attr(self, classValue, example, index):
        if example not in self._probabilities[classValue][index]:
            attr = self._probabilities[classValue][index]
            sum_of_element_apperance = len(attr) + len(set(attr))
            return 1. / sum_of_element_apperance
        else:
            return self._probabilities[classValue][index][example]

    def _is_nominal(self, classValue, index):
        return self._gassian_data[classValue][index] == (None, None)
