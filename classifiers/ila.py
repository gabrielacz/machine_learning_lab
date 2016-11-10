import itertools
import numpy as np


class Example(object):
    def __init__(self, value, is_marked):
        self.is_marked = is_marked
        self.value = value

    def __eq__(self, other):
        return self.value.__eq__(other.value) and self.is_marked.__eq__(other.is_marked)

    def __str__(self):
        return '({}, {})'.format(self.value, self.is_marked)

    def __repr__(self):
        return '({}, {})'.format(self.value, self.is_marked)


class Rule:
    def __init__(self, attributes_values, class_name):
        self.attributes_values = attributes_values
        self.class_name = class_name

    def __str__(self):
        return 'IF {} THEN {}'.format(self.__premise_to_str(), self.class_name)

    def __repr__(self):
        return 'IF {} THEN {}'.format(self.__premise_to_str(), self.class_name)

    def __premise_to_str(self):
        formated_premises = [' {} = {} '.format(attr_index, value) for attr_index, value in
                             self.attributes_values.items()]
        return ' and '.join(formated_premises)

    def is_satisfied(self, row):
        is_satisfied = True
        for index, value in self.attributes_values.items():
            is_satisfied = is_satisfied and row[index] == value
        return is_satisfied


class Ila(object):
    def __init__(self):
        self._rules = []

    def train(self, dataset, target):
        divided = self.divide_by_classes(dataset, target)
        self.iterate_over_subtables(divided)

    def iterate_over_subtables(self, table_divided_by_class):
        for current_class_name, current_examples in table_divided_by_class.items():
            nb_of_attributes = len(current_examples[0].value)
            rest_examples = self.merge_other_examples(table_divided_by_class, current_class_name)
            j = 1
            for _ in range(nb_of_attributes):  # ile ma byc atryb w kombinacji dla j nal do 1, 2, 3, liczny attrybutów
                attributes_combination = list(itertools.combinations(range(nb_of_attributes), j))
                last_max_combination = not None
                while last_max_combination:
                    max_rule, last_max_combination = \
                        self.find_max_combination(current_class_name,
                                                  current_examples,
                                                  rest_examples,
                                                  attributes_combination)
                    if max_rule:  # could be None
                        self._rules.append(max_rule)
                        attributes_combination.remove(last_max_combination)
                j += 1
                # TODO check if there are still unmatch examples in this class

    def find_max_combination(self, current_class_name, current_examples, rest_examples, attributes_combination):
        max_rule = None
        best_value_for_all_combinations = None
        occurrences_of_best = 0
        marked_examples_indexes = []
        best_combination = None
        for combination in attributes_combination:
            best_for_current_combination, occurences_of_current_best, tmp_marked_examples_indexes = \
                self.find_best_value_for_combination(combination, current_examples, rest_examples)
            if occurences_of_current_best and occurences_of_current_best > occurrences_of_best:
                best_value_for_all_combinations = best_for_current_combination
                occurrences_of_best = occurences_of_current_best
                best_combination = combination
                marked_examples_indexes = tmp_marked_examples_indexes
        if best_combination:
            max_rule = Rule(best_value_for_all_combinations, current_class_name)
            for example_index in marked_examples_indexes:
                current_examples[example_index].is_marked = True
                marked_examples_indexes = []
        return max_rule, best_combination

    def find_best_value_for_combination(self, combination, current_examples, rest_examples):  # tested
        best_combination = {}  # key is attr column and value is its value
        timmes_best_ocurent = 0
        ocurrences_of_attr = self.calculate_occurences(combination, current_examples)
        for attrb, nb_of_occurences in ocurrences_of_attr.items():
            is_in_rest_example = self.is_given_attribiute_value_in_table(attrb, combination, rest_examples)
            if nb_of_occurences > timmes_best_ocurent and not is_in_rest_example:
                timmes_best_ocurent = nb_of_occurences
                best_combination = {}
                for i, elem in enumerate(attrb):
                    best_combination[combination[i]] = elem
        # mark examples with best_combination
        marked_examples_indexes = []
        for exam_index, example in enumerate(current_examples):
            matches = [example.value[attr_index] == attr_value for attr_index, attr_value in best_combination.items()]
            if matches and all(matches):
                marked_examples_indexes.append(exam_index)
                # example.is_marked = True

        return best_combination, timmes_best_ocurent, marked_examples_indexes

    def calculate_occurences(self, combination, current_examples):  # tested
        ocurrences_of_attr = {}
        for example in current_examples:
            if not example.is_marked:
                example_values = []
                for index in combination:
                    example_values.append(example.value[index])
                ocurrences_of_attr[tuple(example_values)] = ocurrences_of_attr.get(tuple(example_values), 0) + 1
        return ocurrences_of_attr

    def merge_other_examples(self, table_divided_by_class, current_class_name):  # tested
        rest_examples = []
        for class_name, examples in table_divided_by_class.items():
            if class_name != current_class_name:
                rest_examples.extend(examples)
        return rest_examples

    @staticmethod
    def divide_by_classes(dataset, target):  # tested
        dataset_divided_by_class = {}
        for row_data, row_class in zip(dataset, target):
            example = Example(row_data, False)
            if row_class not in dataset_divided_by_class.keys():
                dataset_divided_by_class[row_class] = []
            dataset_divided_by_class[row_class].append(example)
        return dataset_divided_by_class

    def is_given_attribiute_value_in_table(self, attrb, combination, rest_examples):  # tested
        for example in rest_examples:
            example_values = []
            for index in combination:
                example_values.append(example.value[index])
            if tuple(example_values) == (attrb):
                return True
        return False

    #####PREDICTION
    def predict(self, x_test):
        predicted = []
        for row in x_test:
            predicted.append(self.__predict(row))
        return predicted

    def __predict(self, row):
        for rule in self._rules:
            if rule.is_satisfied(row):
                return rule.class_name
        return '<undefined>'
