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

    def __premise_to_str(self):
        formated_premises = [' {} = {} '.format(attr_index, value) for attr_index, value in
                             self.attributes_values.items()]
        return ' and '.join(formated_premises)


class Ila(object):
    def __init__(self):
        self._rules = []  # TODO choose datatype

    def train(self, dataset, target):
        divided = self.divide_by_classes(dataset, target)
        self.iterate_over_subtables(divided)

    # def all_examples_are_marked(self, examples):
    #     return all(x.is_marked for x in examples)

    def iterate_over_subtables(self, table_divided_by_class):
        for current_class_name, current_examples in table_divided_by_class.items():
            nb_of_attributes = len(current_examples[0])
            rest_examples = self.all_other_examples(table_divided_by_class, current_class_name)
            summup_rest_examples = self.summup_examples(rest_examples, take_marked=True)
            j = 1
            for _ in range(nb_of_attributes):  # ile ma byc atryb w kombinacji dla j nal do 1, 2, 3, liczny attrybutów
                attributes_combination = list(itertools.combinations(range(nb_of_attributes), j))
                last_max_combination = not None
                while last_max_combination:
                    max_rule, last_max_combination = \
                        self.find_max_combination(current_class_name,
                                                  current_examples,
                                                  summup_rest_examples,
                                                  attributes_combination)
                    self._rules.append(max_rule)
                    if max_rule:  # could be None
                        attributes_combination.remove(last_max_combination)
                j += 1

                #     TODO check if there are still unmatch examples in this class

    def find_max_combination(self, current_class_name, current_examples, summup_rest_examples, attributes_combination):
        max_rule = None
        summup_current_examples = self.summup_examples(current_examples, take_marked=False)
        best_value_for_all_combinatoins = None
        occurences_of_best = 0
        best_combination = None
        for combination in attributes_combination:
            best_for_current_combination, occurences_of_current_best = \
                self.find_best_value_for_combination(combination, summup_current_examples, summup_rest_examples, current_examples)
            if occurences_of_current_best and occurences_of_current_best > occurences_of_best:
                best_value_for_all_combinatoins = best_for_current_combination
                occurences_of_best = occurences_of_current_best
                best_combination = combination
        if best_combination:
            max_rule = Rule(best_value_for_all_combinatoins, current_class_name)
        return max_rule, best_combination

    def find_best_value_for_combination(self, combination, summup_current_examples, summup_rest_examples):

        # for attrib_value, nb_of_occurences in summup_current_examples.items():


        return None,None

    def all_other_examples(self, table_divided_by_class, current_class_name):  # tested
        rest_examples = []
        for class_name, examples in table_divided_by_class.items():
            if class_name != current_class_name:
                rest_examples.extend(examples)
        return rest_examples

    def summup_examples(self, examples, take_marked=True):  # tested
        summup = [{} for _ in range(len(examples[0].value))]
        for example in examples:
            for attribute_index, attribute_value in enumerate(example.value):
                if not (not take_marked and example.is_marked):
                    the_same_value_occurences = summup[attribute_index].get(attribute_value, 0) + 1
                    summup[attribute_index][attribute_value] = the_same_value_occurences
        return summup

    @staticmethod
    def divide_by_classes(dataset, target):  # tested
        dataset_divided_by_class = {}
        for row_data, row_class in zip(dataset, target):
            example = Example(row_data, False)
            if row_class not in dataset_divided_by_class.keys():
                dataset_divided_by_class[row_class] = []
            dataset_divided_by_class[row_class].append(example)
        return dataset_divided_by_class

    def predict(self, dataset):
        pass
