import unittest

import discretization
from data import Data
from classifiers.ila import Ila, Example


class TestIla(unittest.TestCase):
    def test_init(self):
        classifier = Ila()

    def test_divide_by_classes(self):
        x_train = [['medium', 'blue', 'brick'],
                   ['small', 'red', 'wedge'],
                   ['small', 'red', 'sphere'],
                   ['large', 'red', 'wedge'],
                   ['large', 'green', 'pillar'],
                   ['large', 'red', 'pillar'],
                   ['large', 'green', 'sphere']]
        y_train = ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes']
        expected_res = {'no': [Example(['small', 'red', 'wedge'], False),
                               Example(['large', 'red', 'wedge'], False),
                               Example(['large', 'red', 'pillar'], False)],
                        'yes': [Example(['medium', 'blue', 'brick'], False),
                                Example(['small', 'red', 'sphere'], False),
                                Example(['large', 'green', 'pillar'], False),
                                Example(['large', 'green', 'sphere'], False)]
                        }

        classifier = Ila()
        divided = classifier.divide_by_classes(x_train, y_train)
        self.assertEqual(divided, expected_res)

    def test_all_other_examples(self):
        table_divided_by_classes = {
            'no': [Example(['small', 'red', 'wedge'], False),
                   Example(['large', 'red', 'wedge'], False),
                   Example(['large', 'red', 'pillar'], False)],
            'yes': [Example(['medium', 'blue', 'brick'], False),
                    Example(['small', 'red', 'sphere'], False),
                    Example(['large', 'green', 'pillar'], False),
                    Example(['large', 'green', 'sphere'], False)],
            'other': [Example(['medium', 'blue', 'brick'], False),
                      Example(['small', 'red', 'sphere'], False),
                      Example(['large', 'green', 'pillar'], False),
                      Example(['large', 'green', 'sphere'], False)]
        }
        expected = [Example(['small', 'red', 'wedge'], False), Example(['large', 'red', 'wedge'], False),
                    Example(['large', 'red', 'pillar'], False), Example(['medium', 'blue', 'brick'], False),
                    Example(['small', 'red', 'sphere'], False), Example(['large', 'green', 'pillar'], False),
                    Example(['large', 'green', 'sphere'], False)]

        a = [(['medium', 'blue', 'brick'], False), (['small', 'red', 'sphere'], False),
             (['large', 'green', 'pillar'], False), (['large', 'green', 'sphere'], False),
             (['small', 'red', 'wedge'], False), (['large', 'red', 'wedge'], False),
             (['large', 'red', 'pillar'], False)]

        current_class = 'yes'
        classifier = Ila()
        other_examples = classifier.merge_other_examples(table_divided_by_classes, current_class)
        self.assertEqual(len(expected), len(other_examples))

    def test_is_given_attribiute_value_in_table(self):
        attrb_which_is_in_table = ('small',)
        attrb_which_isint_in_table = ('medium',)
        combination = (0,)
        rest_examples = [Example(['small', 'red', 'wedge'], False),
                         Example(['large', 'red', 'wedge'], False),
                         Example(['large', 'red', 'pillar'], False)]
        classifier = Ila()
        is_in_table = classifier.is_given_attribiute_value_in_table(attrb_which_is_in_table, combination, rest_examples)
        isnt_in_table = \
            classifier.is_given_attribiute_value_in_table(attrb_which_isint_in_table, combination, rest_examples)
        self.assertTrue(is_in_table)
        self.assertTrue(not isnt_in_table)

        attrb_which_is_in_table = ('small', 'red')
        attrb_which_isint_in_table = ('medium', 'blue')
        combination = (0, 1)
        is_in_table = classifier.is_given_attribiute_value_in_table(attrb_which_is_in_table, combination, rest_examples)
        isnt_in_table = \
            classifier.is_given_attribiute_value_in_table(attrb_which_isint_in_table, combination, rest_examples)
        self.assertTrue(is_in_table)
        self.assertTrue(not isnt_in_table)

    def test_find_best_value_for_combination(self):
        classifier = Ila()
        combination = (0,)
        current_examples = [Example(['medium', 'blue', 'brick'], False),
                            Example(['small', 'red', 'sphere'], False),
                            Example(['large', 'green', 'pillar'], False),
                            Example(['large', 'green', 'sphere'], False)]
        rest_examples = [Example(['small', 'red', 'wedge'], False),
                         Example(['large', 'red', 'wedge'], False),
                         Example(['large', 'red', 'pillar'], False)]
        # expected = ({0: 'medium'}, 1)
        # result = classifier.find_best_value_for_combination(combination, current_examples, rest_examples)
        # self.assertEqual(result, expected)
        combination = (0, 1)
        expected = ({0: 'large', 1: 'green'}, 2)
        result = classifier.find_best_value_for_combination(combination, current_examples, rest_examples)
        self.assertEqual(result, expected)

    def test_find_max_combination(self):
        """
        {'no': [Example(['small', 'red', 'wedge'], False),
                Example(['large', 'red', 'wedge'], False),
                Example(['large', 'red', 'pillar'], False)],
        'yes': [Example(['medium', 'blue', 'brick'], False),
                Example(['small', 'red', 'sphere'], False),
                Example(['large', 'green', 'pillar'], False),
                Example(['large', 'green', 'sphere'], False)]}
        """
        current_class_name = 'yes'
        current_examples = [Example(['medium', 'blue', 'brick'], False),
                            Example(['small', 'red', 'sphere'], False),
                            Example(['large', 'green', 'pillar'], False),
                            Example(['large', 'green', 'sphere'], False)]
        summup_rest_examples = [Example(['small', 'red', 'wedge'], True),
                                Example(['large', 'red', 'wedge'], True),
                                Example(['large', 'red', 'pillar'], True)]
        attributes_combination = [(0,), (1,), (2,)]

        classifier = Ila()
        max_combination, comb = classifier.find_max_combination(current_class_name, current_examples,
                                                                summup_rest_examples, attributes_combination)
        print(max_combination)
        print(classifier._rules)
        print(current_examples)

    def test_train(self):
        # prawidłowe reguły:
        # IF color=green THEN yes
        # IF size=medium THEN yes
        # IF shape=sphere THEN yes
        # IF shape=wedge THEN no
        # IF size=large AND color=red THEN no
        x_train = [['medium', 'blue', 'brick'],
                   ['small', 'red', 'wedge'],
                   ['small', 'red', 'sphere'],
                   ['large', 'red', 'wedge'],
                   ['large', 'green', 'pillar'],
                   ['large', 'red', 'pillar'],
                   ['large', 'green', 'sphere']]
        y_train = ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes']
        classifier = Ila()
        classifier.train(x_train,y_train)
        print(classifier._rules)

    def test_predict(self):
        data = Data()
        data.load('..\datasets\iris.data.txt',5)
        data.discretizie(discretization.divide_into_equal_intervals)
        x_train = data.dataset
        y_train = data.target
        classifier = Ila()
        classifier.train(x_train, y_train)
        print(classifier._rules)

