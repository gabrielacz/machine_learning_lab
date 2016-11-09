import unittest
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

    def test_summup_examples(self):
        examples = [Example(['small', 'red', 'wedge'], False),
                    Example(['large', 'red', 'wedge'], False),
                    Example(['large', 'red', 'pillar'], True)]
        expected_with_marked = [{'small': 1, 'large': 2}, {'red': 3}, {'wedge': 2, 'pillar': 1}]
        expected_without_marked = [{'small': 1, 'large': 1}, {'red': 2}, {'wedge': 2}]
        classifier = Ila()
        summup_with_marked = classifier.summup_examples(examples)
        self.assertEqual(expected_with_marked, summup_with_marked)
        summup_without_marked = classifier.summup_examples(examples, take_marked=False)
        self.assertEqual(summup_without_marked, expected_without_marked)

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

        current_class = 'yes'
        classifier = Ila()
        other_examples = classifier.all_other_examples(table_divided_by_classes, current_class)
        # print(other_examples)
        self.assertListEqual(expected, other_examples)

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
        summup_rest_examples = [{'small': 1, 'large': 2}, {'red': 3}, {'wedge': 2, 'pillar': 1}]
        attributes_combination = [(0,), (1,), (2,)]
        classifier = Ila()
        max_combination = classifier.find_max_combination(current_class_name, current_examples,
                                                          summup_rest_examples, attributes_combination)

    def test_find_best_value_for_combination(self):
        classifier = Ila()
        combination = (0,)
        summup_current_examples = [{'small': 1, 'large': 2, 'medium': 1},
                                   {'red': 1, 'blue': 1, 'green': 2},
                                   {'sphere': 2, 'pillar': 1, 'brick': 1}]
        summup_rest_examples = [{'small': 1, 'large': 2}, {'red': 3}, {'wedge': 2, 'pillar': 1}]
        expected = ({0: 'medium'}, 1)
