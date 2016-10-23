from classifiers import naive_bayes
import unittest

data = [[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5],
        [4.9, 3.0, 1.4, 0.2], [6.4, 3.2, 4.5, 1.5], [5.8, 2.7, 5.1, 1.9],
        [4.7, 3.2, 1.3, 0.2], [6.9, 3.1, 4.9, 1.5], [7.1, 3.0, 5.9, 2.1]]
data_nom = [['5.1', '3.5', '1.4', '0.2'], ['7.0', '3.2', '4.7', '1.4'], ['6.3', '3.3', '6.0', '2.5'],
            ['4.9', '3.0', '1.4', '0.2'], ['6.4', '3.2', '4.5', '1.5'], ['5.8', '2.7', '5.1', '1.9'],
            ['4.7', '3.2', '1.3', '0.2'], ['6.9', '3.1', '4.9', '1.5'], ['7.1', '3.0', '5.9', '2.1']]
target = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica',
          'Iris-setosa', 'Iris-versicolor', 'Iris-virginica',
          'Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


class TestNaiveBayes(unittest.TestCase):
    def test_init(self):
        nb = naive_bayes.NaiveBayes()

    def test_dividing_data_into_classes(self):
        nb = naive_bayes.NaiveBayes()
        expected_dict = {'Iris-virginica': [[6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1]],
                         'Iris-setosa': [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2]],
                         'Iris-versicolor': [[7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5]]}
        nb._divide_by_classes(data, target)
        self.assertDictEqual(expected_dict, nb.dataset_divided_by_class)

    def test_sumarizzation_of_class_params(self):
        data_to_summarize = [[6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1]]
        expected = [(6.3999999999999995, 0.53541261347363367), (3.0, 0.24494897427831769),
                    (5.666666666666667, 0.40276819911981931), (2.1666666666666665, 0.24944382578492946)]
        nb = naive_bayes.NaiveBayes()
        self.assertListEqual(expected, nb._summarize_class_elements(data_to_summarize))
        pass

    def test_calculate_gausian_params(self):
        expected_gaussian_params = {
            'Iris-setosa': [(4.8999999999999995, 0.163299316185545), (3.2333333333333329, 0.20548046676563253),
                            (1.3666666666666665, 0.047140452079103105), (0.20000000000000004, 2.7755575615628914e-17)],
            'Iris-versicolor': [(6.7666666666666666, 0.26246692913372693), (3.1666666666666665, 0.047140452079103209),
                                (4.7000000000000002, 0.16329931618554536), (1.4666666666666668, 0.047140452079103216)],
            'Iris-virginica': [(6.3999999999999995, 0.53541261347363367), (3.0, 0.24494897427831769),
                               (5.666666666666667, 0.40276819911981931), (2.1666666666666665, 0.24944382578492946)]}
        nb = naive_bayes.NaiveBayes()
        nb._divide_by_classes(data, target)
        self.assertDictEqual({}, nb.gassian_data)
        nb._calculate_gassians_params()
        self.assertDictEqual(expected_gaussian_params, nb.gassian_data)

    # def test_summarize_class_nominal_elements(self):
    #     nb = naive_bayes.NaiveBayes()
    #     nb._divide_by_classes(data_nom, target)
    #     nb._summarize_class_nominal_elements()
    #     input = ['7.0' '6.4' '6.9']


    def test_build_matrice_for_nominal_values(self):
        expected = \
            {'Iris-versicolor': [{'6.4': 0.3333333333333333, '7.0': 0.3333333333333333, '6.9': 0.3333333333333333},
                                 {'3.1': 0.4, '3.2': 0.6},
                                 {'4.9': 0.3333333333333333, '4.7': 0.3333333333333333, '4.5': 0.3333333333333333},
                                 {'1.4': 0.4, '1.5': 0.6}],
             'Iris-virginica': [{'7.1': 0.3333333333333333, '6.3': 0.3333333333333333, '5.8': 0.3333333333333333},
                                {'2.7': 0.3333333333333333, '3.3': 0.3333333333333333, '3.0': 0.3333333333333333},
                                {'6.0': 0.3333333333333333, '5.1': 0.3333333333333333, '5.9': 0.3333333333333333},
                                {'2.5': 0.3333333333333333, '1.9': 0.3333333333333333, '2.1': 0.3333333333333333}],
             'Iris-setosa': [{'5.1': 0.3333333333333333, '4.9': 0.3333333333333333, '4.7': 0.3333333333333333},
                             {'3.2': 0.3333333333333333, '3.0': 0.3333333333333333, '3.5': 0.3333333333333333},
                             {'1.4': 0.6, '1.3': 0.4}, {'0.2': 1.0}]}

        nb = naive_bayes.NaiveBayes()
        nb._divide_by_classes(data_nom, target)
        nb._build_matrice_for_nominal_values()
        self.assertDictEqual(nb.probabilities, expected)

    def test_predict_class(self):
        nb = naive_bayes.NaiveBayes()
        nb.train(data, target)
        predicted_class = nb.predict_class([5.1, 3.5, 1.4, 0.2])
        self.assertEqual('Iris-setosa', predicted_class)
        predicted_classes = nb.predict([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]])
        self.assertListEqual(['Iris-setosa', 'Iris-setosa'], predicted_classes)

    def test_predict_class_nominal_values(self):
        nb = naive_bayes.NaiveBayes()
        nb.train(data_nom, target)
        predicted_class = nb.predict_class(['5.1', '3.5', '1.4', '0.2'])
        self.assertEqual('Iris-setosa', predicted_class)
        predicted_classes = nb.predict([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]])
        self.assertListEqual(['Iris-setosa', 'Iris-setosa'], predicted_classes)

    def test_mixed_gaussian_nominal_values(self):
        d = [[5.1, '3.5', '1.4', '0.2'], [7.0, '3.2', '4.7', '1.4'], [6.3, '3.3', '6.0', '2.5'],
             [4.9, '3.0', '1.4', '0.2'], [6.4, '3.2', '4.5', '1.5'], [5.8, '2.7', '5.1', '1.9'],
             [4.7, '3.2', '1.3', '0.2'], [6.9, '3.1', '4.9', '1.5'], [7.1, '3.0', '5.9', '2.1']]
        t = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica',
             'Iris-setosa', 'Iris-versicolor', 'Iris-virginica',
             'Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        nb = naive_bayes.NaiveBayes()
        nb.train(d,t)
        predicted_class = nb.predict_class([5.1, '3.5', '1.4', '0.2'])
        print(predicted_class)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()


    # mean = 1.3666
    # std = 0.047
    # x = 1.4
    # import math
    # exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
    # res = (1 / (math.sqrt(2*math.pi) * std)) * exponent
    # print(res)
