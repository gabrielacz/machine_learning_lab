import unittest
import data


class TestData(unittest.TestCase):
    def test_constructor(self):
        dl = data.Data()

    def test_loading_data(self):
        dl = data.Data()
        dl.load('iris.data.test.txt', 4)
        self.assertListEqual(dl.dataset[0], [5.1, 3.5, 1.4, 0.2])
        self.assertEqual(dl.target[1], 'Iris-setosa')
        dl2 = data.Data()
        dl2.load('../datasets/iris.data.txt', 4, nominal_columns=[1, 1, 0, 1])
        self.assertListEqual(dl2.dataset[0], ['5.1', '3.5', 1.4, '0.2'])
        self.assertEqual(dl2.target[1], 'Iris-setosa')

    def test_spliting_sets(self):
        dl = data.Data()
        dl.load('iris.data.test.txt', 4)
        expected_data = [[[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5]],
                         [[4.9, 3.0, 1.4, 0.2], [6.4, 3.2, 4.5, 1.5], [5.8, 2.7, 5.1, 1.9]],
                         [[4.7, 3.2, 1.3, 0.2], [6.9, 3.1, 4.9, 1.5], [7.1, 3.0, 5.9, 2.1]]]
        expected_target = [['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                           ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                           ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']]
        res_data, res_target = dl.get_splited_dataset(3)
        self.assertListEqual(expected_data, res_data)
        self.assertListEqual(expected_target, res_target)

    def test_crosvalidation(self):
        dl = data.Data()
        d = [[[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5]],
             [[4.9, 3.0, 1.4, 0.2], [6.4, 3.2, 4.5, 1.5], [5.8, 2.7, 5.1, 1.9]],
             [[4.7, 3.2, 1.3, 0.2], [6.9, 3.1, 4.9, 1.5], [7.1, 3.0, 5.9, 2.1]]]
        t = [['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
             ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
             ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']]
        expected = [(([[4.9, 3.0, 1.4, 0.2], [6.4, 3.2, 4.5, 1.5], [5.8, 2.7, 5.1, 1.9],
                       [4.7, 3.2, 1.3, 0.2], [6.9, 3.1, 4.9, 1.5], [7.1, 3.0, 5.9, 2.1]],
                      ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica',
                       'Iris-setosa', 'Iris-versicolor', 'Iris-virginica']),
                     ([[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5]],
                      ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])),

                    (([[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5],
                       [4.7, 3.2, 1.3, 0.2], [6.9, 3.1, 4.9, 1.5], [7.1, 3.0, 5.9, 2.1]],
                      ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica',
                       'Iris-setosa', 'Iris-versicolor', 'Iris-virginica']),
                     ([[4.9, 3.0, 1.4, 0.2], [6.4, 3.2, 4.5, 1.5], [5.8, 2.7, 5.1, 1.9]],
                      ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])),

                    (([[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5],
                       [4.9, 3.0, 1.4, 0.2], [6.4, 3.2, 4.5, 1.5], [5.8, 2.7, 5.1, 1.9]],
                      ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica',
                       'Iris-setosa', 'Iris-versicolor', 'Iris-virginica']),
                     ([[4.7, 3.2, 1.3, 0.2], [6.9, 3.1, 4.9, 1.5], [7.1, 3.0, 5.9, 2.1]],
                      ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))]
        self.assertListEqual([x for x in dl.divide_into_training_and_test(d, t)],expected)

        dl.load('iris.data.test.txt', 4)
        self.assertListEqual([x for x in dl.crossvalidation_gen(3)], expected)

        self.assertListEqual(dl.crossvalidation_sets(3), expected)



        if __name__ == '__main__':
            unittest.main()
