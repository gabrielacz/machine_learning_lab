import discretization as ds
import data
import numpy as np
import unittest


class TestData(unittest.TestCase):
    def test_divide_elements_equally(self):
        input = np.array([1.2, 3.4, 5, 3.2, 4.5, 5, 6.7, 8, 9, 4, 6, 7, 8])
        result = ds.divide_elements_equally(input, 10)
        exp = ['<1.2 3.2>', '<3.4 4.0>', '<4.5 5.0>', '<5.0 5.0>', '<6.0 6.0>',
               '<6.7 6.7>', '<7.0 7.0>', '<8.0 8.0>', '<8.0 8.0>', '<9.0 9.0>']
        self.assertListEqual(result, exp)
        print(result)


    def test_divide_into_equal_intervals(self):
        input = np.array([1.2, 3.4, 5, 3.2, 4.5, 5, 6.7, 8, 9, 4, 6, 7, 8])
        input2 = np.array([1,1,1,4,4,4])
        result = ds.divide_into_equal_intervals(input, 4)
        result2 = ds.divide_into_equal_intervals(input2, 3)
        self.assertListEqual(result,['<1.2 1.2>', '<3.2 5.0>', '<6.0 7.0>', '<8.0 9.0>'])
        self.assertListEqual(result2,['<1 1>', '<x x>', '<4 4>'])

    def test_discretization_method(self):
        dl = data.Data()
        dl.load('iris.data.test.txt', 4)
        dl.discretizie(ds.divide_into_equal_intervals,number_of_final_sets=3)