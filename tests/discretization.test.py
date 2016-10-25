import discretization as ds
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
        result = ds.divide_into_equal_intervals(input, 4)
        print(result)
