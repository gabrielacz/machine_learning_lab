import numpy as np

TEMPLATE = '<{} {}>'


def divide_into_equal_intervals(data_column, number_of_final_sets):
    def __find_first_greater_element_index(elem, array):
        i = 0
        while elem > array[i]:
            i += 1
        return i
    sorted_data = sorted(data_column)
    min = sorted_data[0]
    max = sorted_data[-1]
    limits = np.linspace(min, max, number_of_final_sets+1)[1:]
    splited_array = [[] for _ in range(len(limits))]
    for elem in sorted_data:
        index = __find_first_greater_element_index(elem,limits)
        splited_array[index].append(elem)
    return [TEMPLATE.format(x[0], x[-1]) if x else TEMPLATE.format('x', 'x') for x in splited_array]


def divide_elements_equally(data_column, number_of_final_sets):
    sorted_data = sorted(data_column)
    splited_array = np.array_split(sorted_data, number_of_final_sets)
    return [TEMPLATE.format(x[0], x[-1]) for x in splited_array]

    # http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
