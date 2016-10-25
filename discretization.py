import numpy as np

TEMPLATE = '<{} {}>'


def divide_into_equal_intervals(data_column, number_of_final_sets):
    def __find_bads(elem, bands):
        for band in bands:
            if elem <= band[1]:
                return band
    limits = np.linspace(min(data_column), max(data_column), number_of_final_sets + 1)
    bands = [(limits[i], limits[i + 1]) for i in range(len(limits) - 1)]
    parsed_data_column = []
    for elem in data_column:
        begin, end = __find_bads(elem, bands)
        parsed_data_column.append(TEMPLATE.format(begin, end))
    return parsed_data_column


def divide_elements_equally(data_column, number_of_final_sets):
    def __find_bads(elem, sets):
        for set in sets:
            if elem in set:
                return min(set),max(set)
    sorted_data = sorted(data_column)
    splited_array = np.array_split(sorted_data, number_of_final_sets)
    parsed_data_column = []
    for elem in data_column:
        begin, end = __find_bads(elem, splited_array)
        parsed_data_column.append(TEMPLATE.format(begin, end))
    return parsed_data_column

    # http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
