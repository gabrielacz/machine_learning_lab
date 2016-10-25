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
    sorted_data = sorted(data_column)
    splited_array = np.array_split(sorted_data, number_of_final_sets)

    new_array = []
    for part in splited_array:
        for elem in part:
            new_array.append(TEMPLATE.format(part[0], part[-1]))
    return new_array

    # http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
