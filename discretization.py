import numpy as np

TEMPLATE = '<{} {}>'


def divide_into_equal_intervals(data_column, number_of_final_sets):
    sorted_data = sorted(data_column)
    print(sorted_data)

    min = sorted_data[0]
    max = sorted_data[-1]
    interval_size = (max - min) / number_of_final_sets
    # print('interval_size: {}'.format(interval_size))
    current_limit = min
    splited_list = [[]]
    current_index = 0
    # print(sorted_data)
    for elem in sorted_data:
        # while elem < current_limit:
        #     current_limit+=interval_size

        # print(splited_list)
        # print(current_limit)
        if elem <= current_limit:
            splited_list[current_index].append(elem)
        else:
            current_limit += interval_size
            current_index += 1
            splited_list.append([elem])
    print(splited_list)

    return [TEMPLATE.format(x[0], x[-1]) for x in splited_list]


def divide_elements_equally(data_column, number_of_final_sets):
    sorted_data = sorted(data_column)
    splited_array = np.array_split(sorted_data, number_of_final_sets)
    return [TEMPLATE.format(x[0], x[-1]) for x in splited_array]

    # http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
