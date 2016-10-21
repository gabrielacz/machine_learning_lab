import csv
import functools


def is_float(s):
    return s.replace('.', '', 1).isdigit()


def flatten(list):
    return functools.reduce(lambda x, y: x + y, list)


class Data(object):
    def __init__(self):
        self.target = []
        self.dataset = []

    def load(self, filename, target_column, nominal_columns=None):
        lines = list(csv.reader(open(filename, "r")))
        self.dataset, self.target = self._parse_lines(lines, nominal_columns, target_column)

    def _parse_lines(self, lines, nominal_values, target_column):
        target, dataset = [], []
        for line in lines:
            if line:
                target.append(line[target_column])
                dataset.append(self._parse_line(line[:target_column] + line[target_column + 1:], nominal_values))
        return dataset, target

    def _parse_line(self, line, nominal_values=None):
        if nominal_values:
            # TODO check lenght and parsability to float
            return [x if nominal_values[i] else float(x) for i, x in enumerate(line)]
        else:
            return [float(x) if is_float(x) else x for x in line]

    def crossvalidation_sets(self, nuber_of_parts):
        datasets, targets = self.get_splited_dataset(nuber_of_parts)
        return [x for x in self.divide_into_training_and_test(datasets, targets)]

    def crossvalidation_gen(self, nuber_of_parts):
        datasets, targets = self.get_splited_dataset(nuber_of_parts)
        return self.divide_into_training_and_test(datasets, targets )

    def divide_into_training_and_test(self, datasets, targets):
        for i in range(len(datasets)):
            training_set = flatten(datasets[:i] + datasets[i + 1:])
            training_target = flatten(targets[:i] + targets[i + 1:])
            test_set = datasets[i]
            test_target = targets[i]
            yield ((training_set, training_target), (test_set, test_target))

    def get_splited_dataset(self, n):
        datasets = [[] for _ in range(n)]
        targets = [[] for _ in range(n)]
        for key, elem in enumerate(self.dataset):
            datasets[key % n].append(elem)
            targets[key % n].append(self.target[key])
        return datasets, targets
