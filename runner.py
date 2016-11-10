import data
from classifiers import naive_bayes
from classifiers import ila
import discretization
from collections import namedtuple


class Runner(object):
    def __init__(self):
        self.__data = data.Data()
        self.__confusion_matrix = None  # 2D array
        self.__confusion_matrix_labels = None  # 1D array

    def load_data_set(self, path, target_column):
        self.__data.load(path, target_column)

    def discretize_data_set(self, discretization_method, n_of_sets):
        self.__data.discretizie(discretization_method, number_of_final_sets=n_of_sets)

    def start_crossvalidation(self, number_of_partitions, classifier):
        datasets_generator = self.__data.crossvalidation_gen(number_of_partitions)
        whole_test_set_predicted = []
        whole_test_set_real = []
        for dataset in datasets_generator:
            train_dataset, train_labels = dataset[0]
            test_dataset, test_real_labels = dataset[1]
            # classifier = naive_bayes.NaiveBayes()
            classifier.train(train_dataset, train_labels)
            test_predicted_labels = classifier.predict(test_dataset)
            whole_test_set_predicted.extend(test_predicted_labels)
            whole_test_set_real.extend(test_real_labels)
        self._build_confusion_matrix(whole_test_set_predicted, whole_test_set_real)

    def get_confusion_matrix(self):
        return self.__confusion_matrix

    def get_accuracy(self):
        """Ilość trafionych przez wszystkie - suma przekatniej przez sume macierzy"""
        diagonal_sum = sum(self.__confusion_matrix[i][i] for i in range(len(self.__confusion_matrix)))
        matrix_sum = sum(sum(j for j in i) for i in self.__confusion_matrix)
        return diagonal_sum / matrix_sum

    def get_precision(self):
        """Osobno dla każdej klasy, a potem średnia
        Ile razy trafiłam daną klase/ Ile razy w ogóle wytypowałam daną klase
        Iterowanie po kolumnach albo wierszach pozniej wartośc z przekątnej podzielona przez sume wiersza lub kolumny
        """
        stats_for_classes = []
        for class_index, class_value in enumerate(self.__confusion_matrix_labels):
            stats_for_classes.append(self.__calculate_partial_precision(class_index, class_value))
        return sum(stats_for_classes) / len(stats_for_classes)

    def __calculate_partial_precision(self, class_index, class_value):
        hits = self.__confusion_matrix[class_index][class_index]
        all_shots = sum(self.__confusion_matrix[class_index])
        all_shots = 0.001 if all_shots == 0 else all_shots
        return hits / all_shots

    def get_recall(self):
        """Ile razy trafiłam daną klase/ ilość wystapień klasy w zbiorze - osobno dla kazdej klasy i osobno srednia"""
        stats_for_classes = []
        for class_index, class_value in enumerate(self.__confusion_matrix_labels):
            stats_for_classes.append(self.__calculate_partial_recall(class_index, class_value))
        return sum(stats_for_classes) / len(stats_for_classes)

    def __calculate_partial_recall(self, class_index, class_value):
        hits = self.__confusion_matrix[class_index][class_index]
        all_shots = sum(x[class_index] for x in self.__confusion_matrix)
        all_shots = 0.001 if all_shots == 0 else all_shots
        return hits / all_shots

    def get_Fscore(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2 * (precision * recall) / (precision + recall)

    def _build_confusion_matrix(self, whole_test_set_predicted, whole_test_set_real):
        self.__confusion_matrix_labels = list(set(whole_test_set_predicted + whole_test_set_real))
        matrix_size = len(self.__confusion_matrix_labels)
        self.__confusion_matrix = [[0. for _ in range(matrix_size)] for _ in range(matrix_size)]
        for i, element in enumerate(whole_test_set_predicted):
            real_class = whole_test_set_real[i]
            predicted = whole_test_set_predicted[i]
            self.__increment_matrix_on(real_class, predicted)
        return self.__confusion_matrix

    def __increment_matrix_on(self, real_class, predicted):
        real_class_index = self.__confusion_matrix_labels.index(real_class)
        predicted_class_index = self.__confusion_matrix_labels.index(predicted)
        self.__confusion_matrix[predicted_class_index][real_class_index] += 1.


ALL_DATASETS = [
    # ('datasets/iris.data.txt', 5), # all continuous
    ('datasets/wine.data.txt', 1),  # all continuous
    ('datasets/pima-indians-diabetes.data.txt', 9),  # all continuous
    ('datasets/car.data', 7),  # only nominal values
]
CONTINUOUS_DATASETS = [
    # ('datasets/iris.data.txt', 5), # all continuous
    # ('datasets/wine.data.txt', 1),  # all continuous
    ('datasets/glass.data', 10),  # all continuous
    # ('datasets/pima-indians-diabetes.data.txt', 9),  # all continuous
]
NOMINAL_DATASETS = [
    ('datasets/car.data', 7),  # only nominal values
]
Measures = namedtuple('Measures', 'n similarity_matrix  accuracy precision recall fscore')


def save_data_to(results, filename):
    destination = open(filename, 'w+')
    headline = 'ilosc zbiorow;accuracy;precision;recall;fscore\n'

    for class_name, measures in results.items():
        destination.write('{}\n'.format(class_name))
        destination.write(headline)
        for i, measure in enumerate(measures):
            line = '{};{};{};{};{}\n'.format(measure.n,
                                             measure.accuracy,
                                             measure.precision,
                                             measure.recall,
                                             measure.fscore).replace('.', ',')
            destination.write(line)
        destination.write('\n')


def test_cross_validation(filename):
    results = {}
    for dataset, target_column_index in ALL_DATASETS:
        for i in range(2, 10):
            runner = Runner()
            runner.load_data_set(dataset, target_column_index)
            runner.start_crossvalidation(i)
            measures = Measures(
                i,
                runner.get_confusion_matrix(),
                runner.get_accuracy(),
                runner.get_precision(),
                runner.get_recall(),
                runner.get_Fscore()
            )
            results[dataset] = results.get(dataset, []) + [measures]
    save_data_to(results, filename)


def test_discretization(filename, discretization_method, crosvalidation_sets=3):
    results = {}
    for dataset, target_column_index in CONTINUOUS_DATASETS:
        for i in range(4, 10):
            runner = Runner()
            runner.load_data_set(dataset, target_column_index)
            runner.discretize_data_set(discretization_method, i)
            runner.start_crossvalidation(crosvalidation_sets, ila.Ila())
            measures = Measures(
                i,
                runner.get_confusion_matrix(),
                runner.get_accuracy(),
                runner.get_precision(),
                runner.get_recall(),
                runner.get_Fscore()
            )
            results[dataset] = results.get(dataset, []) + [measures]
            print('dataset:{} n:{}'.format(dataset, i))
    save_data_to(results, filename)


def test_gaus(filename, crosvalidation_sets=3):
    results = {}
    for dataset, target_column_index in CONTINUOUS_DATASETS:
        for i in range(4, 10):
            runner = Runner()
            runner.load_data_set(dataset, target_column_index)
            runner.start_crossvalidation(crosvalidation_sets)
            measures = Measures(
                i,
                runner.get_confusion_matrix(),
                runner.get_accuracy(),
                runner.get_precision(),
                runner.get_recall(),
                runner.get_Fscore()
            )
            results[dataset] = results.get(dataset, []) + [measures]
    save_data_to(results, filename)


def test_everything_on_set(filename, dataset, target_column, crosvalidation_sets=3, discretization_parts=4):
    pass


def run_for_one(discretization_method):
    runner = Runner()
    runner.load_data_set('datasets/wine.data.txt', 1)
    runner.discretize_data_set(discretization_method, 5)
    runner.start_crossvalidation(10, ila.Ila())
    print(runner.get_confusion_matrix())
    measures = Measures(
        -1,
        runner.get_confusion_matrix(),
        runner.get_accuracy(),
        runner.get_precision(),
        runner.get_recall(),
        runner.get_Fscore()
    )
    print(measures)


def main():
    # for _ in range(5):
    #     run_for_one(discretization.divide_into_equal_intervals)
    #
    test_discretization('results/ila_glass_divide_into_equal_intervals.csv',
                        discretization.divide_into_equal_intervals,
                        crosvalidation_sets=6)
    test_discretization('results/ila_glass_divide_elements_equally.csv',
                        discretization.divide_elements_equally,
                        crosvalidation_sets=6)


if __name__ == '__main__':
    main()
