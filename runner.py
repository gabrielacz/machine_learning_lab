import data
from classifiers import naive_bayes
import discretization


# wpływ podziały na przedziały w croswalidacji
# liczba przedziałów dyskretyzacji w met1, met2, gaussie

class Runner(object):
    def __init__(self):
        self.__data = data.Data()
        self.__confusion_matrix = None  # 2D array
        self.__confusion_matrix_labels = None  # 1D array

    def load_data_set(self, path, target_column):
        self.__data.load(path, target_column)

    def discretize_data_set(self, discretization_method, n_of_sets):
        self.__data.discretizie(discretization_method, number_of_final_sets=n_of_sets)

    def start_crossvalidation(self, number_of_partitions):
        datasets_generator = self.__data.crossvalidation_gen(number_of_partitions)
        whole_test_set_predicted = []
        whole_test_set_real = []
        for dataset in datasets_generator:
            train_dataset, train_labels = dataset[0]
            test_dataset, test_real_labels = dataset[1]
            nb = naive_bayes.NaiveBayes()
            nb.train(train_dataset, test_real_labels)
            test_predicted_labels = nb.predict(test_dataset)
            whole_test_set_predicted.extend(test_predicted_labels)
            whole_test_set_real.extend(test_real_labels)
        self.__build_confusion_matrix(whole_test_set_predicted, whole_test_set_real)

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
        return hits / all_shots

    def get_Fscore(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2*(precision * recall) / (precision + recall)

    def __build_confusion_matrix(self, whole_test_set_predicted, whole_test_set_real):
        self.__confusion_matrix_labels = list(set(whole_test_set_predicted + whole_test_set_real))
        matrix_size = len(self.__confusion_matrix_labels)
        self.__confusion_matrix = [[0. for _ in range(matrix_size)] for _ in range(matrix_size)]
        for i, element in enumerate(whole_test_set_predicted):
            real_class = whole_test_set_real[i]
            predicted = whole_test_set_predicted[i]
            self.__increment_matrix_on(real_class, predicted)

    def __increment_matrix_on(self, real_class, predicted):
        real_class_index = self.__confusion_matrix_labels.index(real_class)
        predicted_class_index = self.__confusion_matrix_labels.index(predicted)
        self.__confusion_matrix[predicted_class_index][real_class_index] += 1.


def main():
    runner = Runner()
    runner.load_data_set('datasets/iris.data.txt', 5)
    runner.start_crossvalidation(3)
    print(runner.get_confusion_matrix())
    print(runner.get_accuracy())
    print(runner.get_precision())
    print(runner.get_recall())
    print(runner.get_Fscore())
    # datasets_path = []
    # for dataset_path in datasets_path:


if __name__ == '__main__':
    main()
