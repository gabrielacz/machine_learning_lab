import numpy as np


class NaiveBayes(object):
    def __init__(self, training_set, attributes_types=None, discretization_method=None):
        if attributes_types is None:
            attributes_types = self._identify(training_set)
        self.attributes_types = attributes_types
        self.discretization_method = discretization_method
        self.training_set = training_set
        self._check_dataset(self.training_set, self.attributes_types)


    def _check_dataset(self, training_set, attributes_types):
        if len(training_set) != len(attributes_types):
            raise ValueError("Given attributes types are not covering all attributes")
            # TODO check given params

    def train(self):
        # podzielic na klasy i podzielic na kolumny
        pass

    def predict(self):
        pass
