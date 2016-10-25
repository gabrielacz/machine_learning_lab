from classifiers.naive_bayes import NaiveBayes
from data import Data

# test for all 3 choosen sets
data = Data()
data.load('datasets/iris.data.txt',4)
nb = NaiveBayes()
nb.train(data.dataset,data.target)

# crosvalidation tests