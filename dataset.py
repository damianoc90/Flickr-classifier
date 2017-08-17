#Cancemi Damiano - W82000075

import os, glob
import numpy as np
from skimage import io as sio
from matplotlib import pyplot as plt
from copy import copy

class Dataset:
    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        classes = filter(lambda f: not f.startswith('.'), os.listdir(path_to_dataset))  # no hidden file
        self.paths = dict()
        for cl in classes:
            self.paths[cl] = sorted(glob.glob(os.path.join(path_to_dataset, cl, "*.jpg")))

    def getImagePath(self, class_name, idx):
        return self.paths[class_name][idx]

    def getClasses(self):
        return sorted(self.paths.keys())

    def getNumberOfClasses(self):
        return len(self.getClasses())

    def getClassLength(self, class_name):
        return len(self.paths[class_name])

    def getLength(self):
        tot = 0
        for cl in self.getClasses():
            tot += self.getClassLength(cl)
        return tot

    def restrictToClasses(self, classes):
        new_paths = {cl: self.paths[cl] for cl in classes}
        self.paths = new_paths

    def splitTrainingTest(self, percent_train):
        training_paths = dict()
        test_paths = dict()
        for cl in self.getClasses():
            # lista dei path relativi alla classe corrente
            paths = self.paths[cl]
            shuffled_paths = np.random.permutation(paths)
            # indice attorno al quale "spezzare" l'array in due parti
            split_idx = int(len(shuffled_paths) * percent_train)
            # salva le prime "split_idx" immagini nel training set e le restanti nel test set
            training_paths[cl] = shuffled_paths[0:split_idx]
            test_paths[cl] = shuffled_paths[split_idx::]

        training_dataset = copy(self)
        training_dataset.paths = training_paths
        test_dataset = copy(self)
        test_dataset.paths = test_paths
        return training_dataset, test_dataset