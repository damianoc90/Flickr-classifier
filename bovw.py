#Cancemi Damiano - W82000075

import numpy as np
from skimage import io as sio
from skimage.feature import daisy
from matplotlib import pyplot as plt
from skimage.color import rgb2grey
from time import time
from random import randint

'''Estrae features da tutto il dataset e ritorna un array di features estratte'''
def extract_features(dataset, n_step):
    if not n_step:
        n_step = 20
    nimgs = dataset.getLength()
    features = list()
    ni = 0  # numero di immagini analizzate finora
    total_time = 0
    for cl in dataset.getClasses():
        paths = dataset.paths[cl]
        for impath in paths:
            t1 = time()
            im = sio.imread(impath, as_grey=True)
            feats = daisy(im, step=n_step)  # estrai features
            feats = feats.reshape((-1, 200))
            features.append(feats)
            t2 = time()
            t3 = t2 - t1
            total_time += t3
            # Stampa un messaggio di avanzamento, con la stima del tempo rimanente
            ni += 1
            if randint(0,5) == 0:
                print "Image {0}/{1} [{2:0.2f}/{3:0.2f} sec]".format(ni, nimgs, t3, t3 * (nimgs - ni))
    print "Stacking all features..."
    stacked = np.vstack(features)
    return stacked

'''Data in input una immagine a colori e il dizionario kmeans, estrae le feature locali dall'immagine e le descrive mediante il modello BOVW costruito.'''
def extract_and_describe(img, kmeans):
    # estrai le feature da una immagine
    features = daisy(rgb2grey(img)).reshape((-1, 200))
    # assegna le feature locali alle parole del vocabolario
    assignments = kmeans.predict(features)
    # calcola l'istogramma
    histogram, _ = np.histogram(assignments, bins=500, range=(0, 499))
    # restituisci l'istogramma normalizzato
    return histogram

'''Prende in input la matrice delle rappresentazioni, il vettore delle etichette, la lista delle classi, la lista dei path alle immagini, l'indice di una immagine e mostra l'immagine e le relative features'''
def display_image_and_representation(X, y, paths, classes, i):
    im = sio.imread(paths[i])
    plt.figure(figsize=(12, 4))
    plt.suptitle("Class: {0} - Image: {1}".format(classes[y[i]], i))
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.plot(X[i])
    plt.show()


def show_image_and_representation(img, image_representation):
    plt.figure(figsize=(13, 4))
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.plot(image_representation)
    plt.show()


def describe_dataset(dataset, kmeans):
    y = list()  # inizializziamo la lista delle etichette
    X = list()  # inizializziamo la lista delle osservazioni
    paths = list()  # inizializziamo la lista dei path

    classes = dataset.getClasses()

    ni = 0
    for cl in classes:  # per ogni classe
        for path in dataset.paths[cl]:  # per ogni path relativo alla classe corrente
            img = sio.imread(path, as_grey=True)  # leggi imamgine
            feat = extract_and_describe(img, kmeans)  # estrai features
            X.append(feat)  # inserisci feature in X
            y.append(classes.index(cl))  # inserisci l'indice della classe corrente in y
            paths.append(path)  # inserisci il path dell'immagine corrente alla lista
            ni += 1
            # rimuovere il commento di seguito per mostrare dei messaggi durante l'esecuzione
            # print "Processing Image {0}/{1}".format(ni,total_number_of_images)

    # Adesso X e y sono due liste, convertiamole in array
    X = np.array(X)
    y = np.array(y)
    return X, y, paths
