#Cancemi Damiano - W82000075

# ------------------ IMPORTS ------------------
import os.path
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate

from bovw import *
from dataset import Dataset

#sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.preprocessing import Normalizer
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mae_spearman():
    print "Mean Absolute Error (test_set):", mean_squared_error(y_test, predicted_labels)
    print "Spearman (test_set):", spearmanr(y_test, predicted_labels)[0]

# ------------------ BAG OF WORD MODEL ------------------
dataset = Dataset('Dataset')
training_set, test_set = dataset.splitTrainingTest(0.7)
daisy_step = 2
visual_words = 500

print "> ---- SUMMARY ----"
print "Numero totale di immagini nel dataset:", dataset.getLength()
print "Classi:", dataset.getClasses()
print "Dimensione training set:", training_set.getLength()
print "Dimensione test set:", test_set.getLength()


print "\n> ---- EXTRACTING FEATURES ----"
filename = "bovw_features_" + str(daisy_step) + "step.pkl"
if os.path.isfile(os.getenv("HOME")+"/PycharmProjects/SMM/Progetto/"+filename):
    print "\nTrovate features calcolate precedentemente"
    with open(filename) as inp:
        data = cPickle.load(inp)

    X_training = data["X_training"]
    y_training = data["y_training"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    kmeans = data["kmeans"]
else:
    daisy_features = extract_features(training_set, daisy_step)
    print daisy_features.shape, "\n"
    daisy_features = daisy_features.reshape((-1, daisy_features.shape[1]))

    kmeans = KMeans(visual_words)
    kmeans.fit(daisy_features)
    print "\nShape kmeans.cluster_centers_:", kmeans.cluster_centers_.shape
    X_training, y_training, _ = describe_dataset(training_set, kmeans)
    X_test, y_test, _ = describe_dataset(test_set, kmeans)
    norm = Normalizer(norm='l1')
    X_training = norm.transform(X_training)
    X_test = norm.transform(X_test)

    # per scrivere piu' dati, li inseriamo dentro un dizionario
    with open(filename, 'wb') as out:
        cPickle.dump({
            'X_training': X_training,
            'y_training': y_training,
            'X_test': X_test,
            'y_test': y_test,
            'kmeans': kmeans
        }, out)

print "\n> ---- CLASSIFIER ----"
# ------------------ KNN ------------------
#1-nn
knn = KNN(1)
knn.fit(X_training, y_training)
predicted_labels = knn.predict(X_test)
a_1nn = accuracy_score(y_test, predicted_labels)
print "\n1-NN"
print "Accuracy: %0.2f" % a_1nn
mae_spearman()
print confusion_matrix(y_test, predicted_labels)

#3-nn
knn = KNN(3)
knn.fit(X_training, y_training)
predicted_labels = knn.predict(X_test)
a_3nn = accuracy_score(y_test, predicted_labels)
print "\n3-NN"
print "Accuracy: %0.2f" % a_3nn
mae_spearman()
print confusion_matrix(y_test, predicted_labels)

#5-nn
knn = KNN(5)
knn.fit(X_training, y_training)
predicted_labels = knn.predict(X_test)
a_5nn = accuracy_score(y_test, predicted_labels)
print "\n5-NN"
print "Accuracy: %0.2f" % a_5nn
mae_spearman()
print confusion_matrix(y_test, predicted_labels)


# ------------------ REGRESSORE LOGISTICO ONE-VS-REST ------------------
pca = PCA()
pca.fit(X_training)
X_training_pca = pca.transform(X_training)
X_test_pca = pca.transform(X_test)

#per capire in che modo sono stati trasformati i dati, plotto in uno spazio 2D i campioni presenti in X_training e X_training_pca selezionando solo le prime due componenti
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title("Original data")
plt.plot(X_training[:,0], X_training[:,1],'bx')
plt.subplot(122)
plt.title("Normalized data")
plt.plot(X_training_pca[:,0], X_training_pca[:,1],'rx')
plt.show()

lr = LogisticRegression()
lr.fit(X_training_pca, y_training)

#plots
theta_0 = lr.intercept_
theta_1_n = lr.coef_

#2d
xs = X_training_pca[:,0]
ys = X_training_pca[:,1]
plt.figure()
plt.title('one-vs-rest')
plt.plot(xs[y_training==0], ys[y_training==0],'or')
plt.plot(xs[y_training==1], ys[y_training==1],'xb')
theta_1 = theta_1_n[0][0]
theta_2 = theta_1_n[0][1]
x = np.array([-0.6,0.8])
y = -(theta_0+theta_1*x)/(theta_2)
plt.plot(x,y)
plt.xlim(x)
plt.show()

#3d
plt.figure()
plt.title('one-vs-rest')
plt.subplot(111, projection='3d')
zs = X_training_pca[:,2]
plt.plot(xs[y_training==0], ys[y_training==0], zs[y_training==0],'ro')
plt.plot(xs[y_training==1], ys[y_training==1], zs[y_training==1],'bx')
theta_3 = theta_1_n[0][2]
x = np.arange(-0.6,0.8)
y = np.arange(-0.6,0.8)
x,y = np.meshgrid(x,y)
z = -(theta_0+theta_1*x+theta_2*y)/theta_3
plt.gca().plot_surface(x,y,z,shade=False,color='y')
plt.show()

predicted_labels = lr.predict(X_test_pca)
a_logisticRegression_onevsrest = accuracy_score(y_test, predicted_labels)
print "\nLogistic Regressor (one-vs-rest)"
print "Accuracy: %0.2f" % a_logisticRegression_onevsrest
print confusion_matrix(y_test, predicted_labels)
mae_spearman()


# ------------------ REGRESSORE LOGISTICO MULTINOMIAL ------------------
pca = PCA()
pca.fit(X_training)
X_training_pca = pca.transform(X_training)
X_test_pca = pca.transform(X_test)
lr = LogisticRegression(solver="lbfgs", multi_class="multinomial")
lr.fit(X_training_pca, y_training)
predicted_labels = lr.predict(X_test_pca)
a_logisticRegression_multinomial = accuracy_score(y_test, predicted_labels)
print "\nLogistic Regressor (multinomial)"
print "Accuracy: %0.2f" % a_logisticRegression_multinomial
print confusion_matrix(y_test, predicted_labels)
mae_spearman()

# ------------------ GAUSSIAN NAIVE BAYES ------------------
gnb = GaussianNB()
gnb.fit(X_training, y_training)
predicted_labels = gnb.predict(X_test)
a_naiveBayes = accuracy_score(y_test, predicted_labels)
print "\nGaussian Naive Bayes:"
print "Accuracy: %0.2f" % a_naiveBayes
print confusion_matrix(y_test, predicted_labels)
mae_spearman()

# ------------------ MULTINOMIAL NAIVE BAYES ------------------
nb = NB()
nb.fit(X_training, y_training)
predicted_labels = nb.predict(X_test)
a_MnaiveBayes = accuracy_score(y_test, predicted_labels)
print "\nMultinomial Naive Bayes:"
print "Accuracy: %0.2f" % a_naiveBayes
print confusion_matrix(y_test, predicted_labels)
mae_spearman()


print "\n"
print tabulate(
    [
        ['1-nn', a_1nn],
        ['3-nn', a_3nn],
        ['5-nn', a_5nn],
        ['Logistic regression (one-vs-rest)', a_logisticRegression_onevsrest],
        ['Logistic regression (multinomial)', a_logisticRegression_multinomial],
        ['Gaussian Naive Bayes', a_naiveBayes],
        ['Multinomial Naive Bayes', a_MnaiveBayes]
    ],
    headers=['Classificatore', 'Accuracy score']
)