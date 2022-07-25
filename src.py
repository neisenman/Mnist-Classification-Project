import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats as st

from tensorflow.keras.datasets import mnist

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression




# Pre-processing
def reshaping(X):
    """
    reshapes the matrices inside of X to become vectors
    and scales the values, dividing by 255.
    @param X: input matrices
    @return reshaped and scaled X matrix
    """
    X2 = np.array([np.array([1.0]*784)]*len(X))

    for i in range(len(X)):

        X2[i] = X[i].reshape(784)/255.0
    return X2

def dimReduction(X, dimensions):
    """
    Applies TruncatedSVD to reduce the dimensionality of the vectors in X
    to a chosen dimensionality
    @param X: input matrix
    @dimensions: desired dimensionality of output matrix.
    @return reduced matrix
    """
    truncatedSVD = TruncatedSVD(dimensions)
    return truncatedSVD.fit_transform(reshaping(X))


# Decision Tree
def runTree(X,dimensionality, testSize = .2):
    """
    Runs decision tree and outputs accuracy score
    @param X: input matrix
    @param dimensionality: specifies dimensionality of X matrix, so 
    as to reduce run time
    @param testSize: size of test set
    @return accuracy score of model
    """
    print(dimensionality)
    X = reshaping(mnist.load_data()[0][0])
    X = dimReduction(X,dimensionality)

    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = testSize, shuffle = False, random_state =7)

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    # cmdtree = confusion_matrix(y_test,predictions)

    return accuracy_score(y_test , predictions) 


def plotAccuracyGain(X, maxIndex = 21, testSize = .2):
    """
    Plots accuracy gain for decision tree model as the dimensionality
    increases
    @param X: input matrix
    @param maxIndex: max dimensionality 
    @param testSize: size of test set

    @returns a tuple containing dimensionality and the 
    corresponding accuracy
    """
    dim = []
    accuracy = []
    for i in range(1,maxIndex):
        print(i)
        dim.append(i)
        accuracy.append(runTree(X = X,index = i, testSize = .2))

    dim.append(28)
    accuracy.append(runTree(X = X,index = 28, testSize = .2))

    plt.plot(dim, accuracy)
    plt.legend("Accuracy Gained")
    plt.xlabel("Dimensions")
    plt.ylabel("Accuracy")
    plt.show()
    return (dim, accuracy)


def depthTest(depth = 10, X, y, testSize = .3):
    """
    Tests the optimal tree depth, so as to limit overfitting
    @param X: input matrix
    @param y: results from input
    @param testSize: size of test set
    """
    scores = []
    indices = []
    for i in range(1, depth):
        indices.append(i)
        print(i)

        X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = testSize, shuffle = False, random_state =7)

        dtree = DecisionTreeClassifier(max_depth=i)
        dtree.fit(X_train, y_train)
        predictions = dtree.predict(X_test)

        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))

        print(dtree.score(X_test , y_test))
        scores.append(accuracy_score(y_test, predictions))

def crossVal(X,y,testSize = .2,maxDepth):
    """
    Prints cross validation scores for decision tree model.
    @param X: input matrix
    @param y: results from input
    @param testSize: size of test set
    @param maxDepth: max depth of tree
    """
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = testSize,shuffle = False, random_state =7)
    
    scores = []
    for i in range(3,maxDepth):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        score = cross_val_score(estimator=clf, X=X, y=y, cv=5, n_jobs=4,scoring='accuracy')
        scores.append(score)
        clf.fit(X_train,y_train)
        print(confusion_matrix(y_test, clf.predict(X_test)))
        print(scores)


# Logistic Regression
def runLogistic(X,y,splits = 10,testSize = .2):
    """
    Runs logistic regression model and returns cross validated scores
    @param X: input matrix
    @param y: results from input
    @param splits: number of splits in cross validation
    @param testSize: size of test set
    @return accuracy score for logistic regression
    """
    # create model
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = testSize,shuffle = False, random_state =7)
    model = LogisticRegression()
    model.fit(X_train,y_train)
    print(confusion_matrix(y_test,model.predict(X_test)))
    # evaluate model
    cv = KFold(n_splits=splits, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(scores)
    return print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))



def logistic(X,y,testSize = .2):
    """
    Runs logistic regression model without cross validation
    @param X: input matrix
    @param y: results from input
    @param testSize: size of test set
    @return confusion matrix

    """
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = testSize,shuffle = False, random_state =7)
    model = LogisticRegression()
    model.fit(X_train,y_train)
    return confusion_matrix(y_test,model.predict(X_test))

# KNN with sklearn
def sKNN(X,y,topK):
    """
    Runs cross validation for scicit learn KNN model
    and prints best k value
    @param X: input matrix
    @param y: results from input
    @param topK: max nearest neighbors to consider

    """
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,shuffle = False, random_state =7)

    knn = KNeighborsClassifier()
    k_range = list(range(1, topK))
    param_grid = dict(n_neighbors=k_range)

    # defining parameter range
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)

    # fitting the model for grid search
    grid_search=grid.fit(X_train, y_train)

    print(grid_search.best_params_)


def KNNmetrics(X,y,neighbors):
    """
    Runs k nearest neighbor algorithm, finding accuracy
    @param X: input matrix
    @param y: results from input
    @param neighbors: number of neighbors to consider in knn algorithm
    @return accuracy score of KNN algorithm

    """
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2,random_state=32)

    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    return metrics.accuracy_score(y_test,y_pred)


# Scratch KNN
def distance(x1,x2):
    """
    Finds Euclidean distance between two vectors
    @param x1: first vector to consider 
    @param x2: second vector to consider
    @return distance between two vectors

    """
    x3 = x1 - x2
    return np.dot(x3,x3)**.5


def replace(oldVector, index, newVal):
    """
    Helper method to replace and shift values of the k-closest neighbors
    @param oldVector: old nearest neighbors list
    @param index: the index which we are replacing
    @param newVal: new value to put into the array
    @return updated vector with new value in the proper index
    """
    begin = np.append(oldVector[0:index],newVal)
    return np.concatenate((begin,oldVector[index:len(oldVector)]),axis=0)


def vote(y):
    """
    Method for voting for knn
    @param y: the classification for the k closest neighbors
    @return the mode of the classifications for the k-closest vectors
    """
    return st.mode(y)[0][0]


def knn(index, X, y, k):
    '''
    Runs KNN algorithm for one data point
    @param index: index in input for which to find k-nearest neighbors 
    @param k: number of nearest neighbors we will consider.
    @return classification for the input vector X
    '''
    input = X[index]
    near = np.array([])
    classification = np.array([])
    for i in range(len(X)):

        if len(near) > k:
            near[0:k-1]
            classification[0:k-1]
        if i == index:
            continue

        dist = distance(input, X[i])
        
        b = False
        for j in range(len(near)):
            if near[j] > dist:

                near = replace(oldVector=near,index=j,newVal=dist)
                classification = replace(oldVector = classification, index=j , newVal = y[i])

                b = True
                break
        if b == False & len(near) < k:
            near = np.append(near, dist)
            classification = np.append(classification,y[i])

    return (near[0:k], classification,int(st.mode(classification[1])[0][0]))


def accuracy(k, X, y,length):
    """
    Iterates KNN algorithm finding accuracy of the model.
    @param k: number of nearest neighbors we will consider.
    @param X: input matrix
    @param y: results from input
    @return classification for the input vector X

    """
    correct = 0
    total = 0
    for i in range(length):
        print(i)
        total += 1
        r = random.randint(0, len(X))
        if knn(index = r, X, y, k=12)[2] == y[r]:
            correct += 1
        else:
            print("WRONG!!!")
    
    return str(100 * correct / total) + "%"