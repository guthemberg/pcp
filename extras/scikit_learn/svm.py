import numpy as np
import pylab as pl
import matplotlib.font_manager
from scipy import stats

from sklearn import svm, grid_search, metrics, preprocessing
from sklearn.covariance import EllipticEnvelope

import sys

from numpy import genfromtxt

from time import gmtime, strftime

DEFAULT_CACHE_SIZE=1500


class SVMClassifier:
    """
    generic SVM classifier for binary classification (1/-1)
    with default parameters
    
    """
    
    def __init__(self,training_data):
        """
        it requires a tab-separated training data where the latest
        column has binary outputs 
        """
        my_data = genfromtxt(training_data, delimiter='\t',skip_header=0)
        n_col = my_data.shape[1]
        n_features=n_col-1  #assuming that the latest column
                            #contains the the outputs 
        #preprocessing data
        X = preprocessing.scale(np.hsplit(my_data,[n_features,n_col])[0])
        Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))
        #defining scaling
        self.scaler = preprocessing.Scaler()
        self.scaler.fit(np.hsplit(my_data,[n_features,n_col])[0])
        #define classifier
        self.classifier = svm.SVC(class_weight='auto',cache_size=DEFAULT_CACHE_SIZE)
        self.classifier.fit(X, Y)
        
    def predict(self,X):
        """
        raw predictions from sklearn library
        """
        return self.classifier.predict(X)

class PopularityPredictor(SVMClassifier):
    """
    it extends SVMClassifier for predicting
    if a content is popular or not
    based on an numpy input arrays of features
    """
    
    def predict(self,X):
        """
        predicts the popularity of a content 
        from a numpy array of its features
         
        
        returns 'popular' string if output is equal to 1
        'unpopular' otherwise 
        """
        if (int(self.classifier.predict(X))==1):
            return "popular"
        else:
            return "unpopular"
        
    def print_prediction_to_stdout(self,X):
        """
        performs prediction according to self.predict
        and prints result to standard output
        """
        sys.stdout.write(self.predict(self.scaler.transform(X)))
        sys.stdout.flush()

###functions for testing purposes

def validate(inputs):
    """
    checks which kernel methods is better for SVM
    options: sigmoid and rbf (default method).

    it requires a inputs tab-separated file
    """
    print "running validation"
    my_data = genfromtxt(inputs, delimiter='\t',skip_header=0)
    n_col = my_data.shape[1]
    n_features=n_col-1  #assuming that the latest column
                        #contains the the outputs 
    #processing data without targets
    X = preprocessing.scale(np.hsplit(my_data,[n_features,n_col])[0])
    Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))
    #for further information about parameters, please google sklearn docs
    parameters = {'kernel':('sigmoid', 'rbf'), 'C':[.1,.2,1.0],'cache_size':[500]}
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters,n_jobs=3)
    sys.stdout.write("%s:validating... "%(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    output = clf.fit(X,Y)
    print output
    print "(%s) DONE." % (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    sys.exit(0)
    
def test(training_file, test_file):
    print "testing..."
    sys.stdout.write("%s:training... "%(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    sys.stdout.flush()
    classifier = SVMClassifier(training_file)
    print "(%s) DONE." % (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    #train_classifier(training_file)

    my_data = genfromtxt(test_file, delimiter='\t',skip_header=0)
    n_col = my_data.shape[1]
    n_features=n_col-1  #assuming that the latest column
                        #contains the the outputs 
    #for testing
    X = preprocessing.scale(np.hsplit(my_data,[n_features,n_col])[0])
    Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))    

    predictions = classifier.predict(X)
    fpr, tpr, thresholds = metrics.roc_curve(Y, predictions)
    print fpr, tpr, metrics.auc(fpr, tpr), thresholds
    sys.exit(0)



if __name__ == "__main__":
    training_file=""
    test_file=""
    if (len(sys.argv)>1):
        option = sys.argv[1]
        training_file = sys.argv[2]
        if option=="test":
            test_file = sys.argv[3]
    else:
        training_file = "/tmp/f1.data"

    if option=="validation":
        validate(training_file)
    elif option=="test":
        test(training_file, test_file)
    #my_data = genfromtxt("/tmp/f1.data", delimiter='\t',skip_header=0)
    #for testing
    #X = preprocessing.scale(np.hsplit(my_data,[9,10])[0])
    classifier=PopularityPredictor(training_file)
    classifier.print_prediction_to_stdout(np.array([15.0,14.0,1.0,2.0,1.0,0.0,185.0,0.0,0.0]))
    #classifier.print_prediction_to_stdout(X[6,])
    sys.exit(0)
