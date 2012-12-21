import numpy as np
import pylab as pl
import matplotlib.font_manager
from scipy import stats

from sklearn import svm, grid_search, metrics, preprocessing
from sklearn.covariance import EllipticEnvelope

import sys

from numpy import genfromtxt

from time import gmtime, strftime
n_features=7

test=""
if (len(sys.argv)>1):
    option = sys.argv[1]
    inputs = sys.argv[2]
    if option=="test":
        test = sys.argv[3]
else:
    inputs = "/tmp/f1.data"

def train_classifier(data):
    my_data = genfromtxt(data, delimiter='\t',skip_header=0)
    #processing data without targets
    X = preprocessing.scale(np.hsplit(my_data,[n_features,10])[0])
    Y = np.squeeze(np.asarray(np.hsplit(my_data,[9,10])[1]))
    classifier = svm.SVC(class_weight='auto',cache_size=1500)
    sys.stdout.write("%s:training... "%(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    sys.stdout.flush()
    classifier.fit(X, Y)
    print "(%s) DONE." % (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    return classifier

if option=="validation":
    print "running validation"
    my_data = genfromtxt(inputs, delimiter='\t',skip_header=0)
    #processing data without targets
    X = preprocessing.scale(np.hsplit(my_data,[n_features,10])[0])
    Y = np.squeeze(np.asarray(np.hsplit(my_data,[9,10])[1]))
    print X.shape
    print Y.shape
    parameters = {'kernel':('sigmoid', 'rbf'), 'C':[.1,.2,1.0],'cache_size':[500]}
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters,n_jobs=3)
    sys.stdout.write("%s:validating... "%(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    output = clf.fit(X,Y)
    print output
    print "(%s) DONE." % (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    sys.exit(0)
elif option=="test":
    print "testing..."
    my_data = genfromtxt(test, delimiter='\t',skip_header=0)
    #for testing
    X = preprocessing.scale(np.hsplit(my_data,[n_features,10])[0])
    Y = np.squeeze(np.asarray(np.hsplit(my_data,[9,10])[1]))    
    classifier = train_classifier(inputs)
    predictions = classifier.predict(X)
    fpr, tpr, thresholds = metrics.roc_curve(Y, predictions)
    print fpr, tpr, metrics.auc(fpr, tpr), thresholds
    sys.exit(0)

class SVMClassifier:
    """
    performs svm classification with default parameters
    """
    
    def __init__(self,training_data):
        my_data = genfromtxt(training_data, delimiter='\t',skip_header=0)
        #preprocessing data
        X = preprocessing.scale(np.hsplit(my_data,[n_features,10])[0])
        Y = np.squeeze(np.asarray(np.hsplit(my_data,[9,10])[1]))
        #defining scaling
        self.scaler = preprocessing.Scaler()
        self.scaler.fit(np.hsplit(my_data,[n_features,10])[0])
        #define classifier
        self.classifier = svm.SVC(class_weight='auto',cache_size=1500)
        self.classifier.fit(X, Y)
        
    def predict(self,X):
        if (int(self.classifier.predict(X))==1):
            return "popular"
        else:
            return "unpopular"
    def print_prediction_to_stdout(self,X):
        sys.stdout.write(self.predict(self.scaler.transform(X)))
        sys.stdout.flush()
        
#my_data = genfromtxt("/tmp/f1.data", delimiter='\t',skip_header=0)
#for testing
#X = preprocessing.scale(np.hsplit(my_data,[9,10])[0])
classifier=SVMClassifier(inputs)
#classifier.print_prediction_to_stdout(np.array([59.0,2.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]))
#classifier.print_prediction_to_stdout(X[6,])
