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
        
    def scale(self,X):
        return self.scaler.transform(X)
        
    def predict(self,X):
        """
        raw predictions from sklearn library
        it assumes that inputs have already been 
        set to the correct scale
        """
        return self.classifier.predict(X)

class LinearSVMClassifier:
    """
    generic Linear SVM classifier for binary classification
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
        self.classifier = svm.LinearSVC(class_weight='auto',C=1.0)
        self.classifier.fit(X, Y)
        
    def scale(self,X):
        return self.scaler.transform(X)
        
    def predict(self,X):
        """
        raw predictions from sklearn library
        it assumes that inputs have already been 
        set to the correct scale
        """
        return self.classifier.predict(X)

class OneClassSVMClassifier:
    """
    generic SVM classifier for binary classification (1/-1)
    with default parameters
    
    """
    
    def __init__(self,training_data, outliers_proportion,base_nu=0.95,min_nu=0.05):
        """
        it requires a tab-separated training data containing 
        common (unpopular) data
        
        the classifier is then ready to identify outliers   
        """
        my_data = genfromtxt(training_data, delimiter='\t',skip_header=0)

        #preprocessing data
        X = preprocessing.scale(my_data)

        #defining scaling
        self.scaler = preprocessing.Scaler()
        self.scaler.fit(my_data)

        #define classifier
        self.classifier = svm.OneClassSVM(nu=((base_nu*outliers_proportion)+min_nu), kernel="rbf", gamma=0.1, cache_size=DEFAULT_CACHE_SIZE)
        self.classifier.fit(X)
        
    def scale(self,X):
        return self.scaler.transform(X)
        
    def predict(self,X):
        """
        raw predictions from sklearn library
        it assumes that inputs have already been 
        set to the correct scale
        """
        return self.classifier.predict(X)

class PopularityPredictorOneClassSVM(OneClassSVMClassifier):
    """
    it extends SVMClassifier for predicting
    if a content is popular or not
    based on an numpy input arrays of features
    """

    def __init__(self,training_data, outliers_proportion=0.01):
        OneClassSVMClassifier.__init__(self, training_data, outliers_proportion)
    
    def predict(self,X):
        """
        predicts the popularity of a content 
        from a numpy array of its features
         
        
        returns 'popular' string if output is equal to 1
        'unpopular' otherwise 
        """
        if (int(self.classifier.predict(self.scaler.transform(X)))==-1):
            return "popular"
        else:
            return "unpopular"

    def classify(self,X):
        """
        predicts the popularity of a content 
        from a numpy array of its features
         
        
        returns integer with the class of the request  
        """
        return int(self.classifier.predict(self.scaler.transform(X)))

        
    def print_prediction_to_stdout(self,X):
        """
        performs prediction according to self.predict
        and prints result to standard output
        """
        sys.stdout.write(self.predict(X))
        sys.stdout.flush()
        
        
class ReplicationPredictorSVM(SVMClassifier):
    """
    it extends SVMClassifier for predicting
    if a content needs more replicas
    based on an numpy input arrays of features
    """
    
    def predict(self,X):
        """
        predicts the popularity of a content 
        from a numpy array of its features
         
        
        returns 'popular' string if output is equal to 1
        'unpopular' otherwise 
        """
        if (int(self.classifier.predict(self.scaler.transform(X)))==1):
            return "increase"
        elif (int(self.classifier.predict(self.scaler.transform(X)))==0):
            return "keep"
        else:
            return "decrease"

    def classify(self,X):
        """
        predicts the popularity of a content 
        from a numpy array of its features
         
        
        returns integer with the class of the request  
        """
        return int(self.classifier.predict(self.scaler.transform(X)))

        
    def print_prediction_to_stdout(self,X):
        """
        performs prediction according to self.predict
        and prints result to standard output
        """
        sys.stdout.write(self.predict(X))
        sys.stdout.flush()


##regression

class LinearSVMRegression:
    """
    generic SVM regression 
    with default parameters
    
    """
    
    def __init__(self,training_data):
        """
        it requires a tab-separated training data where the latest
        column has prediction target 
        """
        my_data = genfromtxt(training_data, delimiter='\t',skip_header=0)
        n_col = my_data.shape[1]
        self.n_features=n_col-1  #assuming that the latest column
                            #contains the the outputs 
        #pre-processing data
        X = preprocessing.scale(np.hsplit(my_data,[self.n_features,n_col])[0])
        Y = np.squeeze(np.asarray(np.hsplit(my_data,[self.n_features,n_col])[1]))
        #defining scaling
        self.scaler = preprocessing.Scaler()
        self.scaler.fit(np.hsplit(my_data,[self.n_features,n_col])[0])
        #define classifier
        self.classifier = svm.SVR(kernel='linear', C=1e3, cache_size=DEFAULT_CACHE_SIZE)
        #self.classifier = svm.SVR(kernel='rbf', C=1e3, gamma=0.1, cache_size=DEFAULT_CACHE_SIZE)
        self.classifier.fit(X, Y)
        
    def scale(self,X):
        return self.scaler.transform(X)
        
    def predict(self,X):
        """
        raw predictions from sklearn library
        it assumes that inputs have already been 
        set to the correct scale
        """
        return self.classifier.predict(X)

class ReplicationPredictorLinearSVM(LinearSVMRegression):
    """
    it extends LinearSVMClassifier for predicting
    if a content needs more replicas
    based on an numpy input arrays of features
    """
    
    def predict(self,X):
        """
        predicts the replication degree of a content 
        from a numpy array of its features
         
        
        returns number of replicas 
        """
        if(X.shape[0]>self.n_features):
            """
            there is replica information as the latest column
            """
            inputs=X[:self.n_features]
            target=X[(X.shape[0]-1):][0]
            prediction=int(round(self.classifier.predict(self.scaler.transform(inputs))))
            if (prediction>target):
                return "increase"
            elif (prediction==target):
                return "keep"
            else:
                return "decrease"
        else:
            return int(round(self.classifier.predict(self.scaler.transform(X)))) 
        
    def print_prediction_to_stdout(self,X):
        """
        performs prediction according to self.predict
        and prints result to standard output
        """
        sys.stdout.write(self.predict(X))
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

def validate_one_class(normal,outliers):
    """
    checks which kernel methods is better for SVM
    options: sigmoid and rbf (default method).

    it requires a inputs tab-separated file
    """
    print "running a validation"
    
    #classifier
    
    train_data = genfromtxt(normal, delimiter='\t',skip_header=0)
    test_data = genfromtxt(outliers, delimiter='\t',skip_header=0)

    outliers_proportion = float(test_data.shape[0])/(float(train_data.shape[0])+float(test_data.shape[0]))
    outliers_proportion=0.01

    clf = OneClassSVMClassifier(normal,outliers_proportion,0.95,0.05)
    #processing data without targets
    X_train = clf.scale(train_data)
    X_test = clf.scale(test_data)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    n_error_train = y_pred_train[y_pred_train == 1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
        
    print "training: ", 100.0*(float(n_error_train)/float(X_train.shape[0])), "testing (test/outliers): ",100.0*(float(n_error_test)/float(X_test.shape[0]))

    sys.exit(0)

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
    
    #compute classification accurancy 
    if (np.unique(Y).size==2):
        #auc and roc for binary classification
        fpr, tpr, thresholds = metrics.roc_curve(Y, predictions)
        print "auc/roc report: "
        print fpr, tpr, metrics.auc(fpr, tpr), thresholds
        print "full classification report: "
        print metrics.classification_report(Y,predictions)
        print "report for the rarest class: "
        print metrics.classification_report(Y,predictions,labels=[1])
    else:
        #precision for multi-class (results between 0-1)
        print "precision score: "+str(metrics.precision_score(Y,predictions,None,None,average='weighted'))
        print "full classification report: "
        print metrics.classification_report(Y,predictions)
        print "report for the rarest class: "
        print metrics.classification_report(Y,predictions,labels=[1])
    sys.exit(0)
    
def test_linear(training_file, test_file):
    print "testing..."
    sys.stdout.write("%s:training... "%(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    sys.stdout.flush()
    classifier = LinearSVMClassifier(training_file)
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
    
    #compute classification accurancy 
    if (np.unique(Y).size==2):
        #auc and roc for binary classification
        fpr, tpr, thresholds = metrics.roc_curve(Y, predictions)
        print "auc/roc report: "
        print fpr, tpr, metrics.auc(fpr, tpr), thresholds
        print "full classification report: "
        print metrics.classification_report(Y,predictions)
        print "report for the rarest class: "
        print metrics.classification_report(Y,predictions,labels=[1])
    else:
        #precision for multi-class (results between 0-1)
        print "precision score: "+str(metrics.precision_score(Y,predictions,None,None,average='weighted'))
        print "full classification report: "
        print metrics.classification_report(Y,predictions)
        print "report for the rarest class: "
        print metrics.classification_report(Y,predictions,labels=[1])
    sys.exit(0)
        

def get_regression_absolute_error_int(Y,predictions):
    total=0
    diff=0
    for i in range(0,predictions.shape[0]):
        total+=Y[i]
        diff+=abs(int(Y[i])-int(round(predictions[i])))
    return float(diff)/float(total)
                  


if __name__ == "__main__":
    training_file=""
    test_file=""
    option=""
    if (len(sys.argv)>1):
        option = sys.argv[1]
        training_file = sys.argv[2]
        test_file = sys.argv[3]
    else:
        training_file = "/tmp/f1.data"

    if option=="validation":
        validate(training_file)
    elif option=="test":
        test(training_file, test_file)
    elif option=="once":
        validate_one_class(training_file, test_file)
    elif option=="test_linear":
        test_linear(training_file, test_file)
    #my_data = genfromtxt("/tmp/f1.data", delimiter='\t',skip_header=0)
    #for testing
    #X = preprocessing.scale(np.hsplit(my_data,[9,10])[0])
    classifier=ReplicationPredictorSVM(training_file)
    classifier.print_prediction_to_stdout(np.array([15.0,14.0,1.0,2.0,1.0,0.0,185.0,0.0,0.0]))
    #classifier.print_prediction_to_stdout(X[6,])
    sys.exit(0)
