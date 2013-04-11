import numpy as np
import sys
from numpy import genfromtxt
from time import gmtime, strftime

#svm here below is my own code
#for classification with this method
#it was used in hermes
import svm

from sklearn import grid_search, metrics
from sklearn.ensemble import GradientBoostingClassifier, \
                        RandomForestClassifier, \
                        RandomForestRegressor, \
                        GradientBoostingRegressor
from sklearn.tree.tree import ExtraTreeClassifier

"""We have to implement here our own
learning to rank method.

We must start from a common classification
method. we are going to start with 
RandomForests. 

Assume a four-class problem:
    - non-popular: 0
    - popular: 1
    - very popular: 2
    - viral
    
For defining ranking in a classification, we
will try to apply a DCG metric, where labels
are pre-processed as follows: new_label=
(2^label)-1

In the near future, we 
might try Gradient Boosting (GBRT) and
also Extremely Randomized Trees

#documentation about boosting methods:
http://scikit-learn.org/0.11/modules/ensemble.html
#a tutorial about DCG:
http://www.cc.gatech.edu/~kzhou30/learndcg/web.html
"""
DEFAULT_CACHE_SIZE=1500
DEFAUL_RANDOM_SEED=1

class ETClassifier:
    """
    generic RandomForest classifier
    with default parameters
    
    """
    
    def __init__(self,training_data):
        """
        it requires a tab-separated training data where the latest
        column has labels or classes
        
        for ranking, we assume that data's labels have already been 
        preprocessed properly. 
         
        """
        my_data = genfromtxt(training_data, delimiter='\t',skip_header=0)
        n_col = my_data.shape[1]
        n_features=n_col-1  #assuming that the latest column
                            #contains labels 
        #preprocessing data
        X = (np.hsplit(my_data,[n_features,n_col])[0])
        Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))
        #define classifier
        classifier = ExtraTreeClassifier(random_state=DEFAUL_RANDOM_SEED)
        self.classifier = classifier.fit(X, Y)

    def predict(self,X):
        """
        raw predictions from sklearn library
        """
        return self.classifier.predict(X)

class GBClassifier:
    """
    generic RandomForest classifier
    with default parameters
    
    """
    
    def __init__(self,training_data,loss='deviance'):
        """
        it requires a tab-separated training data where the latest
        column has labels or classes
        
        for ranking, we assume that data's labels have already been 
        preprocessed properly. 
         
        """
        my_data = genfromtxt(training_data, delimiter='\t',skip_header=0)
        n_col = my_data.shape[1]
        n_features=n_col-1  #assuming that the latest column
                            #contains labels 
        #preprocessing data
        X = (np.hsplit(my_data,[n_features,n_col])[0])
        Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))
        #define classifier
        classifier = GradientBoostingClassifier(loss=loss)
        self.classifier = classifier.fit(X, Y)

    def predict(self,X):
        """
        raw predictions from sklearn library
        """
        return self.classifier.predict(X)

class GBRegressor:
    """
    generic RandomForest classifier
    with default parameters
    
    """
    
    def __init__(self,training_data,loss='ls'):
        """
        it requires a tab-separated training data where the latest
        column has labels or classes
        
        for ranking, we assume that data's labels have already been 
        preprocessed properly. 
         
        """
        my_data = genfromtxt(training_data, delimiter='\t',skip_header=0)
        n_col = my_data.shape[1]
        n_features=n_col-1  #assuming that the latest column
                            #contains labels 
        #preprocessing data
        X = (np.hsplit(my_data,[n_features,n_col])[0])
        Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))
        #define classifier
        classifier = GradientBoostingRegressor(random_state=DEFAUL_RANDOM_SEED,loss=loss)
        self.classifier = classifier.fit(X, Y)

    def predict(self,X):
        """
        raw predictions from sklearn library
        """
        return self.classifier.predict(X)
        
class RFClassifier:
    """
    generic RandomForest classifier
    with default parameters
    
    """
    
    def __init__(self,training_data,n_estimators=10,n_jobs=1):
        """
        it requires a tab-separated training data where the latest
        column has labels or classes
        
        for ranking, we assume that data's labels have already been 
        preprocessed properly. 
         
        """
        my_data = genfromtxt(training_data, delimiter='\t',skip_header=0)
        n_col = my_data.shape[1]
        n_features=n_col-1  #assuming that the latest column
                            #contains labels 
        #preprocessing data
        X = (np.hsplit(my_data,[n_features,n_col])[0])
        Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))
        #define classifier
        classifier = RandomForestClassifier(n_jobs=n_jobs,random_state=DEFAUL_RANDOM_SEED,n_estimators=n_estimators)
        self.classifier = classifier.fit(X, Y)
        
    def predict(self,X):
        """
        raw predictions from sklearn library
        """
        return self.classifier.predict(X)

class RFRegressor:
    """
    generic RandomForest classifier
    with default parameters
    
    """
    
    def __init__(self,training_data,n_estimators=10,n_jobs=1):
        """
        it requires a tab-separated training data where the latest
        column has labels or classes
        
        for ranking, we assume that data's labels have already been 
        preprocessed properly. 
         
        """
        my_data = genfromtxt(training_data, delimiter='\t',skip_header=0)
        n_col = my_data.shape[1]
        n_features=n_col-1  #assuming that the latest column
                            #contains labels 
        #preprocessing data
        X = (np.hsplit(my_data,[n_features,n_col])[0])
        Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))
        #define classifier
        classifier = RandomForestRegressor(n_jobs=n_jobs,random_state=DEFAUL_RANDOM_SEED,n_estimators=n_estimators)
        self.classifier = classifier.fit(X, Y)
        
    def predict(self,X):
        """
        raw predictions from sklearn library
        """
        return self.classifier.predict(X)

class PopularityRankWithRF(RFClassifier):
    """
    it extends RFClassifier for predicting
    popularity for fixing replication degree
    accordingly 
    """
    
    def predict(self,X):
        """
        predicts the popularity of a content 
        from a numpy array of its features
         
        
        returns label, viral (3), very popular
        (2), popular (1), and non-popular (0)
        """
#        predicted_value=self.classifier.predict(X)
#        print "preditected value:%s" % str(predicted_value)
        if (int(self.classifier.predict((X)))==3):
            return "viral"
        elif (int(self.classifier.predict((X)))==2):
            return "very popular"
        elif (int(self.classifier.predict((X)))==1):
            return "popular"
        else:
            return "non-popular"

    def classify(self,X):
        """
        predicts the popularity of a content 
        from a numpy array of its features
         
        
        returns integer with the class of the request  
        """
        return int(self.classifier.predict((X)))

        
    def print_prediction_to_stdout(self,X):
        """
        performs prediction according to self.predict
        and prints result to standard output
        """
        sys.stdout.write(self.predict_label(X))
        sys.stdout.flush()

def compute_DCG(input_array):
    return np.sum((np.power(2,input_array)-1)/(np.log2(np.array(range(len(input_array)))+2)))

def compute_DCG_2(input_array):
    return input_array[0] + np.sum((input_array[1:])/(np.log2(np.array(range(len(input_array)-1))+2)))

def compute_nDCG_2(Y,predictions,filter_index=None):
#    print np.sum(Y)
    #Y_sorted_pre=(np.sort(Y)[:filter_index]) #[::-1] creates a view to 
                              #the sorted array in a 
                              #inverted order
#    Y_sorted = np.copy((np.sort(Y))[::-1])[:filter_index]#Y_sorted_pre[::-1]
    Y_sorted = (((Y)))[:filter_index]#Y_sorted_pre[::-1]
#    print 'primeiro total'
#    print np.sum(Y_sorted)
#    print len(Y_sorted)
    #print np.unique(Y_sorted)
    np.savetxt('/tmp/Y.txt', Y_sorted, fmt='%.5f', delimiter='\n')
#    seuil=len(Y_sorted)/4
#    start=0
#    total=0
#    for i in Y_sorted:
#        total=total+i
#        start=start+1
#        if start==seuil:
#            print total
#            total=0
#            start=0
#    print total
#    
#    print 'segundo total'
#    predictions_sorted=np.copy((np.sort(predictions))[::-1])[:filter_index]
    predictions_sorted=(((predictions)))[:filter_index]
#    print np.sum(predictions_sorted)
#    print len(predictions_sorted)
    #print np.unique(predictions_sorted)
    np.savetxt('/tmp/predictions.txt', predictions_sorted, fmt='%.5f', delimiter='\n')
#    seuil=len(Y_sorted)/4
#    start=0
#    total=0
#    for i in predictions_sorted:
#        total=total+i
#        start=start+1
#        if start==seuil:
#            print total
#            total=0
#            start=0
#    print total
    return compute_DCG_2(predictions_sorted)/compute_DCG_2(Y_sorted)

def compute_nDCG(Y,predictions,filter_index=None):
#    Y_sorted=np.copy((np.sort(Y))[::-1])[:filter_index] #[::-1] creates a view to 
    Y_sorted=(((Y)))[:filter_index] #[::-1] creates a view to 
                              #the sorted array in a 
                              #inverted order
#    print len(Y_sorted)
#    print Y_sorted[0]
#    print Y_sorted[len(Y_sorted)-1]
    predictions_sorted=(((predictions)))[:filter_index]
#    predictions_sorted=np.copy((np.sort(predictions))[::-1])[:filter_index]
#    print len(predictions_sorted)
#    print predictions_sorted[0]
#    print predictions_sorted[len(predictions_sorted)-1]
    return compute_DCG(predictions_sorted)/compute_DCG(Y_sorted)
        
def test(training_file, test_file,method="rf"):
    print "testing..."
    sys.stdout.write("%s:training... "%(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    sys.stdout.flush()
    classifier=None
    if method=="gb":
        classifier = GBClassifier(training_file)
    elif method=="et":
        classifier = ETClassifier(training_file)
    elif method=='svm':
        svm.test(training_file,test_file)
        print "(%s) DONE." % (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        sys.exit(0)
    else:
        classifier = RFClassifier(training_file,100)
    print "(%s) DONE." % (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    #train_classifier(training_file)

    my_data = genfromtxt(test_file, delimiter='\t',skip_header=0)
    n_col = my_data.shape[1]
    n_features=n_col-1  #assuming that the latest column
                        #contains the the outputs 
    #for testing
    X = (np.hsplit(my_data,[n_features,n_col])[0])
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
        print 'nDCG'
        head_list_limit=None
        print compute_nDCG(Y, predictions,head_list_limit)
        print 'nDCG 2'
        print compute_nDCG_2(Y, predictions,head_list_limit)#,108801,28032
        #precision for multi-class (results between 0-1)
        print "precision score: "+str(metrics.precision_score(Y,predictions,None,None,average='weighted'))
        print "full classification report: "
        print metrics.classification_report(Y,predictions)
        print "report for the rarest class: "
        print metrics.classification_report(Y,predictions,labels=[1])

def validate(test_file,method="rf"):
    classifier=None
    param_grid=None
    if method=="gb":
        classifier = GradientBoostingClassifier()
    else:
        classifier = RandomForestClassifier()
        param_grid = {'n_estimators': [10, 50, 100], 'criterion': ['gini','entropy'],\
                      'max_features':['sqrt','log2',None], 'max_depth':[10,100,1000,None],\
                      'min_samples_split':[2,4,6,10], 'min_samples_leaf':[1,2,3,4],\
                      'min_density':[.1,.01,.9], 'bootstrap':[True,False] }
    print "validating..."
    sys.stdout.write("%s:validating... "%(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    sys.stdout.flush()
    clf = grid_search.GridSearchCV(classifier, param_grid)
    my_data = genfromtxt(test_file, delimiter='\t',skip_header=0)
    n_col = my_data.shape[1]
    n_features=n_col-1  #assuming that the latest column
                        #contains the the outputs 
    #for testing
    X = (np.hsplit(my_data,[n_features,n_col])[0])
    Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))    
    print clf.fit(X, Y)
    print "(%s) DONE." % (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    #train_classifier(training_file)

#
#
#    param_grid = [ \
#                  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},\
#                  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],\
#                   'kernel': ['rbf']},\
#                  ]
#    predictions = classifier.predict(X)

def test_regression(training_file, test_file,method="rf"):
    print "testing..."
    sys.stdout.write("%s:training... "%(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    sys.stdout.flush()
    classifier=None
    if method=="gb":
        classifier = GBRegressor(training_file)
    else:
        classifier = RFRegressor(training_file,100)
    print "(%s) DONE." % (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    #train_classifier(training_file)

    my_data = genfromtxt(test_file, delimiter='\t',skip_header=0)
    n_col = my_data.shape[1]
    n_features=n_col-1  #assuming that the latest column
                        #contains the the outputs 
    #for testing
    X = (np.hsplit(my_data,[n_features,n_col])[0])
    Y = np.squeeze(np.asarray(np.hsplit(my_data,[n_features,n_col])[1]))    

    predictions = classifier.predict(X)
    
    print 'mean square error'
    print metrics.mean_squared_error(Y, predictions)
    print 'nDCG'
    head_list_limit=None
    print compute_nDCG(Y, predictions,head_list_limit)
    print 'nDCG 2'
    print compute_nDCG_2(Y, predictions,head_list_limit)#,108801,28032
    for i in range(0,len(predictions)):
        if predictions[i]>=2.5:
            predictions[i]=3.0
        elif predictions[i]>=1.5:
            predictions[i]=2.0
        elif predictions[i]>=.5:
            predictions[i]=1.0
        else:
            predictions[i]=0.0
    print 'precision'
    #precision for multi-class (results between 0-1)
    print "precision score: "+str(metrics.precision_score(Y,predictions,None,None,average='weighted'))
    print "full classification report: "
    print metrics.classification_report(Y,predictions)
#    print "report for the rarest class: "
#    print metrics.classification_report(Y,predictions,labels=[1])
    
#    for i in max_min:
#        print i
        
        
#    for i in range(100,140):
#        print "%s: %s" % (str(Y[i]),str(predictions[i]))
#    #compute classification accurancy 
#    if (np.unique(Y).size==2):
#        #auc and roc for binary classification
#        fpr, tpr, thresholds = metrics.roc_curve(Y, predictions)
#        print "auc/roc report: "
#        print fpr, tpr, metrics.auc(fpr, tpr), thresholds
#        print "full classification report: "
#        print metrics.classification_report(Y,predictions)
#        print "report for the rarest class: "
#        print metrics.classification_report(Y,predictions,labels=[1])
#    else:
#        #precision for multi-class (results between 0-1)
#        print "precision score: "+str(metrics.precision_score(Y,predictions,None,None,average='weighted'))
#        print "full classification report: "
#        print metrics.classification_report(Y,predictions)
#        print "report for the rarest class: "
#        print metrics.classification_report(Y,predictions,labels=[1])


if __name__ == "__main__":
    training_file=""
    test_file=""
    option=""
    method=""
    if (len(sys.argv)>1):
        option = sys.argv[1]
        method = sys.argv[2]
        training_file = sys.argv[3]
        test_file = sys.argv[4]
    else:
        training_file = "/tmp/f1.data"

    if option=="test":
        test(training_file, test_file,method)
        sys.exit(0)
    elif option=="test_regression":
        test_regression(training_file, test_file,method)
        sys.exit(0)
    elif option=="validation":
        validate(test_file,method)
        sys.exit(0)
    sys.exit(0)
