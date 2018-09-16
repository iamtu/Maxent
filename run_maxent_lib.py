import nltk
import sys
sys.path.insert(0, '.')
from dataset import dataset 
from sklearn import linear_model
from utils import compute_precision_recall

def list_to_dict(words_list):
    return dict([(word, True) for word in words_list])


def get_data_from_file(filename):
    data  = []
    with open(filename, 'r') as ins:
        for line in ins:
            [sentence, label_str] = line.strip('\n\t ').split(',')
            sentence = sentence.strip('\t\n ')
            words = sentence.split()
            data.append((words, label_str))
    return data

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'USAGE: python test_maxent_lib.py train_file test_file'
        exit(1)
    dataset = dataset(sys.argv[1], sys.argv[2])
    dataset.info()

    X_train, Y_train, X_test, Y_test = dataset.convert_2_numpy()
    
    model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100,verbose=2)
    
    print 'X_train', X_train
    print 'Y_train', Y_train
    model.fit(X_train, Y_train)
    print 'model.lambda = ', model.coef_
    
    print 'X_Test', X_test
    print 'Y_test', Y_test
    print  model.predict_proba(X_test)
    print model.predict_log_proba(X_test)
    
    
    
    
    
    '''
    (pre_1, rec_1, f1_1),(pre_0, rec_0, f1_0) = compute_precision_recall(Y_pred, Y_test)
    print pre_1, rec_1, f1_1
    print pre_0, rec_0, f1_0
    '''
