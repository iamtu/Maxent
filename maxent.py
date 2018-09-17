import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import logsumexp
import sys
sys.path.insert(0, '.')

from dataset import dataset
from utils import compute_precision_recall
import copy

class maxent_model():
    def __init__(self, dataset):

        self.dataset = dataset

        self.train_docs = dataset.train_docs
        self.test_docs = dataset.test_docs
        self.cp_str_2_int = dataset.cp_str_2_int # self.cp_str_2_int['context_predicate_str'] = context_predicate_id in train data
        self.cp_int_2_str = dataset.cp_int_2_str
        self.label_str_2_idx = dataset.label_str_2_idx
        self.label_idx_2_str = dataset.label_idx_2_str

        self.labels = self.label_idx_2_str.keys()
        self.label_count = len(self.labels)
        
        self.V = dataset.V
        self.V_count = len(self.V)

        # Model parameters
        # lambda shape = matrix CxV
        self.lmbda = np.zeros((self.label_count, self.V_count))
        self.train_iter_count = 0
    
    def run(self):
        self.train()
        self.test(self.lmbda)
       
    def softmax(self,x):
        '''
        input x:ndarray
        output e^x /sum(e^x) (trick to avoid overflow)
        '''
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def compute_doc_feature(self, doc, word_id):
        '''
        compute f_i(c,d) = f_(w,c') (c,d) = N(w,d)/N(d) if c == c' or 0 if c!= c'
        '''
        return 1.0 * doc.cp_ids_counts[word_id] / doc.length
    
    def compute_sum_features(self, doc, label_idx, lmbda):
        '''
        compute sum_i lambda[i]* f_i (doc, label_idx)
        '''
        _sum = 0.0
        word_ids = doc.cp_ids_counts.keys()
        for word_id in word_ids:
            _sum += lmbda[label_idx*self.V_count + word_id] * self.compute_doc_feature(doc, word_id)
        return _sum
    
    def compute_log_li_grad(self, lmbda):        
        '''
        compute log_li and its gradient
        log_li = sum_d (log P(c(d)|d) - C*sum(lmbda**2)  //with c(d) is human label of document d, C*sum(lmbda**2) is the regulation term 
        grad_i = grad(lambda_i) = sum_d ( f_i(c(d),d) + (sum_c f_i(c,d)*exp(sum_i lambda_i * f_i(c,d))) / (sum_c exp(sum_i lambda_i * f_i(c,d))) )
        NOTE: 
            *) log_li is computed directly follow the above formular
            *) grad:
                for each document:
                    update grad_i accumulating in this document
        '''

        #log_li = 0.0
        #grad = np.zeros(lmbda.shape)
        log_li = -np.sum(lmbda**2)
        grad = -2.0 * lmbda


        for doc in self.train_docs:
            ########### compute log_li #############
            doc_log_li = 0.0
            ep_1 = self.compute_sum_features(doc, doc.human_label, lmbda)
            
            temp = np.zeros(self.label_count)           
            for label_idx in self.labels:
                temp[label_idx] = self.compute_sum_features(doc, label_idx, lmbda)                
            
            doc_log_li = ep_1 - logsumexp(temp)
            log_li += doc_log_li
            
            ###### Update feature_idx with word_id in docs ##########
            word_ids = doc.cp_ids_counts.keys()
            # update feature idx with d.humanlabel
            for word_id in word_ids:
                feature_idx = doc.human_label * self.V_count + word_id
                grad[feature_idx] += self.compute_doc_feature(doc, word_id)
            
            #NOTE - This is the same in computing doc_log_li, but remained here for clear
            temp = np.zeros(self.label_count)                
            for label_idx in self.labels:
                temp[label_idx] = self.compute_sum_features(doc, label_idx, lmbda)
            temp = self.softmax(temp)            
            
            for word_id in word_ids:
                for label_idx in self.labels:
                    feature_idx = label_idx * self.V_count + word_id
                    grad[feature_idx] -= self.compute_doc_feature(doc, word_id) * temp[label_idx]
            
        return -log_li, np.negative(grad) #negate it because fmin_l_bfgs_b is minimization function
            
    def train(self):
        '''
        Using fmin_l_bfgs_b to maxmimum log likelihood
        NOTE: fmin_l_bfgs_b returns 3 values
        '''
        print "\nTraining max_ent with LBFGS algorithm. Change iprint = 99 to more logs..."
        self.lmbda, log_li, dic = fmin_l_bfgs_b(self.compute_log_li_grad, self.lmbda, iprint = 1) 
        '''callback = self._test_while_train'''
    

    def _inference_doc(self, doc, lmbda):
        '''
            return c_star = argmax_c P(c|d)
        '''
        temp = self._score_doc(doc, lmbda)
        label_idx = np.argmax(temp)
        return self.label_idx_2_str[label_idx]
    
    def _score_doc(self, doc, lmbda):
        temp = np.zeros(self.label_count)
        for label_ in self.labels:
            temp[label_] = self.compute_sum_features(doc, label_, lmbda)
        temp = self.softmax(temp)
        result = {}
        for i in range(len(temp)):
            result[self.label_idx_2_str[i]] = temp[i]
        return result


    def _inference(self, lmbda):
        for doc in self.test_docs:
            doc.model_label = self._inference_doc(doc, lmbda)
    
    def _test(self, lmbda):
        self._inference(lmbda)        
        model_labels = [int(doc.model_label) for doc in self.test_docs]
        human_labels = [int(doc.human_label) for doc in self.test_docs]
        
        (pre_1, rec_1, f1_1),(pre_0, rec_0, f1_0) = compute_precision_recall(human_labels, model_labels)
        '''
        print 'Iteration ', self.train_iter_count
        print '\tClass \tPRECISION \tRECALL \t\tFSCORE'
        print '\t', 1, '\t', pre_1, '\t', rec_1, '\t', f1_1, '\n'
        print '\t', 0, '\t', pre_0, '\t', rec_0, '\t', f1_0, '\n'
        '''
        return (pre_1, rec_1, f1_1),(pre_0, rec_0, f1_0) 
    
    def _test_while_train(self, temp_lambda):
        self.train_iter_count += 1
        res = self._test(temp_lambda)
           
    def test(self, lmbda):
        self._test(lmbda)
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'USAGE: python maxent.py train_file test_file'
        exit(1)

    dataset = dataset(sys.argv[1], sys.argv[2])
    maxent = maxent_model(dataset)
    maxent.train()
    
    print 'lambda', maxent.lmbda

    print "score test dataset"
    print maxent._score_doc(dataset.test_docs[0], maxent.lmbda)
    print maxent._score_doc(dataset.test_docs[1], maxent.lmbda)



