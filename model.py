import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp

class model():
    def __init__(self, data):
        self.data = data
        self.label_count = len(self.data.labels)
        self.labels = self.data.labels
        self.V_count = len(self.data.cp_map)
        self.V = self.data.cp_map.values()
        
        self.lmbda = np.zeros( self.label_count* self.V_count)
        print 'number of context predicates: ', self.V_count
        print 'labels', self.labels
        print 'lambda.shape',self.lmbda.shape
        self.is_trained = False
        
                            
    def compute_contidional_prob(self, label_idx, doc, lmbda):
        _p = 0.0
        _numerator = self.compute_sum_features(doc, label_idx, lmbda)
        _numerator = np.exp(_numerator)
        
        _demoninator = 0.0
        for label_ in self.labels:
            temp = self.compute_sum_features(doc, label_, lmbda)
            _demoninator += np.exp(temp)
        _p = _numerator / _demoninator
        return _p

    def compute_sum_features(self, doc, label_idx, lmbda):
        '''
        compute sum_i lambda[i]* f_i (doc, label_idx)
        '''
        _sum = 0.0
        word_ids = doc.cp_ids_counts.keys()
        for word_id in word_ids:
            _sum += 1.0 * lmbda[label_idx * self.V_count + word_id] * doc.cp_ids_counts[word_id] / doc.length
        return _sum
    
    def compute_log_li_grad(self, lmbda):
        print 'Computing log li and grad'
        log_li = -1.0/200 * np.sum(lmbda**2)
        train_docs = self.data.train
        grad = -1.0/100 * lmbda
        
        for doc in train_docs:
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
            temp = np.zeros(self.label_count)
            for word_id in word_ids:
                feature_idx = doc.human_label * self.V_count + word_id
                grad[feature_idx] += 1.0 * doc.cp_ids_counts[word_id] / doc.length
                
                for label_idx in self.labels:
                    temp[label_idx] += 1.0 * lmbda[label_idx * self.V_count + word_id] * doc.cp_ids_counts[word_id] / doc.length
            
            sum_exp_temp = np.sum(np.exp(temp))
            
            for word_id in word_ids:
                for label_idx in self.labels:
                    feature_idx = label_idx * self.V_count + word_id
                    grad[feature_idx] -= temp[label_idx] / sum_exp_temp
            
        return -log_li, np.negative(grad)

    #########################################################
    
        
    def train(self):
        self.lmbda, log_li, dic = fmin_l_bfgs_b(self.compute_log_li_grad, self.lmbda, fprime = None, \
                                                iprint = 99, pgtol=1e-10, factr=1e-30)
        self.is_trained = True
        print 'trained. lambda = ', self.lmbda
    
    def inference(self):
        if not self.is_trained:
            print "please train me to get lambda first"
            return
        for doc in self.data.test:
            c_star = self.labels[0]
            for label_idx in self.labels:
                if self.compute_contidional_prob(label_idx, doc, self.lmbda) > self.compute_contidional_prob(c_star, doc, self.lmbda):
                    c_star = label_idx
            doc.model_label = c_star
    
    def validate(self):
        num_doc_tests = len(self.data.test)
        pre = 0
        for doc in self.data.test:
            if doc.human_label == doc.model_label:
                pre += 1
        
        pre = 100.0 * pre / num_doc_tests
        print 'precision = %0.2f (%%)'%pre
        return pre
            