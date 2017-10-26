import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp

class model():
    def __init__(self, data):
        self.data = data
        
        self.labels = self.data.labels # C set [0,1,2...]
        self.label_count = len(self.labels)
        
        self.V = self.data.cp_map.values()  # V set [0,1,2....V-1] 
        self.V_count = len(self.V)
        
        # Model parameters
        self.lmbda = np.zeros( self.label_count* self.V_count)

    def softmax(self,x):
        '''
        input x:ndarray
        output e^x /sum(e^x) (trick to avoid overflow)
        '''
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # REMOVEME                            
    def compute_contidional_prob(self, label_idx, doc, lmbda):
        '''
        compute P(c|d) = exp(sum_i lambda_i * f_i(c,d)) / sum_c exp(sum_i lambda_i * f_i(c,d))
                                    i = (w,c) for w \in V, c \in C
        '''
        temp = np.zeros(self.label_count)
        for label_ in self.labels:
            temp[label_] = self.compute_sum_features(doc, label_, lmbda)
        temp = self.softmax(temp)
        return temp[label_idx]

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
        '''
        compute log_li and its gradient
        log_li = sum_d (log P(c(d)|d) with c(d) is human label of document d
        grad_i = grad(lambda_i) = sum_d ( f_i(c(d),d) + (sum_c f_i(c,d)*exp(sum_i lambda_i * f_i(c,d))) / (sum_c exp(sum_i lambda_i * f_i(c,d))) )
        NOTE: 
            *) log_li is computed directly follow the above formular
            *) grad:
                for each document:
                    update grad_i accumulating in this document
        '''

        log_li = 0.0
        grad = np.zeros(lmbda.shape)

        for doc in self.data.train:
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
                grad[feature_idx] += 1.0 * doc.cp_ids_counts[word_id] / doc.length
            
            #NOTE - This is the same in computing doc_log_li, but remained here for clear
            temp = np.zeros(self.label_count)                
            for label_idx in self.labels:
                temp[label_idx] = self.compute_sum_features(doc, label_idx, lmbda)
            temp = self.softmax(temp)            
            
            for word_id in word_ids:
                for label_idx in self.labels:
                    feature_idx = label_idx * self.V_count + word_id
                    grad[feature_idx] -= 1.0 * doc.cp_ids_counts[word_id] / doc.length * temp[label_idx]
            
        return -log_li, np.negative(grad) #negate it because fmin_l_bfgs_b is minimization function
            
    def train(self):
        '''
        Using fmin_l_bfgs_b to maxmimum log likelihood
        NOTE: fmin_l_bfgs_b returns 3 values
        '''
        self.lmbda, log_li, dic = fmin_l_bfgs_b(self.compute_log_li_grad, self.lmbda, iprint = 99)
    
    def doc_inference(self, doc):
        '''
            return c_star = argmax_c P(c|d)
        '''
        temp = np.zeros(self.label_count)
        for label_ in self.labels:
            temp[label_] = self.compute_sum_features(doc, label_, self.lmbda)
        temp = self.softmax(temp)
        return np.argmax(temp)

        
    def inference(self):
        for doc in self.data.test:
            doc.model_label = self.doc_inference(doc)
    
    def validate(self):
        num_doc_tests = len(self.data.test)
        print 'number of doc test', num_doc_tests
        pre = 0
        for doc in self.data.test:
            if doc.human_label == doc.model_label:
                pre += 1
        
        pre = 100.0 * pre / num_doc_tests
        print 'precision = %0.2f (%%)'%pre
        return pre
            