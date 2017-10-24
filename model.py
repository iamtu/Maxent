import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class model():
    def __init__(self, data):
        self.data = data
        self.label_count = len(self.data.labels)
        self.labels = self.data.labels
        self.V_count = len(self.data.cp_map)
        self.num_feafures = self.label_count * self.V_count
        
        self.lmbda = np.zeros(self.num_feafures)
        print 'number of context predicates: ', self.V_count
        print 'labels', self.labels
        self.is_trained = False
        
        
    def map_feature_idx_2_cp_idx_label(self, feature_idx):
        label_idx = feature_idx / self.V_count
        cp_idx = feature_idx % self.V_count
        return (label_idx, cp_idx)
        
    def compute_feature(self, feature_idx, doc, label):
        feature_value = 0.0
        (label_idx, cp_idx) = self.map_feature_idx_2_cp_idx_label(feature_idx)
        if (label_idx == label) and (cp_idx in doc.cp_ids_counts):
            feature_value = 1.0 * doc.cp_ids_counts[cp_idx] / doc.length
        else:
            feature_value = 0.0
            
#         print '\n---compute feature for doc - label'
#         print 'word,label', cp_idx, label_idx 
#         print 'doc', doc
#         print 'label', label
#         print 'feature value', feature_value
        
        return feature_value
        
    def compute_log_li(self, lmbda):
        log_li = 0.0
        train_docs = self.data.train
        
        for doc in train_docs:
            doc_log_li = 0.0
            ep_1 = 0.0
            for i in xrange(len(lmbda)):
                ep_1 += 1.0 * lmbda[i] * self.compute_feature(i, doc, doc.human_label)
            
            ep_2 = 0.0           
            for label_idx in self.labels:
                temp = 0.0
                for i in xrange(len(lmbda)):
                    temp += 1.0 * lmbda[i] * self.compute_feature(i, doc, label_idx)
                ep_2 += np.exp(temp)    
            
            doc_log_li = ep_1 - np.log(ep_2)
            log_li += doc_log_li
        return -log_li
        
    def compute_contidional_prob(self, label_idx, doc, lmbda):
        p = 0.0
        _numerator = 0.0
        for i in xrange(len(lmbda)):
            _numerator += lmbda[i] * self.compute_feature(i, doc, label_idx)
        _numerator = np.exp(_numerator)
        
        _demoninator = 0.0
        for label_ in self.labels:
            temp = 0.0
            for i in xrange(len(lmbda)):
                temp += lmbda[i] * self.compute_feature(i, doc, label_)
            _demoninator += np.exp(temp)
        p = _numerator / _demoninator
        return p
    def compute_grad(self, lmbda):
        grad = np.zeros(len(lmbda))
        for feature_idx in range(len(grad)):
            grad[feature_idx] = 0.0
            for doc in self.data.train:
                ep_1 = self.compute_feature(feature_idx, doc, doc.human_label)
                ep_2 = 0.0
                for label_idx in self.labels:
                    ep_2 += self.compute_feature(feature_idx, doc, label_idx) * self.compute_contidional_prob(label_idx, doc, lmbda)
                grad[feature_idx] += ep_1 - ep_2
        return np.negative(grad)
        
        
    def train(self):
        self.lmbda, log_li, dic = fmin_l_bfgs_b(self.compute_log_li, self.lmbda, self.compute_grad, iprint = 99)
        self.is_trained = True
    
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
            