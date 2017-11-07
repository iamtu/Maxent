import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc 

class model():
    def __init__(self, data, is_lifelong):
        self.IS_LIFELONG = is_lifelong
        
        self.cue = [] #contain word_str s
        self.DUBPLICATE_CUE = 3
        self.MAX_CUE_EACH_CLASS = 50
        
        self.data = data
        
        self.labels = self.data.LABELS # C set [0,1]
        self.label_count = len(self.labels)
        
        self.V = self.data.cp_str_2_int.values()  # V set [0,1,2....V-1] 
        self.V_count = len(self.V)
        
        # Model parameters
        # lambda shape = matrix CxV
        self.lmbda = np.zeros( self.label_count* self.V_count)            
            
    def change_domain(self, data):
        print 'Change to domain', data.domain
        self.data = data
        # update cue for train_data
        cue_ids = []
        for cue_str in self.cue:
            cue_id = self.data.cp_str_2_int.get(cue_str)
            if cue_id != None:
                cue_ids.append(cue_id)
        
        for doc in self.data.train:
            for cue_id in cue_ids:
                if cue_id in doc.cp_ids_counts.keys():
                    doc.cp_ids_counts[cue_id] += self.DUBPLICATE_CUE
            doc.length = sum(doc.cp_ids_counts.values())
        
        # update self.V, self.V, self.lmbda s.t new data
        self.V = self.data.cp_str_2_int.values()  # V set [0,1,2....V-1] 
        self.V_count = len(self.V)
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
        print "Training max_ent with LBFGS algorithm. Change iprint = 99 to more logs..."
        self.lmbda, log_li, dic = fmin_l_bfgs_b(self.compute_log_li_grad, self.lmbda, iprint = 0)
    
        if self.IS_LIFELONG == True:
            self.update_cue()
    
    def update_cue(self):
        print 'Updating cue...'
        cue_ids = []
        for label_idx in self.labels:
            lmbda_label_idx = self.lmbda[label_idx * self.V_count: label_idx * self.V_count + self.V_count]
            temp = np.argpartition(-lmbda_label_idx, self.MAX_CUE_EACH_CLASS)
            max_idxes = temp[:self.MAX_CUE_EACH_CLASS]
            for idx in max_idxes:    
                cue_ids.append(idx)
        
        cue_strs = []
        for cue_id in cue_ids:
            cue_str = self.data.cp_int_2_str.get(cue_id)
            if cue_str == None:
                print "ERROR: cue str not found in data.dictionary"
                exit(1)
            cue_strs.append(cue_str)     
                    
        for cue_str in cue_strs:
            if cue_str not in self.cue:
                self.cue.append(cue_str)        
    def inference_doc(self, doc):
        '''
            return c_star = argmax_c P(c|d)
        '''
        temp = np.zeros(self.label_count)
        for label_ in self.labels:
            temp[label_] = self.compute_sum_features(doc, label_, self.lmbda)
        temp = self.softmax(temp)
        return np.argmax(temp)

        
    def inference(self):
#         print "Inference test docs"
        for doc in self.data.test:
            doc.model_label = self.inference_doc(doc)
    
    def validate(self):
                
        print 'Result in domain: ', self.data.domain
        model_labels = [doc.model_label for doc in self.data.test]
        human_labels = [doc.human_label for doc in self.data.test]
        
        prec, recall, _ = precision_recall_curve(human_labels, model_labels)
        auc_metric = auc(recall, prec)
#         print 'prec, recall', prec, recall
        print 'AUC: ', auc_metric
        precision, recall, fscore, support = precision_recall_fscore_support(human_labels, model_labels)
        print'PRECISION: ', np.mean(precision)
        print'RECALL: ', np.mean(recall)
        print 'FSCORE: ', np.mean(fscore), "\n"
        
#     def save_model(self):
#         print 'saving model...'
#         # create folder to save RUN_TIME
#         directory = './Data/lifelong/timerun' + str(self.TIME_RUN)
#         if not os.path.isdir(directory):
#             print 'creating directory', directory
#             os.makedirs(directory)
#         sub_dir = directory + '/' + self.data.domain
#         if not os.path.exists(sub_dir):
#             os.makedirs(sub_dir)
#         # train_data
#         train_file = sub_dir + '/train.txt'
#         fout = open(train_file, 'w')
#         for doc in self.data.train:
#             fout.write(doc.origin_str)
#         fout.close()
#         # test_data
#         test_file = sub_dir + '/test.txt'
#         fout = open(test_file, 'w')
#         for doc in self.data.test:
#             fout.write(doc.origin_str)
#         fout.close()
#         # V
#         vocab_file = sub_dir + '/vocab.txt'
#         fout = open(vocab_file, 'w')
#         for cp_str in self.data.cp_str_2_int.keys():
#             cp_id = self.data.cp_str_2_int.get(cp_str)
#             fout.write(str(cp_id) + " " + cp_str + '\n')
#         fout.close()
#         # Labels
#         labels_file = sub_dir + '/labels.txt'
#         fout = open(labels_file, 'w')
#         for label_idx in self.data.LABELS:
#             fout.write(str(label_idx) + '\n')
#         fout.close()
#         # lambda
#         lambda_file = sub_dir + '/lambda.txt'
#         fout = open(lambda_file, 'w')
#         for label_idx in self.data.LABELS:
#             for w in xrange(self.V_count):
#                 fout.write(str(self.lmbda[label_idx * self.V_count + w]) + ' ')
#             fout.write('\n')
#         fout.close()
