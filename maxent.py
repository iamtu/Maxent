import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc 
import sys, os 
sys.path.append(os.getcwd())
from document import document
from utils import get_file_path
import copy

class maxent_model():
    def __init__(self, train_domain, test_domain, is_lifelong, alpha):
        
        self.train_domain = train_domain
        self.test_domain = test_domain
        
        self.IS_LIFELONG = is_lifelong
        self.cue = [] #contain word_str s
        self.DUBPLICATE_CUE = alpha
        self.MAX_CUE_EACH_CLASS = 50
        
        # update me from data
        self.train_docs = []
        self.test_docs = []
        self.cp_str_2_int = {} # self.cp_str_2_int['context_predicate_str'] = context_predicate_id in train data
        self.cp_int_2_str = {}
        
        #FIXME - read me from train_data, test_data
        self.labels = [0,1] # C set [0,1]
        self.label_count = 2
        
        self.V = []
        self.V_count = 0
        # Model parameters
        # lambda shape = matrix CxV
        self.last_iter_lmbda = np.zeros([])
        
        self.update_parameters()            
        
        self.train_iter_count = 0
        self.best_f1_pos_results = {'iter': 0, 'auc' : 0., 'precision' : [0., 0.], 'recall' : [0., 0.], 'fscore' : [0., 0.]}
        self.best_f1_results = {'iter': 0, 'auc' : 0., 'precision' : [0., 0.], 'recall' : [0., 0.], 'fscore' : [0., 0.]}
        self.best_f1_pos_lambda = np.zeros([])
        self.best_f1_lambda = np.zeros([])
            
    def update_parameters(self):
        print 'Updating models parameters...'
        self.cp_str_2_int = {}
        self.cp_int_2_str = {}
        self.train_docs = []
        self.test_docs = []
        
        # read train_docs from file
        print '\ttrain file to read: ', get_file_path(self.train_domain)
        with open(get_file_path(self.train_domain), "r") as ins:
            for line in ins:
                if len(line) < 1:
                    continue
                origin_line_str = line
                # for each document
                doc = {}
                [sentence, label_str] = line.strip('\n\t ').split(',')
                sentence = sentence.strip('\t\n ')
                label_id = int(label_str)
                tokens = sentence.split()
                for token_str in tokens:
                    token_id = self.cp_str_2_int.get(token_str)
                    if token_id == None: # not in context_predicate_map string - id
                        token_id = len(self.cp_str_2_int)
                        self.cp_str_2_int[token_str] = token_id
                        self.cp_int_2_str[token_id] = token_str
                        
                    if token_id not in doc: # check if in doc or not
                        doc[token_id] = 1
                    else :
                        doc[token_id] += 1
                
                aDoc = document(doc, label_id, origin_line_str)
                self.train_docs.append(aDoc)
        
        # read test_docs from file
        print '\ttest file to read: ', get_file_path(self.test_domain)
        with open(get_file_path(self.test_domain), "r") as ins:
            for line in ins:
                if len(line) < 1:
                    continue
                origin_line_str = line
                # for each document
                doc = {}
                [sentence, label_str] = line.strip('\n\t ').split(',')
                sentence = sentence.strip('\t\n ')
                label_id = int(label_str)
                tokens = sentence.split()
                for token_str in tokens:
                    token_id = self.cp_str_2_int.get(token_str)
                    if token_id == None: # not in context_predicate_map string - id
                        continue
                    if token_id not in doc: # check if in doc or not
                        doc[token_id] = 1
                    else :
                        doc[token_id] += 1
                
                aDoc = document(doc, label_id, origin_line_str)
                self.test_docs.append(aDoc)
                
        total_test_count = len(self.test_docs)
        neg_count = 0
        for doc in self.test_docs:
            if doc.human_label == 0:
                neg_count += 1
        print '\t#doc tests: ', total_test_count, '\t# neg_docs: ', neg_count, '\t#pos_docs', total_test_count-neg_count        

        
        
        self.V = self.cp_str_2_int.values()
        self.V_count = len(self.V)
        self.last_iter_lmbda = np.zeros(self.label_count* self.V_count)            
        self.best_f1_pos_lambda = np.zeros(self.label_count* self.V_count)
        self.best_f1_lambda = np.zeros(self.label_count* self.V_count)
        self.train_iter_count = 0
        
    def change_domain(self, new_train_file):
        print '\n\nCHANGING TO DOMAIN ', new_train_file
        self.train_domain = new_train_file
        self.update_parameters()
        
        # update cue for new train_data
        print '\tUpdating cues for documents in new train data...'
        cue_ids = []
        for cue_str in self.cue:
            cue_id = self.cp_str_2_int.get(cue_str)
            if cue_id != None:
                cue_ids.append(cue_id)
         
        for doc in self.train_docs:
            for cue_id in cue_ids:
                if cue_id in doc.cp_ids_counts.keys():
                    doc.cp_ids_counts[cue_id] += self.DUBPLICATE_CUE
            doc.length = sum(doc.cp_ids_counts.values())
        
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
        print '\tCurrent train: ', self.train_domain, '----- test: ', self.test_domain
        self.last_iter_lmbda, log_li, dic = fmin_l_bfgs_b(self.compute_log_li_grad, self.last_iter_lmbda, iprint = 0, 
                                                callback = self._test_while_train)
    
        if self.IS_LIFELONG == True:
            self.update_cue()
    
    def update_cue(self):
        print '\nCalculating cue...'
        cue_ids = []
        
        # cue for only positive 
        lmbda_pos_idx = self.last_iter_lmbda[1 * self.V_count: 1 * self.V_count + self.V_count]
        temp = np.argpartition(-lmbda_pos_idx, self.MAX_CUE_EACH_CLASS)
        max_idxes = temp[:self.MAX_CUE_EACH_CLASS]
        for idx in max_idxes:    
            cue_ids.append(idx)
                 
        cue_strs = []
        for cue_id in cue_ids:
            cue_str = self.cp_int_2_str.get(cue_id)
            if cue_str == None:
                print "ERROR: cue str not found in data.dictionary"
                exit(1)
            cue_strs.append(cue_str)     

        print '\tCues of POSITIVE CLASS for domain ', self.train_domain, ': ', cue_strs
                    
        for cue_str in cue_strs:
            if cue_str not in self.cue:
                self.cue.append(cue_str)
        print '\tUpdated new cues into General Knowledge Base'
        print '\tCues in General Knowledge Base : ', self.cue
                                
    def _inference_doc(self, doc, lmbda):
        '''
            return c_star = argmax_c P(c|d)
        '''
        temp = np.zeros(self.label_count)
        for label_ in self.labels:
            temp[label_] = self.compute_sum_features(doc, label_, lmbda)
        temp = self.softmax(temp)
        return np.argmax(temp)

    def _inference(self, lmbda):
        for doc in self.test_docs:
            doc.model_label = self._inference_doc(doc, lmbda)
    
    def _test(self, lmbda):
        self._inference(lmbda)        
        model_labels = [doc.model_label for doc in self.test_docs]
        human_labels = [doc.human_label for doc in self.test_docs]
        
        prec, recall, _ = precision_recall_curve(human_labels, model_labels)
        auc_metric = auc(recall, prec)
        precision, recall, fscore, support = precision_recall_fscore_support(human_labels, model_labels)
        
        if not self.IS_LIFELONG:
            print 'Iteration ', self.train_iter_count
            print '\tClass \tSupport \tAUC \t\tPRECISION \tRECALL \t\tFSCORE'
            for i in xrange(2):
                print '\t', i, '\t', support[i], '\t\t\t\t', precision[i], '\t', recall[i], '\t', fscore[i], '\n'
            print '\t\t\t\t--------------------------------------------------------------'
            print '\tAverage:', '\t\t',auc_metric, '\t', np.mean(precision), '\t', np.mean(recall), '\t', np.mean(fscore), '\n'
        
        return {'auc': auc_metric, 'precision' : precision, 'recall' : recall, 'fscore' : fscore}
    
    def _test_while_train(self, temp_lambda):
        self.train_iter_count += 1
        res = self._test(temp_lambda)
        if res['fscore'][1] > self.best_f1_pos_results['fscore'][1]:
            self.best_f1_pos_results = copy.deepcopy(res)
            self.best_f1_pos_results['iter'] = self.train_iter_count
            self.best_f1_pos_lambda = np.copy(temp_lambda)
            
        if np.mean(res['fscore']) > np.mean(self.best_f1_results['fscore']):
            self.best_f1_results = copy.deepcopy(res)
            self.best_f1_results['iter'] = self.train_iter_count
            self.best_f1_lambda = np.copy(temp_lambda)
            
    def test(self, lmbda):
        self._test(lmbda)
    
    def print_results(self):
        print '\n*******Results: train_domain ', self.train_domain, ' / test_domain: ', self.test_domain , '******'
        print 'Fscore for positive class : iteration', self.best_f1_pos_results['iter']
        print '\tClass \tSupport \tAUC \t\tPRECISION \tRECALL \t\tFSCORE'
        for i in xrange(2):
            print '\t', i, '\t\t\t\t\t', self.best_f1_pos_results['precision'][i], '\t', self.best_f1_pos_results['recall'][i], '\t', self.best_f1_pos_results['fscore'][i], '\n'
        print '\t\t\t\t--------------------------------------------------------------'
        print '\tAverage:', '\t\t', self.best_f1_pos_results['auc'], '\t', np.mean(self.best_f1_pos_results['precision']), '\t', np.mean(self.best_f1_pos_results['recall']), '\t', np.mean(self.best_f1_pos_results['fscore']), '\n'

        
        
        print 'Fscore for both classes : iteration', self.best_f1_results['iter']
        print '\tClass \tSupport \tAUC \t\tPRECISION \tRECALL \t\tFSCORE'
        for i in xrange(2):
            print '\t', i, '\t\t\t\t\t', self.best_f1_results['precision'][i], '\t', self.best_f1_results['recall'][i], '\t', self.best_f1_results['fscore'][i], '\n'
        print '\t\t\t\t--------------------------------------------------------------'
        print '\tAverage:', '\t\t', self.best_f1_results['auc'], '\t', np.mean(self.best_f1_results['precision']), '\t', np.mean(self.best_f1_results['recall']), '\t', np.mean(self.best_f1_results['fscore']), '\n'
    