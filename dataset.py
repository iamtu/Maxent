import numpy as np
from document import document
from utils import pre_process_doc

class dataset():
    def __init__(self, train_file, test_file):
           
        self.train_docs = []
        self.test_docs = []
        self.cp_str_2_int = {} # self.cp_str_2_int['context_predicate_str'] = context_predicate_id in train data
        self.cp_int_2_str = {}
        self.label_idx_2_str = {}
        self.label_str_2_idx = {}

        # read train_docs from file
        print '\t *** train file:', train_file
        with open(train_file, "r") as ins:
            for line in ins:
                if len(line) < 1:
                    continue
                origin_line_str = line
                # for each document
                doc = {}
                [sentence, label_str] = pre_process_doc(line)
                label_id = self.label_str_2_idx.get(label_str)
                if label_id == None:
                    label_id = len(self.label_str_2_idx)
                    self.label_str_2_idx[label_str] = label_id
                    self.label_idx_2_str[label_id] = label_str

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
        print '\t *** test file:', test_file
        with open(test_file, "r") as ins:
            for line in ins:
                if len(line) < 1:
                    continue
                origin_line_str = line
                # for each document
                doc = {}
                [sentence, label_str] = pre_process_doc(line)
                label_id = self.label_str_2_idx.get(label_str)
                if label_id == None:
                    print 'Warning: sentence:', sentence, 'with label', label_str , ' not found in training dataset'
                    continue 
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
                

        self.V = self.cp_int_2_str.keys()
        self.V_count = len(self.V)

    def info(self):
        print 'Dataset information:'
        print 'Dictionary size:', self.V_count
        print 'Training dataset:'
        count = {}
        for doc in self.train_docs:
            if doc.human_label not in count.keys():
                count[doc.human_label] = 1
            else:
                count[doc.human_label] += 1
        print 'class\t#docs'
        for label_idx in count.keys():
            print self.cp_int_2_str[label_idx], '\t', count[label_idx]
        
        print 'Test dataset:'
        count = {}
        for doc in self.test_docs:
            if doc.human_label not in count.keys():
                count[doc.human_label] = 1
            else:
                count[doc.human_label] += 1
        print 'class\t#docs'
        for label_idx in count.keys():
            print self.cp_int_2_str[label_idx], '\t', count[label_idx]


    def convert_2_numpy(self):
        n_doc_train = len(self.train_docs)
        n_doc_test = len(self.test_docs)
        X_train = np.zeros((n_doc_train, self.V_count))
        Y_train = np.zeros(n_doc_train)
        X_test = np.zeros((n_doc_test, self.V_count))
        Y_test = np.zeros(n_doc_test)
    
        for i in xrange(len(self.train_docs)):
            doc = self.train_docs[i]
            for word_id, cnt in doc.cp_ids_counts.iteritems():
                X_train[i][word_id] = 1.0*cnt/doc.length
            Y_train[i] = doc.human_label
        
        for i in xrange(len(self.test_docs)):
            doc = self.test_docs[i]
            for word_id, cnt in doc.cp_ids_counts.iteritems():
                X_test[i][word_id] = 1.0*cnt/doc.length
            Y_test[i] = doc.human_label

        return X_train, Y_train, X_test, Y_test








