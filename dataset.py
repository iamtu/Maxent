from document import document
import numpy as np

class dataset():
    def __init__(self, train_file, test_file):
           
        self.train_docs = []
        self.test_docs = []
        self.cp_str_2_int = {} # self.cp_str_2_int['context_predicate_str'] = context_predicate_id in train data
        self.cp_int_2_str = {}
        
        self.labels = [0,1] # C set [0,1]
        self.label_count = 2
        
        self.V = []
        self.V_count = 0
        
        self.V_labels = []

        # read train_docs from file
        print '\t *** train file:', train_file
        with open(train_file, "r") as ins:
            for line in ins:
                if len(line) < 1:
                    continue
                origin_line_str = line
                # for each document
                doc = {}
                [sentence, label_str] = line.strip('\n\t ').split(',')
                sentence = sentence.strip('\t\n ')
                label_id = int(label_str)
                if label_id not in self.V_labels:
                    self.V_labels.append(label_id)


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
                [sentence, label_str] = line.strip('\n\t ').split(',')
                sentence = sentence.strip('\t\n ')
                label_id = int(label_str)
                if label_id not in self.V_labels:
                    self.V_labels.append(label_id)

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


        self.V = self.cp_int_2_str.keys()
        self.V_count = len(self.V)

        print self.V_count
        print self.V_labels
    
    def info(self):
        print 'V', self.V
        print 'train_docs', [doc.origin_str for doc in self.train_docs]
        print 'test_docs', [doc.origin_str for doc in self.test_docs]



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








