import os,sys
sys.path.append(os.getcwd())
from document import document
'''
Read file data with line format:
sentence_1,label
sentence_2,label
'''
DOMAINS = ['electronics', 'hotel', 'suggForum', 'SuggHashtagTweets', 'TravelAdviceRetagged']


class data():
        
    def __init__(self, train_domain, test_domain, fold):
        if fold not in [0, 1,2,3,4,5]:
            print "ERROR with fold name ", fold
            exit(1)
    
        if train_domain not in DOMAINS:
            print "ERROR with train_domain name ,not found in DB", train_domain
            exit(1)
        
        if test_domain not in DOMAINS:
            print "ERROR with train_domain name ,not found in DB", test_domain
            exit(1)

    
        self.train_domain = train_domain
        self.test_domain = test_domain
        
        self.fold = str(fold)
        self.train = [] # list of documents
        self.test = []
        
        # load vocab
        self.cp_str_2_int = {} # self.cp_str_2_int['context_predicate_str'] = context_predicate_id in train data
        self.cp_int_2_str = {}
        self.LABELS = [0,1]
        
        train_filename = './Data/FOLDS/TIMERUN' + str(fold) + '/' + self.train_domain + '/train.txt'
        with open(train_filename, "r") as ins:
            for line in ins:
                if len(line) < 1:
                    continue
                
                origin_line_str = line
                # for each document
                doc = {}
                [sentence, label_str] = line.strip('\n\t ').split(',')
                sentence = sentence.strip('\t\n ')
                
                label_str = label_str.strip('\t\n ')
                label_id = int(label_str)
                if label_id not in self.LABELS:
                    print 'ERROR in input file. balel_id %d not found in LABELS%s'%(label_id, self.LABELS)
                    exit(1)
                    
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
                self.train.append(aDoc)

        
        test_filename = './Data/FOLDS/TIMERUN' + str(fold) + '/' + self.test_domain + '/test.txt'
        with open(test_filename, "r") as ins:
            for line in ins:
                if len(line) < 1:
                    continue
                
                origin_line_str = line
                # for each document
                doc = {}
                [sentence, label_str] = line.strip('\n\t ').split(',')
                sentence = sentence.strip('\t\n ')
                
                label_str = label_str.strip('\t\n ')
                label_id = int(label_str)
                if label_id not in self.LABELS:
                    print 'ERROR in input file. balel_id %d not found in LABELS%s'%(label_id, self.LABELS)
                    exit(1)
                    
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
                self.test.append(aDoc)
         
        print 'Finished read input file'

    def get_positive_train_docs(self):
        pos_docs = []
        for doc in self.train:
            if doc.human_label == 1:
                pos_docs.append(doc.origin_str)
        return pos_docs
                
                