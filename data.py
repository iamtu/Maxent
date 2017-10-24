import os,sys
sys.path.append(os.getcwd())
from document import document
import copy
'''
Read file data with line format:
sentence , label
'''
class data():
    def __init__(self, filename):
        self.train = [] # list of documents
        self.test = []
        self.cp_map = {} # self.cp_map['context_predicate_str'] = context_predicate_id
        self.labels = []
        
        total_docs = []
        with open(filename, "r") as ins:
            for line in ins:
                # for each document
                doc = {}
                [sentence, label_str] = line.strip('\n\t ').split(',')
                sentence = sentence.strip('\t\n ')
                label_str = label_str.strip('\t\n ')

                label_id = int(label_str)
                if label_id not in self.labels:
                    self.labels.append(label_id)
                                    
                tokens = sentence.split()
                for token in tokens:
                    token_id = self.cp_map.get(token)
                    if token_id == None: # not in context_predicate_map string - id
                        token_id = len(self.cp_map)
                        self.cp_map[token] = token_id
                        
                    if token_id not in doc: # check if in doc or not
                        doc[token_id] = 1
                    else :
                        doc[token_id] += 1
                
                aDoc = document(doc, label_id)
                total_docs.append(aDoc)
        
        SIZE = len(total_docs)
        self.train = copy.deepcopy(total_docs[0: 4*SIZE/5])
        self.test = copy.deepcopy(total_docs[4*SIZE/5 : ])
        del total_docs

if __name__ == '__main__':
    print 'reading file'
    data = data('./data/electronics.csv')
    print 'data.labels', data.labels
    print 'train data'
    for doc in data.train:
        print doc
    print 'test data'
    for doc in data.test:
        print doc
                
        
        