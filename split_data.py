import os,sys
sys.path.append(os.getcwd())
from document import document
from random import shuffle
import copy

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'USAGE: python split.py [data file name]'
        exit(1)
        
    filename = sys.argv[1]
    domain = filename.split('/')[-1].split('_')[0]
    
    cp_str_2_int = {} # cp_str_2_int['context_predicate_str'] = context_predicate_id
    cp_int_2_str = {}
    LABELS = [0, 1]
    
    print 'Begin reading document from file %s' %filename
    total_docs = []
    with open(filename, "r") as ins:
        for line in ins:
            origin_line_str = line
            # for each document
            doc = {}
            [sentence, label_str] = line.strip('\n\t ').split(',')
            sentence = sentence.strip('\t\n ')
            label_str = label_str.strip('\t\n ')

            label_id = int(label_str)
            if label_id not in LABELS:
                print 'ERROR in input file. balel_id %d not found in LABELS%s'%(label_id, LABELS)
                exit(1)
            tokens = sentence.split()
            for token_str in tokens:
                token_id = cp_str_2_int.get(token_str)
                if token_id == None: # not in context_predicate_map string - id
                    token_id = len(cp_str_2_int)
                    cp_str_2_int[token_str] = token_id
                    cp_int_2_str[token_id] = token_str
                    
                if token_id not in doc: # check if in doc or not
                    doc[token_id] = 1
                else :
                    doc[token_id] += 1
            
            aDoc = document(doc, label_id, origin_line_str)
            total_docs.append(aDoc)
        
        SIZE = len(total_docs)
        
        TIME_RUN = '0'
        print 'TIME RUN ', TIME_RUN
#         shuffle(total_docs)    
        
        train = copy.deepcopy(total_docs[0: 4*SIZE/5])
        test = copy.deepcopy(total_docs[4*SIZE/5 : ])
        
        # write vocab file
        print 'saving vocab file'
        output_vocab_file = './Data/FOLDS/TIMERUN' + TIME_RUN + '/'+ domain +'/vocab.txt'
        fout = open(output_vocab_file, 'w')
        for (cp_str, cp_id) in cp_str_2_int.iteritems():
            fout.write(cp_str + ' ' + str(cp_id) + '\n')
        fout.close()
        print 'saved vocab file'

        # write train file
        print 'saving train file'
        output_train_file = './Data/FOLDS/TIMERUN' + TIME_RUN + '/'+ domain +'/train.txt'
        fout = open(output_train_file, 'w')
        for doc in train:
            fout.write(doc.origin_str)
        fout.close()
        print 'saved train file'

        # write test file
        print 'saving test file'
        output_test_file = './Data/FOLDS/TIMERUN' + TIME_RUN + '/'+ domain +'/test.txt'
        fout = open(output_test_file, 'w')
        for doc in test:
            fout.write(doc.origin_str)
        fout.close()
        print 'saved test file'
        
        
        
#         for i in xrange(5):
#             TIME_RUN = str(i+1)
#             print 'TIME RUN ', TIME_RUN
#             shuffle(total_docs)    
#             
#             train = copy.deepcopy(total_docs[0: 4*SIZE/5])
#             test = copy.deepcopy(total_docs[4*SIZE/5 : ])
#             
#             # write vocab file
#             print 'saving vocab file'
#             output_vocab_file = './Data/FOLDS/TIMERUN' + TIME_RUN + '/'+ domain +'/vocab.txt'
#             fout = open(output_vocab_file, 'w')
#             for (cp_str, cp_id) in cp_str_2_int.iteritems():
#                 fout.write(cp_str + ' ' + str(cp_id) + '\n')
#             fout.close()
#             print 'saved vocab file'
# 
#             # write train file
#             print 'saving train file'
#             output_train_file = './Data/FOLDS/TIMERUN' + TIME_RUN + '/'+ domain +'/train.txt'
#             fout = open(output_train_file, 'w')
#             for doc in train:
#                 fout.write(doc.origin_str)
#             fout.close()
#             print 'saved train file'
# 
#             # write test file
#             print 'saving test file'
#             output_test_file = './Data/FOLDS/TIMERUN' + TIME_RUN + '/'+ domain +'/test.txt'
#             fout = open(output_test_file, 'w')
#             for doc in test:
#                 fout.write(doc.origin_str)
#             fout.close()
#             print 'saved test file'
# 


