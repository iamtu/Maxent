import sys,os
sys.path.append(os.getcwd())

from model import model
from data import data

'''
Run maxent for each domain 5 times (5 folds)
'''
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'USAGE: python run_maxent.py [train_domain] [test_domain]'
        exit(1)
        
        
    train_domain = sys.argv[1]
    test_domain = sys.argv[2]
    
    is_lifelong = False
    
    TIME_RUN = 0
    print 'timerun', TIME_RUN
    
    data = data(train_domain, test_domain, TIME_RUN)
    max_ent = model(data, is_lifelong)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    