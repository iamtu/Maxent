import sys,os
sys.path.append(os.getcwd())

from model import model
from data import data

'''
Run maxent for each domain 5 times (5 folds)
'''
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'USAGE: python run_maxent.py [alpha] [train_domain] [test_domain]'
        exit(1)
    alpha = int(sys.argv[1])
    train_domain = sys.argv[2]
    test_domain = sys.argv[3]
    
    print 'Run cross domain with max_ent, alpha = %d does not effect' % alpha
    print '\ttrain domain: ', train_domain, '/ test domain: ', test_domain
    
    is_lifelong = False
    
    TIME_RUN = 0
    print 'timerun', TIME_RUN
    
    data = data(train_domain, test_domain, TIME_RUN)
    max_ent = model(data, is_lifelong, alpha)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    max_ent.save_model()
    