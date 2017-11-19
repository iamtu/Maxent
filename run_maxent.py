import sys,os
sys.path.append(os.getcwd())
from maxent import maxent_model
from utils import DOMAINS

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'USAGE: python run_maxent.py [train_domain] [test_domain]'
        print 'Use domain in ', DOMAINS
        exit(1)
    train_domain = sys.argv[1]
    test_domain = sys.argv[2]
    
    print 'Run cross domain with maxent'
    print '\ttrain domain: ', train_domain, '/ test domain: ', test_domain
        
    model = maxent_model(train_domain, test_domain, False, 0)
    model.train()
    model.print_results()