import sys,os
sys.path.append(os.getcwd())

from maxent import maxent_model
from data import data

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'USAGE: python run_maxent.py [alpha] [train_file] [test_file]'
        exit(1)
    alpha = int(sys.argv[1])
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    
    print 'Run cross domain with maxent, alpha = %d does not effect' % alpha
    print '\ttrain file: ', train_file, '/ test domain: ', test_file
        
    is_lifelong = False
    model = maxent_model(train_file, test_file, is_lifelong, alpha)
    model.train()
    model.print_results()