import sys,os
sys.path.append(os.getcwd())
from maxent import maxent_model
from utils import DOMAINS

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'USAGE: python lifelong.py [alpha] [train_domain] [test_target_domain]'
        print 'Use domain in ', DOMAINS
        exit(1)
    alpha = int(sys.argv[1])
    train_domain = sys.argv[2]
    test_domain = sys.argv[3]
    
    if train_domain not in DOMAINS or test_domain not in DOMAINS:
        print 'Incorrect train_domain', train_domain, 'test_domain', test_domain
        exit(1)
    
    PAST_DOMAINS = [domain for domain in DOMAINS if domain not in [train_domain, test_domain]]
    
    print '===================================================================='
    print PAST_DOMAINS[0], '-->', train_domain, '-->' , test_domain 
    print '\n'
    model = maxent_model(PAST_DOMAINS[0], test_domain, True, alpha)
    model.train()
    
    model.change_domain(train_domain)
    model.train()
    model.print_results()
    
    
    print '====================================================================='
    print PAST_DOMAINS[1], '-->', PAST_DOMAINS[0], '-->', train_domain, '-->' , test_domain 
    print '\n'
    
    model = maxent_model(PAST_DOMAINS[1], test_domain, True, alpha)
    model.train()
    
    model.change_domain(PAST_DOMAINS[0])
    model.train()
    
    model.change_domain(train_domain)
    model.train()
    model.print_results()

    print '======================================================================='
    print PAST_DOMAINS[2], '-->',PAST_DOMAINS[1], '-->', PAST_DOMAINS[0], '-->', train_domain, '-->' , test_domain 
    print '\n'

    model = maxent_model(PAST_DOMAINS[2], test_domain, True, alpha)
    model.train()
    
    model.change_domain(PAST_DOMAINS[1])
    model.train()
    
    model.change_domain(PAST_DOMAINS[0])
    model.train()
    
    model.change_domain(train_domain)
    model.train()
    model.print_results()
    
    print '======================================================================='
    print PAST_DOMAINS[3], '-->', PAST_DOMAINS[2], '-->',PAST_DOMAINS[1], '-->', PAST_DOMAINS[0], '-->', train_domain, '-->' , test_domain 
    print '\n'
    
    model = maxent_model(PAST_DOMAINS[3], test_domain, True, alpha)
    model.train()

    model.change_domain(PAST_DOMAINS[2])
    model.train()
    
    model.change_domain(PAST_DOMAINS[1])
    model.train()
    
    model.change_domain(PAST_DOMAINS[0])
    model.train()
    
    model.change_domain(train_domain)
    model.train()
    model.print_results()


    