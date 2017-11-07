import sys,os
sys.path.append(os.getcwd())

from model import model
from data import data

'''
Run maxent for each domain 5 times (5 folds)
'''
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'USAGE: python run_maxent.py [domain]'
        exit(1)
        
        
    domain = sys.argv[1]
    is_lifelong = False
    
    print 'run max_ent for domain', domain
    
    TIME_RUN = 0
    print 'timerun', TIME_RUN
    data_domain_1 = data(domain, TIME_RUN)
    max_ent = model(data_domain_1, is_lifelong)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    
#     
#     TIME_RUN = 2
#     print 'timerun', TIME_RUN
#     data_domain_2 = data(domain, TIME_RUN)
#     max_ent = model(data_domain_2, is_lifelong)
#     max_ent.train()
#     max_ent.inference()
#     max_ent.validate()
#     
#     TIME_RUN = 3
#     print 'timerun', TIME_RUN
#     data_domain_3 = data(domain, TIME_RUN)
#     max_ent = model(data_domain_3, is_lifelong)
#     max_ent.train()
#     max_ent.inference()
#     max_ent.validate()
#     
#     TIME_RUN = 4
#     print 'timerun', TIME_RUN
#     data_domain_4 = data(domain, TIME_RUN)
#     max_ent = model(data_domain_4, is_lifelong)
#     max_ent.train()
#     max_ent.inference()
#     max_ent.validate()
#     
#     TIME_RUN = 5
#     print 'timerun', TIME_RUN
#     data_domain_5 = data(domain, TIME_RUN)
#     max_ent = model(data_domain_5, is_lifelong)
#     max_ent.train()
#     max_ent.inference()
#     max_ent.validate()