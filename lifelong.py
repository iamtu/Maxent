import sys,os
sys.path.append(os.getcwd())

from model import model
from data import data

DOMAINS = ['electronics', 'hotel', 'suggForum', 'SuggHashtagTweets', 'TravelAdviceRetagged']
is_lifelong = True

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'USAGE: python lifelong.py [alpha] [TIMERUN] [train_domain] [test_domain]'
        exit(1)
    alpha = int(sys.argv[1])
    TIME_RUN = int(sys.argv[2])
    train_domain = sys.argv[3]
    test_domain = sys.argv[4]
    
    print '\n\n\t\talpha: ', alpha, ' --- TIME_RUN', TIME_RUN
    print 'train domain:,' , train_domain, ' / test domain: ', test_domain  
    past_domain = [domain for domain in DOMAINS if domain not in [train_domain, test_domain] ]
    
    data_domain_4 = data(train_domain, test_domain, TIME_RUN)
    
    
    print "Order 1: ", past_domain[0], '->', past_domain[1], '->', past_domain[2]
    data_domain_1 = data(past_domain[0], test_domain, TIME_RUN)
    data_domain_2 = data(past_domain[1], test_domain, TIME_RUN)
    data_domain_3 = data(past_domain[2], test_domain, TIME_RUN)

    max_ent = model(data_domain_1, is_lifelong, alpha)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()

    max_ent.change_domain(data_domain_2)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    
    max_ent.change_domain(data_domain_3)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    
    max_ent.change_domain(data_domain_4)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    
    print "Order 2: ", past_domain[1], '->', past_domain[2], '->', past_domain[0]
    data_domain_1 = data(past_domain[1], test_domain, TIME_RUN)
    data_domain_2 = data(past_domain[2], test_domain, TIME_RUN)
    data_domain_3 = data(past_domain[0], test_domain, TIME_RUN)

    max_ent = model(data_domain_1, is_lifelong, alpha)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()

    max_ent.change_domain(data_domain_2)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    
    max_ent.change_domain(data_domain_3)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    
    max_ent.change_domain(data_domain_4)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()

    print "Order 3: ", past_domain[2], '->', past_domain[0], '->', past_domain[1]
    data_domain_1 = data(past_domain[2], test_domain, TIME_RUN)
    data_domain_2 = data(past_domain[0], test_domain, TIME_RUN)
    data_domain_3 = data(past_domain[1], test_domain, TIME_RUN)

    max_ent = model(data_domain_1, is_lifelong, alpha)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()

    max_ent.change_domain(data_domain_2)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    
    max_ent.change_domain(data_domain_3)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    
    max_ent.change_domain(data_domain_4)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()

