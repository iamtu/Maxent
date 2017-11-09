import sys,os
sys.path.append(os.getcwd())

from model import model
from data import data

if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print 'USAGE: python lifelong.py [TIMERUN]'
#         exit(1)
        
    TIME_RUN = 0
    print 'time run', TIME_RUN
    
#     data_domain_1 = data('electronics', 'hotel', TIME_RUN)
#     data_domain_2 = data('TravelAdviceRetagged', 'hotel',TIME_RUN)
#     data_domain_3 = data('SuggHashtagTweets', 'hotel', TIME_RUN)
    
#     data_domain_1 = data('SuggHashtagTweets', 'hotel', TIME_RUN)
#     data_domain_2 = data('electronics', 'hotel',TIME_RUN)
#     data_domain_3 = data('TravelAdviceRetagged', 'hotel', TIME_RUN)

#     data_domain_1 = data('TravelAdviceRetagged', 'electronics', TIME_RUN)
#     data_domain_2 = data('SuggHashtagTweets', 'electronics',TIME_RUN)
#     data_domain_3 = data('hotel', 'electronics', TIME_RUN)

    
#     data_domain_1 = data('SuggHashtagTweets', 'electronics', TIME_RUN)
#     data_domain_2 = data('hotel', 'electronics',TIME_RUN)
#     data_domain_3 = data('TravelAdviceRetagged', 'electronics', TIME_RUN)

    data_domain_1 = data('hotel', 'electronics', TIME_RUN)
    data_domain_2 = data('TravelAdviceRetagged', 'electronics',TIME_RUN)
    data_domain_3 = data('SuggHashtagTweets', 'electronics', TIME_RUN)

    data_domain_4 = data('suggForum', 'electronics', TIME_RUN)
    
    
    is_lifelong = True

    max_ent = model(data_domain_1,is_lifelong)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    max_ent.save_model()

    max_ent.change_domain(data_domain_2)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    max_ent.save_model()
    
    max_ent.change_domain(data_domain_3)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    max_ent.save_model()
    
    max_ent.change_domain(data_domain_4)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    max_ent.save_model()
    
    
