import sys,os
sys.path.append(os.getcwd())

from model import model
from data import data

if __name__ == '__main__':
    data_domain_1 = data('./Data/electronics_2_gram.txt')
    data_domain_2 = data('./Data/hotel_2_gram.txt')
    data_domain_3 = data('./Data/suggForum_uservoice_2_gram.txt')
    data_domain_4 = data('./Data/SuggHashtagTweets_2_gram.txt')
    data_domain_5 = data('./Data/TravelAdviceRetagged_2_gram.txt')
    
    max_ent = model(data_domain_1)
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
    
    max_ent.change_domain(data_domain_5)
    max_ent.train()
    max_ent.inference()
    max_ent.validate()
    