import os
DATA_FOLDER = './Data_sugg'
DOMAINS = [x for x in os.listdir(DATA_FOLDER) if '.' not in x]

def get_file_path(domain):
    return DATA_FOLDER + '/' + domain + '/train.tagged.txt' 
