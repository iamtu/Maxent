import sys,os
sys.path.append(os.getcwd())

from model import model
from data import data
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'USAGE: python run.py [data file name]'
    input_filename = sys.argv[1]
    data = data(input_filename)
    maxent_model = model(data)
    maxent_model.train()
    maxent_model.inference()
    maxent_model.validate()