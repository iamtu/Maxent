This package implements maximum entropy for text classification.

Imput Data format:
line 1: 'sentence, label'
line 2: 'sentence, label'

run: python pre_process_data.py [datafile] [n_gram]
	to create [datafile.n_gram.txt] in data folder

run: python run.py [path to input file]