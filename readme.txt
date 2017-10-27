This package implements maximum entropy for text classification.

Imput Data format: (last chacracter is label)
line 1: 'sentence,label'
line 2: 'sentence,label'

label is integer 0,1,2...

*)run: python pre_process_data.py [datafile] [n_gram]
	to create [datafile.n_gram.txt]

example: python pre_process_data.py ./Data/electronics 2

*)run: python run.py [path to input file]

example: python run.py ./Data/electronics_2gram.txt