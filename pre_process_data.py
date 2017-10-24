from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys, re
from nltk.stem import PorterStemmer
ps = PorterStemmer()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'USAGE: python pre_process_data.py [data file name] [n-gram]'
    filename = sys.argv[1]
    n_gram = int(sys.argv[2])
    if n_gram not in [1,2,3] :
        print 'please Choose n_gram is 1,2 or 3'
        exit(1)
    output_file = filename + "_" + str(n_gram) + 'gram.txt'
    fout = open(output_file, 'w')
    
    STOP_WORDS = set(stopwords.words('english'))
    
    with open(filename, "r") as ins:
        for line in ins:
            [sentence, label_str] = line.strip('\n\t ').split(',')
            label_str = label_str.strip('\t\n ')
            sentence = re.sub(r'[^\w\s]','',sentence) # remove punctuation
            word_tokens = word_tokenize(sentence) # tokenize sentence
#             word_tokens = [w for w in word_tokens if not w in STOP_WORDS]
            for w in word_tokens:
                w = ps.stem(w)
            
            output_string = ''
            for w in word_tokens:
                output_string += w + " "
            sentence_length = len(word_tokens)
            if n_gram == 2:
                for i in range(sentence_length-1):
                    n_gram_word = ':'.join([word_tokens[i], word_tokens[i+1]])
                    output_string += n_gram_word + " "
                
            if n_gram == 3:
                for i in range(sentence_length-2):
                    n_gram_word = ':'.join([word_tokens[i], word_tokens[i+1], word_tokens[i+2]])
                    output_string += n_gram_word + " "
            
            output_string = output_string[:-1]
            
            output_string += ',' + label_str + '\n'
            fout.write(output_string)
    fout.close()
            





