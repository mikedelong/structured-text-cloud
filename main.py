import json
# import pandas as pd
# from gensim.models import Word2Vec
# from nltk.tokenize import casual_tokenize
# from sklearn.decomposition import PCA
# from sklearn.manifold.t_sne import TSNE
# from nltk.corpus import stopwords
import logging
from time import time

if __name__ == '__main__':
    time_start = time()
    with open('./settings.json') as settings_fp:
        settings = json.load(settings_fp)
        print(settings)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    if input_file is None:
        print('input file not in settings. Quitting.')
        quit(1)

    with open(input_file, 'r', encoding='utf-8') as input_fp:
        text = input_fp.readlines()
        print('our input data has {} lines.'.format(len(text)))
