import json
from time import time

# import pandas as pd
# from gensim.models import Word2Vec
# from nltk.tokenize import casual_tokenize
# from sklearn.decomposition import PCA
# from sklearn.manifold.t_sne import TSNE
# from nltk.corpus import stopwords
import logging

if __name__ == '__main__':
    time_start = time()
    with open('./settings.json') as settings_fp:
        settings = json.load(settings_fp)
        print(settings)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    quit(0)

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    if input_file is None:
        print('input file not in settings. Quitting.')
        quit(1)
