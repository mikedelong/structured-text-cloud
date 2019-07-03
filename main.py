import json
# from sklearn.decomposition import PCA
# from sklearn.manifold.t_sne import TSNE
# from nltk.corpus import stopwords
import logging
from time import time

# import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import casual_tokenize

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

    time_word2vec = time()

    random_state_ = 1
    word2vec_size_ = 400  # how big are the word-to-vec vectors?
    word2vec_min_count_ = 10  # how many times does a word have to appear to be interesting?
    word2vec_workers_ = 4  # how many threads will we use?
    word2vec_compute_loss_ = True
    word2vec_model = Word2Vec(compute_loss=word2vec_compute_loss_, min_count=word2vec_min_count_,
                              seed=random_state_, size=word2vec_size_, workers=word2vec_workers_)
    training_data = [[word.lower() for word in casual_tokenize(item)] for item in text]
    word2vec_model.build_vocab(training_data)
    total_examples = word2vec_model.corpus_count
    print('word2vec total examples: {}'.format(total_examples))
    epochs_ = 1000
    word2vec_model.train(training_data, epochs=epochs_, total_examples=total_examples)
    X = word2vec_model.wv[word2vec_model.wv.vocab]
    print('word2vec took {:5.2f}s'.format(time() - time_word2vec))

    print('total time: {:5.2f}s'.format(time() - time_start))
