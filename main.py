import json
import logging
from time import time

import matplotlib.pyplot as plt
# import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import casual_tokenize
from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import TSNE

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
    do_plot = True
    if do_plot:
        time_projection = time()

        n_components_ = 2
        do_pca = False
        projection_model = PCA(n_components=n_components_) if do_pca else \
            TSNE(
                n_components=n_components_,
                n_iter=1000,
                n_iter_without_progress=300
            )
        result = projection_model.fit_transform(X)
        print('projection took {:5.2f}s'.format(time() - time_projection))

        words = list(word2vec_model.wv.vocab)
        print('the model vocabulary has {} words and they are {}'.format(len(words), words))
        stop_words = stopwords.words('english')
        filtered = list()
        for index, word in enumerate(words):
            if word not in stop_words and len(word) > 1 and not word.isdigit():
                filtered.append((word, result[index, 0], result[index, 1]))
        print('after we filter stopwords our vocabulary has {} words'.format(len(filtered)))

        # now reconstruct the words and results from the filtered result
        words = [word[0] for word in filtered]
        xs = [x[1] for x in filtered]
        ys = [y[2] for y in filtered]

        words_to_plot = 1000
        fig = plt.figure()
        if n_components_ == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(result[:words_to_plot, 0], result[:words_to_plot, 1], result[:words_to_plot, 2])
        elif n_components_ == 2:
            ax = fig.add_subplot(111)
            ax.scatter(xs[:words_to_plot], ys[:words_to_plot], s=1)
        else:
            raise ValueError('we should be plotting in 2 or 3 dimensions but n_components is {}'.format(n_components_))

        # todo only plot the most important words or the most popular words
        if n_components_ == 3:
            for i, word in enumerate(words[:words_to_plot]):
                ax.text(result[i, 0], result[i, 1], result[i, 2], '%s' % word, size=8, zorder=1, color='k')
        elif n_components_ == 2:
            for i, word in enumerate(words[:words_to_plot]):
                # ax.text(result[i, 0], result[i, 1], s=word, size=5, zorder=1, color='k')
                ax.text(xs[i], ys[i], s=word, size=5, zorder=1, color='k')
        else:
            raise ValueError('we should be labeling in 2 or 3 dimensions but n_components is {}'.format(n_components_))
        plt.axis('off')
        plt.show()

    print('total time: {:5.2f}s'.format(time() - time_start))
