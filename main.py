import json
import logging
from time import time

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import casual_tokenize
from plotly.offline import plot
from sklearn.manifold.isomap import Isomap
from sklearn.manifold.t_sne import TSNE

if __name__ == '__main__':
    time_start = time()
    with open('./settings.json') as settings_fp:
        settings = json.load(settings_fp)
        print(settings)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    random_state_ = settings['random_state'] if 'random_state' in settings.keys() else 0
    start_line = settings['text_start_line'] if 'text_start_line' in settings.keys() else 0
    stop_line = settings['text_stop_line'] if 'text_stop_line' in settings.keys() else -1
    word2vec_epochs_ = settings['word2vec_epochs'] if 'word2vec_epochs' in settings.keys() else 100
    word2vec_size_ = settings['word2vec_size'] if 'word2vec_size' in settings.keys() else 100
    # how many times does a word have to appear to be interesting?
    word2vec_min_count_ = settings['word2vec_min_count'] if 'word2vec_min_count' in settings.keys() else 10
    # how many threads will we use?
    if 'word2vec_workers' not in settings.keys():
        logging.warning('setting word2vec workers to default')
    word2vec_workers_ = settings['word2vec_workers'] if 'word2vec_workers' in settings.keys() else 1
    word2vec_compute_loss_ = settings['word2vec_compute_loss'] if 'word2vec_compute_loss' in settings.keys() else False
    do_plot = settings['do_plot'] if 'do_plot' in settings.keys() else False
    # todo make this a setting
    n_components_ = 2
    if input_file is None:
        print('input file not in settings. Quitting.')
        quit(1)

    with open(input_file, 'r', encoding='utf-8') as input_fp:
        text = input_fp.readlines()
        print('our input data has {} lines.'.format(len(text)))

    text = text[start_line: stop_line]  # exclude everything outside our window of interest
    time_word2vec = time()

    word2vec_model = Word2Vec(compute_loss=word2vec_compute_loss_, min_count=word2vec_min_count_,
                              seed=random_state_, size=word2vec_size_, workers=word2vec_workers_)
    training_data = [[word.lower() for word in casual_tokenize(item)] for item in text]
    word2vec_model.build_vocab(training_data)
    total_examples = word2vec_model.corpus_count
    print('word2vec total examples: {}'.format(total_examples))
    word2vec_model.train(training_data, epochs=word2vec_epochs_, total_examples=total_examples)
    print('word2vec took {:5.2f}s'.format(time() - time_word2vec))
    tsne_verbose_ = 2
    isomap_n_jobs_ = 4
    isomap_n_neighbors_ = 10
    if do_plot:
        time_projection = time()
        do_tsne = True
        n_iter_ = 10000
        # todo see if we can have the model not start with a random guess
        projection_model = TSNE(n_components=n_components_, n_iter=n_iter_, verbose=tsne_verbose_,
                                n_iter_without_progress=300) if do_tsne else Isomap(n_neighbors=isomap_n_neighbors_,
                                                                                    n_components=n_components_,
                                                                                    max_iter=n_iter_,
                                                                                    n_jobs=isomap_n_jobs_)

        X = word2vec_model.wv[word2vec_model.wv.vocab]
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
        counts = [word2vec_model.wv.vocab[word[0]].count for word in filtered]

        words_to_plot = 2100
        # todo add plotly plotting
        # todo add a slider for count
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
        do_matplotlib = False
        if do_matplotlib:
            for i, word in enumerate(words[:words_to_plot]):
                # ax.text(result[i, 0], result[i, 1], s=word, size=5, zorder=1, color='k')
                ax.text(xs[i], ys[i], s=word, size=10, zorder=1, color='k')
        else:
            mode_ = 'text'  # 'markers+text'

            trace1 = go.Scatter(hoverinfo='none',
                                marker=dict(line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1), opacity=0.8,
                                            size=6),
                                mode=mode_, text=words, x=xs, y=ys)
            data = [trace1]
            layout = go.Layout(margin=dict(l=0, t=0, r=0, b=0))
            fig = go.Figure(data=data, layout=layout)
            # todo move the output file name to settings
            output_file_name = input_file.replace('.txt', '.html')
            plot(fig, filename=output_file_name, auto_open=False)

        plt.axis('off')
        plt.show()

    print('total time: {:5.2f}s'.format(time() - time_start))
