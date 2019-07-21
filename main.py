import json
import logging
from time import time

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
    n_components_ = settings['plot_dimensions'] if 'plot_dimensions' in settings.keys() else 2
    if n_components_ != 2:
        raise ValueError('we should be plotting in 2 or 3 dimensions but n_components is {}'.format(n_components_))
    tsne_verbose_ = settings['tsne_verbose'] if 'tsne_verbose' in settings.keys() else 0
    if 'tsne_verbose' not in settings.keys():
        logging.warning('setting t-SNE verbosity to default')
    isomap_n_jobs_ = settings['isomap_job_count'] if 'isomap_job_count' in settings.keys() else 1
    if 'isomap_job_count' not in settings.keys():
        logging.warning('setting IsoMap parallelism (job count) to default/serial')

    # todo make this a setting
    isomap_n_neighbors_ = 10
    isomap_n_neighbors_ = settings['isomap_neighbor_count'] if 'isomap_neighbor_count' in settings.keys() else 5
    if 'isomap_n_neighbors_' not in settings.keys():
        logging.warning('setting Isomap neighbor count to default.')
    n_iter_ = 10000
    do_tsne = True
    if input_file is None:
        print('input file not in settings. Quitting.')
        quit(1)

    input_encoding = 'utf-8'
    with open(input_file, 'r', encoding=input_encoding) as input_fp:
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
    if do_plot:
        time_projection = time()
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
        report_vocabulary_limit = 20
        print('the model vocabulary has {} words and they are {}'.format(len(words), words[:report_vocabulary_limit]))
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

        mode_ = 'text'  # 'markers+text'
        fig = go.Figure(data=[], layout=[])
        for step in range(1, word2vec_min_count_ + 1):
            # todo only plot the most important words or the most popular words

            fig.add_trace(go.Scatter(hoverinfo='none',
                                     marker=dict(line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1), opacity=0.8,
                                                 size=6),
                                     mode=mode_, text=words,
                                     x=[xs[index] for index in range(len(filtered)) if counts[index] < step],
                                     y=[ys[index] for index in range(len(filtered)) if counts[index] < step]))

        # data = []
        fig.data[10].visible = True
        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="restyle",
                args=["visible", [False] * len(fig.data)],
            )
            step["args"][1][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        fig.update_layout(
            sliders=[dict(
                active=10,
                currentvalue={"prefix": "Frequency: "},
                pad={"t": 50},
                steps=steps
            )]
        )

        # data = [trace1]
        # layout = go.Layout(margin=dict(l=0, t=0, r=0, b=0))
        # fig = go.Figure(data=data, layout=layout)
        # todo move the output file name to settings
        output_file_name = input_file.replace('.txt', '.html')
        plot(fig, filename=output_file_name, auto_open=False)

    print('total time: {:5.2f}s'.format(time() - time_start))
