import json
import logging
from time import time

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import casual_tokenize
from sklearn.manifold.isomap import Isomap
from sklearn.manifold.t_sne import TSNE

if __name__ == '__main__':
    time_start = time()
    with open('./locations_and_counts.json') as settings_fp:
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

    isomap_n_neighbors_ = settings['isomap_neighbor_count'] if 'isomap_neighbor_count' in settings.keys() else 5
    if 'isomap_n_neighbors_' not in settings.keys():
        logging.warning('setting Isomap neighbor count to default.')
    n_iter_ = settings['projection_iteration_count'] if 'projection_iteration_count' in settings.keys() else 100
    if 'projection_iteration_count' not in settings.keys():
        logging.warning('setting projection (t-SNE/Isomap) iteration count to default'.format(n_iter_))
    n_iter_without_progress_ = settings[
        'tsne_iterations_without_progress'] if 'tsne_iterations_without_progress' in settings.keys() else 50
    if 'tsne_iterations_without_progress' not in settings.keys():
        logging.warning('setting t-SNE iterations without progress to default'.format(n_iter_without_progress_))
    tsne_init_ = settings['tsne_initialization'] if 'tsne_initialization' in settings.keys() else 'random'
    do_tsne = settings['do_tsne'] if 'do_tsne' in settings.keys() else False
    do_isomap = settings['do_isomap'] if 'do_isomap' in settings.keys() else False
    if do_tsne and do_isomap:
        logging.error('Check settings: do_tsne and do_isomap cannot both be true. Quitting.')
        quit(1)
    if input_file is None:
        print('input file not in settings. Quitting.')
        quit(1)

    input_encoding = settings['input_encoding'] if 'input_encoding' in settings.keys() else 'utf-8'
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
        projection_model = TSNE(n_components=n_components_, n_iter=n_iter_, verbose=tsne_verbose_,
                                n_iter_without_progress=n_iter_without_progress_,
                                init=tsne_init_) if do_tsne else Isomap(n_neighbors=isomap_n_neighbors_,
                                                                        n_components=n_components_, max_iter=n_iter_,
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

    print('total time: {:5.2f}s'.format(time() - time_start))
