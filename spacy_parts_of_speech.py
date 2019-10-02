import json
import logging
from time import time

import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.manifold.isomap import Isomap
from sklearn.manifold.t_sne import TSNE
from spacy import load

from common import get_setting


def token_lower(arg):
    pieces = arg.split('/')
    return '/'.join([pieces[0].lower(), pieces[1]])


if __name__ == '__main__':
    time_start = time()
    with open('./locations_and_counts.json') as settings_fp:
        settings = json.load(settings_fp)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('settings are {}'.format(settings))

    do_plot = get_setting('do_plot', settings)
    if do_plot is None:
        do_plot = False
        logging.warning('do_plot is missing from settings; using default value {}'.format(do_plot))
    else:
        logging.info('do plot: {}'.format(do_plot))

    do_isomap = get_setting('do_isomap', settings)
    if do_isomap is None:
        do_isomap = False
        logging.warning('do_isomap is missing from settings; using default value {}'.format(do_plot))
    else:
        logging.info('do isomap: {}'.format(do_isomap))

    do_tsne = get_setting('do_tsne', settings)
    if do_tsne is None:
        do_tsne = False
        logging.warning('do_tsne is missing from settings; using default value {}'.format(do_plot))
    else:
        logging.info('do TNSE: {}'.format(do_tsne))

    if do_tsne and do_isomap:
        logging.error('Check settings: do_tsne and do_isomap cannot both be true. Quitting.')
        quit(1)

    input_encoding = get_setting('input_encoding', settings)
    if input_encoding is None:
        input_encoding = 'utf-8'
        logging.warning('input_encoding is missing from settings; using default value {}'.format(input_encoding))
    else:
        logging.info('input_encoding: {}'.format(input_encoding))

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    if input_file is None:
        logging.error('input file not in settings. Quitting.')
        quit(1)
    else:
        logging.info('input file: {}'.format(input_file))

    isomap_n_jobs_ = get_setting('isomap_job_count', settings)
    if isomap_n_jobs_ is None:
        isomap_n_jobs_ = 1
        logging.warning('isomap job count not set; using default value {}'.format(isomap_n_jobs_))
    else:
        logging.info('isomap job count: {}'.format(isomap_n_jobs_))
    if 'isomap_job_count' not in settings.keys():
        logging.warning('setting IsoMap parallelism (job count) to default/serial')

    isomap_n_neighbors_ = get_setting('isomap_neighbor_count', settings)
    if isomap_n_neighbors_ is None:
        isomap_n_neighbors_ = 5
        logging.warning('isomap neighbors not set; using default value {}'.format(isomap_n_neighbors_))
    else:
        logging.info('isomap neighbors: {}'.format(isomap_n_neighbors_))
    if 'isomap_n_neighbors_' not in settings.keys():
        logging.warning('setting Isomap neighbor count to default.')

    n_components_ = get_setting('plot_dimensions', settings)
    if n_components_ is None:
        n_components_ = 2
        logging.warning('plot dimensions not set; using default value {}'.format(n_components_))
    else:
        logging.info('plot dimensions: {}'.format(n_components_))

    if n_components_ != 2:
        raise ValueError('we should be plotting in 2 or 3 dimensions but n_components is {}'.format(n_components_))

    n_iter_ = get_setting('projection_iteration_count', settings)
    if n_iter_ is None:
        n_iter_ = 100
        logging.warning('projection iteration count (t-SNE/Isomap) not set; using default value {}'.format(n_iter_))
    else:
        logging.info('projection iteration count (t-SNE/Isomap): {}'.format(n_iter_))

    random_state_ = get_setting('random_state', settings)
    if random_state_ is None:
        random_state_ = 0
        logging.warning('random state not in settings, defaulting to: {}'.format(random_state_))
    else:
        logging.info('random state: {}'.format(random_state_))

    start_line = get_setting('text_start_line', settings)
    if start_line is None:
        start_line = 0
        logging.warning('text start line not in settings, defaulting to: {}'.format(start_line))
    else:
        logging.info('text start line: {}'.format(start_line))

    stop_line = get_setting('text_stop_line', settings)
    if stop_line is None:
        stop_line = -1
        logging.warning('text stop line not in settings, defaulting to: {}'.format(stop_line))
    else:
        logging.info('text stop line: {}'.format(stop_line))

    word2vec_epochs_ = get_setting('word2vec_epochs', settings)
    if word2vec_epochs_ is None:
        word2vec_epochs_ = 100
        logging.warning('model (word2vec) epochs not in settings; defaulting to {}'.format(word2vec_epochs_))
    else:
        logging.info('model (word2vec) epochs: {}'.format(word2vec_epochs_))

    tsne_verbose_ = settings['tsne_verbose'] if 'tsne_verbose' in settings.keys() else 0
    if 'tsne_verbose' not in settings.keys():
        logging.warning('setting t-SNE verbosity to default')

    n_iter_without_progress_ = settings[
        'tsne_iterations_without_progress'] if 'tsne_iterations_without_progress' in settings.keys() else 50
    if 'tsne_iterations_without_progress' not in settings.keys():
        logging.warning('setting t-SNE iterations without progress to default'.format(n_iter_without_progress_))
    tsne_init_ = settings['tsne_initialization'] if 'tsne_initialization' in settings.keys() else 'random'

    word2vec_compute_loss_ = settings['word2vec_compute_loss'] if 'word2vec_compute_loss' in settings.keys() else False
    word2vec_epochs_ = settings['word2vec_epochs'] if 'word2vec_epochs' in settings.keys() else 100
    # how many times does a word have to appear to be interesting?
    word2vec_min_count_ = settings['word2vec_min_count'] if 'word2vec_min_count' in settings.keys() else 10
    word2vec_size_ = settings['word2vec_size'] if 'word2vec_size' in settings.keys() else 100
    # how many threads will we use?
    word2vec_workers_ = settings['word2vec_workers'] if 'word2vec_workers' in settings.keys() else 1
    if 'word2vec_workers' not in settings.keys():
        logging.warning('setting word2vec workers to default')

    with open(input_file, 'r', encoding=input_encoding) as input_fp:
        text = input_fp.readlines()
        logging.info('our input data has {} lines.'.format(len(text)))

    # exclude everything outside our window of interest
    # concatenate all the text into one long string, because that's what spacy needs to see
    # remove newlines and squeeze out extra space
    logging.info('started squeezing text')
    text = ' '.join(text[start_line: stop_line]).replace('\n', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    logging.info('finished squeezing text')

    parser = load('en_core_web_sm')
    parser.max_length = len(text) + 1
    logging.info('current available pipes are {}'.format({item for item in parser.pipe_names}))
    # todo combine noun tokens to convert compound nouns
    # https://hackernoon.com/word2vec-part-1-fe2ec6514d70
    with parser.disable_pipes('ner'):
        logging.info('starting parsing')
        result = parser(text=text)
        logging.info('parsing complete')
        logging.info('we have {} sentences'.format(len(list(result.sents))))
        sentences = list(result.sents)
        for index in range(10):
            sentence = sentences[index]
            logging.info(' '.join([token_lower(str('{}/{}'.format(item, item.tag_))) for item in sentence]))

    time_word2vec = time()
    word2vec_model = Word2Vec(compute_loss=word2vec_compute_loss_, min_count=word2vec_min_count_,
                              seed=random_state_, size=word2vec_size_, workers=word2vec_workers_)
    logging.info('created the Word2Vec model')
    training_data = [
        [token_lower(token) for token in ' '.join([str('{}/{}'.format(item, item.tag_)) for item in sentence]).split()]
        for sentence in sentences]
    logging.info('created the part-marked, properly-cased training data for Word2Vec')
    word2vec_model.build_vocab(training_data)
    total_examples = word2vec_model.corpus_count
    logging.info('word2vec total examples: {}'.format(total_examples))
    word2vec_model.train(training_data, epochs=word2vec_epochs_, total_examples=total_examples)
    logging.info('word2vec took {:5.2f}s'.format(time() - time_word2vec))

    if do_plot:
        time_projection = time()
        projection_model = TSNE(n_components=n_components_, n_iter=n_iter_, verbose=tsne_verbose_,
                                n_iter_without_progress=n_iter_without_progress_,
                                init=tsne_init_) if do_tsne else Isomap(n_neighbors=isomap_n_neighbors_,
                                                                        n_components=n_components_, max_iter=n_iter_,
                                                                        n_jobs=isomap_n_jobs_)

        X = word2vec_model.wv[word2vec_model.wv.vocab]
        result = projection_model.fit_transform(X)
        logging.info('projection took {:5.2f}s'.format(time() - time_projection))

        words = list(word2vec_model.wv.vocab)
        # todo make this a setting or remove it
        report_vocabulary_limit = 20
        logging.info('the model vocabulary has {} words and they are {}'.format(len(words),
                                                                                words[:report_vocabulary_limit]))
        stop_words = stopwords.words('english')
        filtered = [(word, result[index, 0], result[index, 1]) for index, word in enumerate(words) if
                    word not in stop_words and len(word) > 1 and not word.isdigit()]

        logging.info('after we filter stopwords our vocabulary has {} words'.format(len(filtered)))

        # now reconstruct the words and results from the filtered result
        words = [word[0] for word in filtered]
        counts = [word2vec_model.wv.vocab[word[0]].count for word in filtered]
        result_df = pd.DataFrame.from_dict({'word': [item.split('/')[0] for item in words],
                                            'x': [x[1] for x in filtered], 'y': [y[2] for y in filtered],
                                            'count': counts, 'part_of_speech': [item.split('/')[1] for item in words]})
        output_file = input_file.replace('.txt', '_pos.csv')
        logging.info('writing result DataFrame to {}'.format(output_file))
        result_df.to_csv(output_file, index=True, header=True)

    logging.info('total time: {:5.2f}s'.format(time() - time_start))
