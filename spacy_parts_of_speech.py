import json
import logging
from time import time

from spacy import load


def token_lower(arg):
    pieces = arg.split('/')
    return '/'.join([pieces[0].lower(), pieces[1]])


if __name__ == '__main__':
    time_start = time()
    with open('./locations_and_counts.json') as settings_fp:
        settings = json.load(settings_fp)
        # print(settings)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('settings are {}'.format(settings))

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    random_state_ = settings['random_state'] if 'random_state' in settings.keys() else 0
    start_line = settings['text_start_line'] if 'text_start_line' in settings.keys() else 0
    stop_line = settings['text_stop_line'] if 'text_stop_line' in settings.keys() else -1
    if input_file is None:
        logging.warning('input file not in settings. Quitting.')
        quit(1)

    input_encoding = settings['input_encoding'] if 'input_encoding' in settings.keys() else 'utf-8'
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
    # todo think about whether we need to handle capitalization before we use word2vec
    with parser.disable_pipes('ner'):
        logging.info('starting parsing')
        result = parser(text=text)
        logging.info('parsing complete')
        logging.info('we have {} sentences'.format(len(list(result.sents))))
        sentences = list(result.sents)
        for index in range(10):
            sentence = sentences[index]
            logging.info(' '.join([token_lower(str('{}/{}'.format(item, item.tag_))) for item in sentence]))

    logging.info('total time: {:5.2f}s'.format(time() - time_start))
