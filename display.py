import json
import logging
from collections import Counter
from time import time

import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from wiktionaryparser import WiktionaryParser


# todo add file storage for performance
def get_part_of_speech(arg, arg_parser, arg_known):
    if arg in arg_known.keys():
        return arg_known[arg]
    else:
        result = arg_parser.fetch(arg)
        if not len(result):
            return 'unknown'
        return result[0]['definitions'][0]['partOfSpeech'] if len(result[0]['definitions']) > 0 else ''


part_of_speech_color_map = {
    'noun': 'red',
    'verb': 'blue',
    'adjective': 'orange',
    'adverb': 'green',
    'preposition': 'purple',
    'numeral': 'yellow',
    'proper noun': 'tomato',
    'pronoun': 'fuchsia',
    'determiner': 'purple',
    'conjunction': 'orchid',
    'interjection': 'honeydew',
    'unknown': 'black'
}

if __name__ == '__main__':
    time_start = time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    with open('./display.json') as settings_fp:
        settings = json.load(settings_fp)
        logging.info(settings)

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    if input_file is None:
        logging.error('No input file specified. Quitting.')
        quit(1)

    part_of_speech_file = settings['part_of_speech_file'] if 'part_of_speech_file' in settings.keys() else None
    if part_of_speech_file is None:
        logging.error('No part of speech file specified. Quitting.')
        quit(2)

    parser = WiktionaryParser()

    data_df = pd.read_csv(input_file)

    output_file_name = settings['output_file'] if 'output_file' in settings.keys() else None
    if output_file_name is None:
        logging.error('No output file specified. Quitting.')
        quit(1)
    mode_ = 'text'  # 'markers+text'
    slice_limit = 10
    slices = sorted(data_df['count'].unique().tolist())

    total = sum(slices)
    counts = Counter(data_df['count'].values.tolist())
    data_df = data_df.sort_values(by=['count'], axis=0, ascending=True)
    data_df['cumulative'] = data_df['count'].cumsum()
    # if we have more than ten slices refit to ten
    if len(slices) > slice_limit:
        quantiles = [1.0 / float(slice_limit) * index - 1.0 / (2.0 * float(slice_limit)) for index in
                     range(1, slice_limit + 1)]
    else:
        quantiles = [float(index) / float(len(slices)) for index in range(1, len(slices))]
    interpolation_ = 'lower'

    known_part_of_speech_df = pd.read_csv(part_of_speech_file, usecols=['word', 'part_of_speech'])
    known_part_of_speech = {row['word']: row['part_of_speech'] for _, row in known_part_of_speech_df.iterrows()}
    # known_part_of_speech = {}
    data_df['part_of_speech'] = data_df['word'].apply(get_part_of_speech, args=(parser, known_part_of_speech))
    data_df['part_of_speech'] = data_df['part_of_speech'].fillna('unknown')
    # write the known parts of speech to a file before we proceed
    # todo we want to accumulate known parts of speech from case to case rather than overwriting
    for index, row in data_df.iterrows():
        known_part_of_speech[row['word']] = row['part_of_speech']
    pd.DataFrame.from_dict({'word': list(known_part_of_speech.keys()),
                            'part_of_speech': list(known_part_of_speech.values())},
                           orient='columns').sort_values(axis=0, by='word').to_csv('./data/part_of_speech.csv',
                                                                                   index=True, header=True)
    logging.info(data_df['part_of_speech'].value_counts().to_dict())
    data_df['color'] = data_df['part_of_speech'].map(part_of_speech_color_map)

    stretch_factor = 1.05
    fig = go.Figure(data=[go.Scatter(
        marker=dict(line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1), opacity=0.8, size=6), mode=mode_,
        text=data_df[data_df['cumulative'] > data_df['cumulative'].quantile(q=quantile,
                                                                            interpolation=interpolation_)]['word'],
        x=data_df[data_df['cumulative'] > data_df['cumulative'].quantile(q=quantile,
                                                                         interpolation=interpolation_)]['x'],
        y=data_df[data_df['cumulative'] > data_df['cumulative'].quantile(q=quantile,
                                                                         interpolation=interpolation_)]['y'],
        hovertext=data_df[data_df['cumulative'] > data_df['cumulative'].quantile(q=quantile,
                                                                                 interpolation=interpolation_)][
            'count'],
        name='level: {}'.format(index),
        textfont=dict(color=data_df[
            data_df['cumulative'] > data_df['cumulative'].quantile(q=quantile, interpolation=interpolation_)]['color']),

    ) for index, quantile in enumerate(quantiles)],
        layout=dict(
            sliders=[dict(
                active=len(quantiles) // 2,
                pad={'t': 1},
                steps=[dict(method='restyle', args=['visible', [j == i for j in range(len(quantiles))]]) for i in
                       range(len(quantiles))]
            )],
            xaxis=dict(visible=False, range=[stretch_factor * data_df['x'].min(), stretch_factor * data_df['x'].max()]),
            yaxis=dict(visible=False, range=[stretch_factor * data_df['y'].min(), stretch_factor * data_df['y'].max()]),
        ))
    plot(fig, filename=output_file_name, auto_open=False)

    logging.info('total time: {:5.2f}s'.format(time() - time_start))
