import json
import logging
from collections import Counter
from time import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib.pyplot import cm
from plotly.offline import plot
from wiktionaryparser import WiktionaryParser

from common import get_setting

# 'HYPH',
supported_parts_of_speech = ['-LRB-', '-RRB-', 'ADD', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                             'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
                             'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
                             'WRB']


def float_color_to_hex(arg_float, arg_colormap):
    color_value = tuple([int(255 * arg_colormap(arg_float)[index]) for index in range(3)])
    return '#{:02x}{:02x}{:02x}'.format(color_value[0], color_value[1], color_value[2])


def get_part_of_speech(arg, arg_parser, arg_known):
    if arg in arg_known.keys():
        return arg_known[arg]
    else:
        result = arg_parser.fetch(arg)
        if not len(result):
            return 'unknown'
        return result[0]['definitions'][0]['partOfSpeech'] if len(result[0]['definitions']) > 0 else ''


def get_quantile(arg_df, arg_column, arg_quantile, arg_interpolation):
    return arg_df[arg_df[arg_column] > data_df[arg_column].quantile(q=arg_quantile, interpolation=arg_interpolation)]


if __name__ == '__main__':
    time_start = time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    with open('./display.json') as settings_fp:
        settings = json.load(settings_fp)
        logging.info('settings: {}'.format(settings))

    colormap_name = get_setting('colormap', settings)
    if colormap_name is None:
        colormap_name = 'jet'
        logging.warning('colormap not set, defaulting to default value: {}'.format(colormap_name))

    colormap_strategy = get_setting('color_map_strategy', settings)
    if colormap_strategy is None:
        colormap_strategy = 'uniform'
        logging.warning('Color map strategy not specified. Defaulting to {}'.format(colormap_strategy))
    color_map_strategies = ['cumsum', 'uniform']
    if colormap_strategy not in color_map_strategies:
        logging.error(
            'Color map strategy must be in {} but is {}. Quitting.'.format(color_map_strategies, colormap_strategy))
        quit(3)

    input_file = get_setting('input_file', settings)
    if input_file is None:
        logging.error('No input file specified. Quitting.')
        quit(1)

    max_pages = get_setting('max_pages_to_show', settings)
    if max_pages is None:
        max_pages = 10
        logging.warning('Max pages to show not set, defaulting to default value: {}'.format(max_pages))

    max_words_to_show = get_setting('max_words_to_show', settings)
    if max_words_to_show is None:
        max_words_to_show = 300
        logging.warning('Max words to show not set, defaulting to default value: {}'.format(max_words_to_show))

    output_file_name = get_setting('output_file', settings)
    if output_file_name is None:
        logging.error('No output file specified. Quitting.')
        quit(1)

    part_of_speech_file = settings['part_of_speech_file'] if 'part_of_speech_file' in settings.keys() else None
    if part_of_speech_file is None:
        logging.error('No part of speech file specified. Quitting.')
        quit(2)

    rebuild_part_of_speech_file = settings[
        'rebuild_part_of_speech_file'] if 'rebuild_part_of_speech_file' in settings.keys() else None
    if rebuild_part_of_speech_file is None:
        rebuild_part_of_speech_file = False
        logging.warning(
            'Rebuild part of speech flag not specified. Defaulting to {}'.format(rebuild_part_of_speech_file))

    data_df = pd.read_csv(input_file)
    # we need to squeeze out spaces from the column names before we proceed
    data_df.rename(columns={item: item.strip() for item in list(data_df)}, inplace=True)
    logging.info('column names after load and rename: {}'.format(list(data_df)))

    token_map = {item: item if item in supported_parts_of_speech else 'PUNCT' for item in
                 data_df['part_of_speech'].unique().tolist()}
    for key, value in token_map.items():
        data_df['part_of_speech'] = data_df['part_of_speech'].replace(key, value)

    slices = sorted(data_df['count'].unique().tolist())
    counts = Counter(data_df['count'].values.tolist())
    data_df = data_df.sort_values(by=['count'], axis=0, ascending=True)
    # if we have more than max_pages slices refit
    if len(slices) > max_pages:
        quantiles = [1.0 / float(max_pages) * index - 1.0 / (2.0 * float(max_pages)) for index in
                     range(1, max_pages + 1)]
    else:
        quantiles = [float(index) / float(len(slices)) for index in range(1, len(slices))]

    # we only need to create the part of speech data if it does not already exist
    if 'part_of_speech' not in list(data_df):
        parser = WiktionaryParser()
        known_part_of_speech_df = pd.read_csv(part_of_speech_file, usecols=['word', 'part_of_speech'])
        known_part_of_speech = {row['word']: row['part_of_speech'] for _, row in known_part_of_speech_df.iterrows()}
        if rebuild_part_of_speech_file:
            known_part_of_speech = {}
        data_df['part_of_speech'] = data_df['word'].apply(get_part_of_speech, args=(parser, known_part_of_speech))
        data_df['part_of_speech'] = data_df['part_of_speech'].fillna('unknown')
        # write the known parts of speech to a file before we proceed
        for index, row in data_df.iterrows():
            known_part_of_speech[row['word']] = row['part_of_speech']
        pd.DataFrame.from_dict({'word': list(known_part_of_speech.keys()),
                                'part_of_speech': list(known_part_of_speech.values())},
                               orient='columns').sort_values(axis=0, by='word').to_csv(part_of_speech_file,
                                                                                       index=True, header=True)
    logging.info('part of speech counts: {}'.format(data_df['part_of_speech'].value_counts().to_dict()))

    colormap = cm.get_cmap(colormap_name)
    do_original = False
    if do_original:
        if colormap_strategy == 'cumsum':
            # use the cumsum of the value counts to assign a color from the colormap by hex string
            part_of_speech_color_map = data_df['part_of_speech'].value_counts(normalize=True).cumsum(
            ).apply(lambda x: float_color_to_hex(x, colormap)).to_dict()
        elif colormap_strategy == 'uniform':
            # use evenly-spaced colors from a colormap
            part_of_speech_color_map = {data_df['part_of_speech'].unique()[index]: float_color_to_hex(
                np.linspace(0, 1, data_df['part_of_speech'].nunique())[index], colormap) for index in
                range(data_df['part_of_speech'].nunique())}
        else:
            raise NotImplementedError('color map: {}'.format(colormap_strategy))
    else:
        # https://stackoverflow.com/questions/60208/replacements-for-switch-statement-in-python?rq=1
        supported_color_maps = {'cumsum', 'uniform'}
        if colormap_strategy in supported_color_maps:
            part_of_speech_color_map = {
                'cumsum': data_df['part_of_speech'].value_counts(normalize=True).cumsum(
                ).apply(lambda x: float_color_to_hex(x, colormap)).to_dict(),
                'uniform': {data_df['part_of_speech'].unique()[index]: float_color_to_hex(
                    np.linspace(0, 1, data_df['part_of_speech'].nunique())[index], colormap) for index in
                    range(data_df['part_of_speech'].nunique())}
            }[colormap_strategy]
        else:
            raise NotImplementedError('color map: {}'.format(colormap_strategy))

    logging.info('color map {}: {}'.format(colormap_strategy, part_of_speech_color_map))
    data_df['color'] = data_df['part_of_speech'].map(part_of_speech_color_map)

    # get the cut level
    cut_level = data_df.nlargest(n=max_words_to_show, columns=['count'], keep='all')['count'].min()
    data_df = data_df[data_df['count'] >= cut_level]
    logging.info('our cut level of {} leaves us with {} rows/words to display'.format(max_words_to_show, len(data_df)))

    mode_ = 'text'  # 'markers+text'
    color_ = 'rgba(217, 217, 217, 0.14)'
    interpolation_ = 'lower'
    opacity_ = 0.8
    size_ = 6
    stretch_factor = 1.05
    width_ = 0.1
    data_df['cumulative'] = data_df['count'].cumsum()
    counts = ['{}'.format(get_quantile(data_df, 'cumulative', quantile, interpolation_)['count'].min()) for quantile in
              quantiles]

    # todo figure out how to add an item to the legend per color/part of speech
    figure = go.Figure(data=[go.Scatter(
        hoverinfo='text',
        hovertext=get_quantile(data_df, 'cumulative', quantile, interpolation_)['count'],
        marker=dict(line=dict(color=color_, width=width_), opacity=opacity_, size=size_), mode=mode_,
        text=get_quantile(data_df, 'cumulative', quantile, interpolation_)['word'],
        textfont=dict(color=get_quantile(data_df, 'cumulative', quantile, interpolation_)['color']),
        x=get_quantile(data_df, 'cumulative', quantile, interpolation_)['x'],
        y=get_quantile(data_df, 'cumulative', quantile, interpolation_)['y'],
    ) for index, quantile in enumerate(quantiles)],
        layout=dict(
            showlegend=False,  # let's turn this off until we have figured out how to do it properly
            sliders=[dict(
                active=len(quantiles) // 2,
                pad={item: 1 for item in ['b', 'l', 'r', 't']},  # top margin in px
                steps=[dict(method='restyle',
                            label=counts[i],
                            args=['visible', [j == i for j in range(len(quantiles))]]) for i in range(len(quantiles))]
            )],
            xaxis=dict(visible=False, range=[stretch_factor * data_df['x'].min(), stretch_factor * data_df['x'].max()]),
            yaxis=dict(visible=False, range=[stretch_factor * data_df['y'].min(), stretch_factor * data_df['y'].max()]),
        ))
    plot(figure, filename=output_file_name, auto_open=False)

    logging.info('total time: {:5.2f}s'.format(time() - time_start))
