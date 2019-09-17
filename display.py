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

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    if input_file is None:
        logging.error('No input file specified. Quitting.')
        quit(1)

    part_of_speech_file = settings['part_of_speech_file'] if 'part_of_speech_file' in settings.keys() else None
    if part_of_speech_file is None:
        logging.error('No part of speech file specified. Quitting.')
        quit(2)

    max_words_to_show = settings['max_words_to_show'] if 'max_words_to_show' in settings.keys() else None
    if max_words_to_show is None:
        max_words_to_show = 300
        logging.error('Max words to show not set, defaulting to default value: {}'.format(max_words_to_show))

    parser = WiktionaryParser()

    data_df = pd.read_csv(input_file)
    # we need to squeeze out spaces from the column names before we proceed
    data_df.rename(columns={item: item.strip() for item in list(data_df)}, inplace=True)
    logging.info('column names after load and rename: {}'.format(list(data_df)))

    output_file_name = settings['output_file'] if 'output_file' in settings.keys() else None
    if output_file_name is None:
        logging.error('No output file specified. Quitting.')
        quit(1)
    mode_ = 'text'  # 'markers+text'
    # todo move the slice strategy to a setting
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

    # we only need to create the part of speech data if it does not already exist
    if 'part_of_speech' not in list(data_df):
        known_part_of_speech_df = pd.read_csv(part_of_speech_file, usecols=['word', 'part_of_speech'])
        known_part_of_speech = {row['word']: row['part_of_speech'] for _, row in known_part_of_speech_df.iterrows()}
        # known_part_of_speech = {}
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

    which_color_map = 'uniform'  # was 'uniform' / 'cumsum'
    # todo move the colormap name to a setting
    colormap = cm.get_cmap('jet')
    do_original = False
    if do_original:
        if which_color_map == 'cumsum':
            # use the cumsum of the value counts to assign a color from the colormap by hex string
            part_of_speech_color_map = data_df['part_of_speech'].value_counts(normalize=True).cumsum(
            ).apply(lambda x: float_color_to_hex(x, colormap)).to_dict()
        elif which_color_map == 'uniform':
            # use evenly-spaced colors from a colormap
            part_of_speech_color_map = {data_df['part_of_speech'].unique()[index]: float_color_to_hex(
                np.linspace(0, 1, data_df['part_of_speech'].nunique())[index], colormap) for index in
                range(data_df['part_of_speech'].nunique())}
        else:
            raise NotImplementedError('color map: {}'.format(which_color_map))
    else:
        # https://stackoverflow.com/questions/60208/replacements-for-switch-statement-in-python?rq=1
        supported_color_maps = {'cumsum', 'uniform'}
        if which_color_map in supported_color_maps:
            part_of_speech_color_map = {
                'cumsum': data_df['part_of_speech'].value_counts(normalize=True).cumsum(
                ).apply(lambda x: float_color_to_hex(x, colormap)).to_dict(),
                'uniform': {data_df['part_of_speech'].unique()[index]: float_color_to_hex(
                    np.linspace(0, 1, data_df['part_of_speech'].nunique())[index], colormap) for index in
                    range(data_df['part_of_speech'].nunique())}
            }[which_color_map]
        else:
            raise NotImplementedError('color map: {}'.format(which_color_map))

    logging.info('color map {}: {}'.format(which_color_map, part_of_speech_color_map))
    data_df['color'] = data_df['part_of_speech'].map(part_of_speech_color_map)

    # get the cut level
    cut_level = data_df.nlargest(n=max_words_to_show, columns=['count'], keep='all')['count'].min()
    data_df = data_df[data_df['count'] >= cut_level]
    logging.info('our cut level of {} leaves us with {} rows/words to display'.format(max_words_to_show, len(data_df)))

    color_ = 'rgba(217, 217, 217, 0.14)'
    interpolation_ = 'lower'
    opacity_ = 0.8
    size_ = 6
    stretch_factor = 1.05
    width_ = 0.1
    fig = go.Figure(data=[go.Scatter(
        hovertext=get_quantile(data_df, 'cumulative', quantile, interpolation_)['count'],
        marker=dict(line=dict(color=color_, width=width_), opacity=opacity_, size=size_), mode=mode_,
        text=get_quantile(data_df, 'cumulative', quantile, interpolation_)['word'],
        textfont=dict(color=get_quantile(data_df, 'cumulative', quantile, interpolation_)['color']),
        # , colorscale='Viridis'),
        x=get_quantile(data_df, 'cumulative', quantile, interpolation_)['x'],
        y=get_quantile(data_df, 'cumulative', quantile, interpolation_)['y'],
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
