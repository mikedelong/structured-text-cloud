import json
import logging
from collections import Counter
from time import time

import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

if __name__ == '__main__':
    time_start = time()
    with open('./display.json') as settings_fp:
        settings = json.load(settings_fp)
        print(settings)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    if input_file is None:
        logging.error('No input file specified. Quitting.')
        quit(1)

    data_df = pd.read_csv(input_file)

    output_file_name = settings['output_file'] if 'output_file' in settings.keys() else None
    if output_file_name is None:
        logging.error('No output file specified. Quitting.')
        quit(1)
    mode_ = 'text'  # 'markers+text'
    slice_limit = 10
    do_basic = False
    if do_basic:
        fig = go.Figure(data=[go.Scatter(hoverinfo='none',
                                         marker=dict(line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1),
                                                     opacity=0.8, size=6), mode=mode_, text=data_df['word'],
                                         x=data_df['x'], y=data_df['y']
                                         )],
                        layout=go.Layout(margin=dict(l=0, t=0, r=0, b=0)))
    else:
        slices = sorted(data_df['count'].unique().tolist())

        # if we have more than ten slices refit to ten
        if len(slices) > slice_limit:
            total = sum(slices)
            counts = Counter(data_df['count'].values.tolist())
            data_df = data_df.sort_values(by=['count'], axis=0, ascending=True)
            data_df['cumulative'] = data_df['count'].cumsum()
            quantiles = [1.0 / float(slice_limit) * index - 1.0 / (2.0 * float(slice_limit)) for index in
                         range(1, slice_limit + 1)]
            pass
        else:
            quantiles = [float(index) / float(len(slices)) for index in range(1, len(slices))]
        interpolation_ = 'lower'
        fig = go.Figure(data=[go.Scatter(
            # hoverinfo='none',
            marker=dict(line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1),
                        opacity=0.8, size=6), mode=mode_,
            # text=data_df[data_df['count'] > index]['word'],
            text=
            data_df[data_df['cumulative'] > data_df['cumulative'].quantile(q=quantile, interpolation=interpolation_)][
                'word'],
            # x=data_df[data_df['count'] > index]['x'],
            x=data_df[data_df['cumulative'] > data_df['cumulative'].quantile(q=quantile, interpolation=interpolation_)][
                'x'],
            # y=data_df[data_df['count'] > index]['y'],
            y=data_df[data_df['cumulative'] > data_df['cumulative'].quantile(q=quantile, interpolation=interpolation_)][
                'y'],
            # hovertext=data_df[data_df['count'] > index]['count'],
            hovertext=
            data_df[data_df['cumulative'] > data_df['cumulative'].quantile(q=quantile, interpolation=interpolation_)][
                'count'],
            name='level: {}'.format(index)
        ) for index, quantile in enumerate(quantiles)],
            layout=dict(
                sliders=[dict(
                    active=min(5, len(quantiles)),
                    pad={'t': 1},
                    # steps=[dict(method='restyle', args=['visible', [j == i for j in range(len(slices))]]) for i  in range(len(slices))]
                    steps=[dict(method='restyle', args=['visible', [j == i for j in range(len(quantiles))]]) for i in
                           range(len(quantiles))]
                )]
            ))
    plot(fig, filename=output_file_name, auto_open=False)

    print('total time: {:5.2f}s'.format(time() - time_start))
