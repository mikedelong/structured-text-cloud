import json
import logging
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
    do_basic = False
    if do_basic:
        fig = go.Figure(data=[go.Scatter(hoverinfo='none',
                                         marker=dict(line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1),
                                                     opacity=0.8, size=6), mode=mode_, text=data_df['word'],
                                         x=data_df['x'], y=data_df['y']
                                         )],
                        layout=go.Layout(margin=dict(l=0, t=0, r=0, b=0)))
    else:
        length = data_df['count'].nunique()
        fig = go.Figure(data=[go.Scatter(hoverinfo='none',
                                         marker=dict(line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1),
                                                     opacity=0.8, size=6), mode=mode_,
                                         text=data_df[data_df['count'] > index]['word'],
                                         x=data_df[data_df['count'] > index]['x'],
                                         y=data_df[data_df['count'] > index]['y'],
                                         name='level: {}'.format(index)
                                         ) for index in sorted(data_df['count'].unique().tolist())],
                        layout=dict(
                            sliders=[dict(
                                active=min(10, length),
                                pad={'t': 1},
                                steps=[dict(method='restyle', args=['visible', [j == i for j in range(length)]]) for i
                                       in range(length)]
                            )]
                        ))
    plot(fig, filename=output_file_name, auto_open=False)

    print('total time: {:5.2f}s'.format(time() - time_start))
