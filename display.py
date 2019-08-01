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

    # todo move the output file name to settings
    output_file_name = input_file.replace('.csv', '.html')
    mode_ = 'text'  # 'markers+text'
    do_basic = True
    if do_basic:
        # todo only plot the most important words or the most popular words
        data = [go.Scatter(hoverinfo='none',
                           marker=dict(line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1), opacity=0.8,
                                       size=6),
                           mode=mode_, text=data_df['word'],
                           x=data_df['x'], y=data_df['y'])]

        layout = go.Layout(margin=dict(l=0, t=0, r=0, b=0))
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=output_file_name, auto_open=False)

    print('total time: {:5.2f}s'.format(time() - time_start))
