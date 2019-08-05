import numpy as np
from plotly.graph_objects import Figure
from plotly.graph_objects import Scatter
from plotly.offline import plot

if __name__ == '__main__':
    # Create figure
    fig = Figure()

    # Add traces, one for each slider step
    length = len(list(np.arange(0, 5, 0.1)))
    for index, step in enumerate(np.arange(0, 5, 0.1)):
        fig.add_trace(Scatter(
            visible=False if index != 10 else True,
            line=dict(color="#00CED1", width=6),
            name="nu = " + str(step),
            x=np.arange(0, 10, 0.01),
            y=np.sin(step * np.arange(0, 10, 0.01))))

    # Make 10th trace visible
    fig.data[10].visible = True

    # Create and add slider
    steps = []
    for i in range(length):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=[dict(method='restyle', args=['visible', [False if i != j else True for j in range(length)]]) for i
               in range(length)]
    )]

    fig.update_layout(
        sliders=sliders
    )

    # fig = go.Figure(data=[
    #     # fig.add_trace(
    #     go.Scatter(
    #         visible=False if index != 10 else True,
    #         line=dict(color="#00CED1", width=6),
    #         name="nu = " + str(step),
    #         x=np.arange(0, 10, 0.01),
    #         y=np.sin(step * np.arange(0, 10, 0.01)))
    #
    #     for index, step in enumerate(np.arange(0, 5, 0.1))
    #
    # ], layout=[dict(sliders=dict(
    #     active=10,
    #     currentvalue={"prefix": "Frequency: "},
    #     pad={"t": 50},
    #     steps=[dict(method='restyle', args=['visible', [False if i != j else True for j in range(length)]]) for i
    #            in range(length)]
    # )
    # )])
    output_file_name = './demo_plotly_slider.html'
    plot(fig, filename=output_file_name, auto_open=False)

    # fig.show()