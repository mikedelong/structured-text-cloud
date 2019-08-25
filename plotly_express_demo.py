import plotly.express as px
from plotly.offline import plot

gapminder = px.data.gapminder().query("year==2007 and continent=='Americas'")

fig = px.scatter(gapminder, x="gdpPercap", y="lifeExp", text="country", log_x=True, size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(
    height=800,
    title_text='GDP and Life Expectancy (Americas, 2007)'
)

plot(fig, filename='./plotly_express.demo.html', auto_open=False)
