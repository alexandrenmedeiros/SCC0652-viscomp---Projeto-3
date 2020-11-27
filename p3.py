import numpy as np
import pandas as pd

from ipywidgets import VBox, HBox, widgets
import plotly.graph_objects as go
import plotly.express as px

MAX_THRESHOLD = 300
INITIAL_THRESHOLD = 50

LINE_SIZE = (1800, 400)
BAR_SIZE = (800, 500)
SCT_SIZE = BAR_SIZE

class App(object):
    """ Dashboard class that does a visualization on the
        "Video Games Sales with Ratings" dataset
    """

    def __init__(self, data: pd.DataFrame):
        """ Class constructor
            
            Creates all plots
        """

        self.data = data

        # set threshold and sliced data
        self.threshold = widgets.IntSlider(
            value=MAX_THRESHOLD, min=1, max=MAX_THRESHOLD, step=1,
            description=' Threshold: ', readout_format='d',
            disabled=False, continuous_update=False,
        )
        self.threshold.observe(self.on_threshold_change, 'value')
        self.selected_data = self.data[:self.threshold.value]

        # --- CREATE PLOTS ---

        # bar plot for categorical variables
        self.cat_labels = ['Genre', 'Platform', 'Rating','Year_of_Release'] #, 'Publisher', 'Developer']

        self.cat_plots = []
        for i in range(0, len(self.cat_labels)):
            self.cat_plots.append(self.bar_plot(x_axis=self.cat_labels[i], y_axis='Global_Sales', figsize=BAR_SIZE))

        self.cat_tab = widgets.Tab(self.cat_plots)
        for i in range(0, len(self.cat_labels)):
            self.cat_tab.set_title(i, self.cat_labels[i])

        # Line plots for sales variables
        self.sales_x_axis = []
        for i in range(1, self.threshold.value + 1):
            self.sales_x_axis.append(str(i) + ' - ' + self.selected_data['Name'][i - 1] + ' - ' + self.selected_data['Platform'][i - 1])
        self.sales_y_labels = ['Global_Sales', 'NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales']

        self.sales_lines = self.line_plots(self.sales_x_axis, self.sales_y_labels, figsize=LINE_SIZE)
        # self.sales_doughnut = self.

        # scatter plot for score variables
        self.score_color = widgets.Dropdown(
            options=['Genre', 'Platform', 'Rating', 'Publisher'],
            value='Genre',
            description='Color by:',
            disabled=False,
        )
        self.score_color.observe(self.on_color_change, 'value')

        self.score_plots = [] 
        self.score_plots.append(self.scatter_plot('Critic_Score', 'Global_Sales', remove=True, remove_labels=['Critic_Count'], figsize=SCT_SIZE, log_y=True))
        self.score_plots.append(self.scatter_plot('User_Score', 'Global_Sales', remove=True, remove_labels=['User_Count'], figsize=SCT_SIZE, log_y=True))
        self.score_plots.append(self.scatter_plot('Critic_Score', 'User_Score', remove=True, remove_labels=['User_Count', 'Critic_Count'], figsize=SCT_SIZE))

        self.score_tab = widgets.Tab(self.score_plots)
        self.score_tab.set_title(0, 'Sales x Critic Score')
        self.score_tab.set_title(1, 'Sales x User Score')
        self.score_tab.set_title(2, 'User x Critic Score')

        # Add selection between graphs
        self.selected_sales = []
        for i in self.sales_lines.data:
            i.on_selection(self.selec_in_line)
            i.on_deselect(self.deselection)

        self.selected_cat = []
        for i in self.cat_plots:
            i.data[0].on_selection(self.selec_in_cat)
            i.data[0].on_deselect(self.deselection)

        self.selected_score = []
        for i in self.score_plots:
            i.data[0].on_selection(self.selec_in_score)
            i.data[0].on_deselect(self.deselection)

    def index_converter(self, index_list, remove_labels):
        """ Function that convert index from original data to removed data
        """
        if index_list == None:
            return None
        
        converted_index = []
        alt_index = -1
        for index, i in self.selected_data.iterrows():
            counts = True
            for label in remove_labels:
                if(i[label] == 0):
                    counts = False
            if (counts):
                alt_index += 1

            for j in index_list:
                if(j == index and counts):
                    converted_index.append(alt_index)

        return converted_index

    def change_selected_points(self, points_list):

        with self.sales_lines.batch_animate():
            for i in self.sales_lines.data:
                i.selectedpoints = points_list

        # select the same data in the bar plots
        for i in range(0, len(self.cat_plots)):
            with self.cat_plots[i].batch_animate():
                for i in self.cat_plots[i].data:
                    i.selectedpoints = points_list

        with self.score_plots[0].batch_animate():
            # for i in self.score_plots[0].data:
            self.score_plots[0].data[0].selectedpoints = self.index_converter(points_list, ['Critic_Count'])
        
        with self.score_plots[1].batch_animate():
            self.score_plots[1].data[0].selectedpoints = self.index_converter(points_list, ['User_Count'])
            
        with self.score_plots[2].batch_animate():
            self.score_plots[2].data[0].selectedpoints = self.index_converter(points_list, ['User_Count','Critic_Count'])

    def deselection(self, trace, points):
        self.change_selected_points(None)

    def selec_in_line(self, trace, points, selector):
        if points.trace_index == 0:
            self.selected_sales = []

        # append selected points
        for i in points.point_inds:
            if i != None:
                self.selected_sales.append(i)

        # does the selection on other graphs on the last trace call
        if points.trace_index == 4:

            self.selected_sales = list(set(self.selected_sales)) # get unique elements
            if len(self.selected_sales) == 0: # select all elements if none is select
                self.selected_sales = None

            self.change_selected_points(self.selected_sales)

    def selec_in_cat(self, trace, points, selector):
        self.selected_cat = []

        # append selected points
        for i in points.point_inds:
            if i != None:
                self.selected_cat.append(i)

        self.selected_cat = list(set(self.selected_cat)) # get unique elements
        if len(self.selected_cat) == 0: # select all elements if none is select
            self.selected_cat = None

        self.change_selected_points(self.selected_cat)

    def selec_in_score(self, trace, points, selector):
        self.selected_score = []

        # append selected points
        for i in points.point_inds:
            if i != None:
                plat = trace.customdata[i, 1]
                name = trace.hovertext[i]
                j = self.selected_data.loc[(self.selected_data['Name'] == name) & (self.selected_data['Platform'] == plat)]
                self.selected_score.append(j.iloc[0, 0])

        self.selected_score = list(set(self.selected_score))  # get unique elements
        if len(self.selected_score) == 0:  # select all elements if none is select
            self.selected_score = None

        self.change_selected_points(self.selected_score)

    def color_scatter(self, plot_fig, new_color, remove=False, remove_labels=None):
        colors = self.generate_colors(new_color, remove=remove, remove_labels=remove_labels)

        with plot_fig.batch_animate():
            for i in plot_fig.data:
                i.marker.color = colors

    def on_color_change(self, color):
        self.color_scatter(self.score_plots[0], color['new'], remove=True, remove_labels=['Critic_Count'])
        self.color_scatter(self.score_plots[1], color['new'], remove=True, remove_labels=['User_Count'])
        self.color_scatter(self.score_plots[2], color['new'], remove=True, remove_labels=['User_Count','Critic_Count'])

    def clean_data(self, data, remove=False, remove_labels=None):
        """ Function that may remove elements from the data that are 
            equal to 0 on the selected labels
        """

        df = data
        if(remove):
            for i in remove_labels:
                df = df[df[i] != 0]
            
        return df

    def animate_bar(self, plot_fig):
        with plot_fig.batch_animate():
            for i in plot_fig.data:
                i.x = self.selected_data[plot_fig.layout.xaxis.title.text]
                i.y = self.selected_data[plot_fig.layout.yaxis.title.text]

    def animate_lines(self, plot_fig):
        with plot_fig.batch_animate():
            count = 0
            for i in plot_fig.data:
                i.x = self.sales_x_axis[:self.threshold.value]
                i.y = self.selected_data[self.sales_y_labels[count]]
                count += 1

    def animate_scatter(self, plot_fig, remove=False, remove_labels=None):
        df = self.clean_data(self.selected_data, remove=remove, remove_labels=remove_labels)
        colors = self.generate_colors(self.score_color.value, remove=remove, remove_labels=remove_labels)

        with plot_fig.batch_animate():
            for i in plot_fig.data:
                i.x = df[plot_fig.layout.xaxis.title.text]
                i.y = df[plot_fig.layout.yaxis.title.text]
                i.marker.color = colors

    def on_threshold_change(self, threshold):
        """ Callback function when threshold changes
            It animates all plots
        """

        self.selected_data = self.data[:threshold['new']]
        # change categorical plots
        for i in range(0, len(self.cat_labels)):
            self.animate_bar(self.cat_plots[i])
        # change line plots
        self.animate_lines(self.sales_lines)
        # change score plots
        self.animate_scatter(self.score_plots[0], remove=True, remove_labels=['Critic_Count'])
        self.animate_scatter(self.score_plots[1], remove=True, remove_labels=['User_Count'])
        self.animate_scatter(self.score_plots[2], remove=True, remove_labels=['User_Count','Critic_Count'])

    def generate_colors(self, axis, remove=False, remove_labels=None):
        """ Function that return a color list based on given axis """

        df = self.clean_data(self.selected_data, remove=remove, remove_labels=remove_labels)

        colors = []
        unique_list = list(df[axis].unique())
        for i in df[axis]:
            colors.append(unique_list.index(i))

        return colors

    def bar_plot(self, x_axis, y_axis, flip=True, categorical=True, sort=True, asc=False, figsize=(800, 600)):
        """ Returns a bar plot """

        colors = self.generate_colors(x_axis)

        order = {}
        if (sort):
            order[x_axis] = self.selected_data[[x_axis, y_axis]].groupby(
                x_axis).sum().sort_values(by=[y_axis], ascending=asc).index

        orientation = 'v'
        if (flip):
            aux = x_axis
            x_axis = y_axis
            y_axis = aux
            orientation = 'h'

        y_name = [y_axis for i in range(0, self.threshold.value)]
        x_name = [x_axis for i in range(0, self.threshold.value)]
            
        fig_plot = go.FigureWidget(px.bar(
            self.selected_data, x=x_axis, y=y_axis,
            orientation=orientation,
            hover_name='Name', 
            custom_data=['Name', x_axis, y_axis, y_name, x_name],
            width=figsize[0], height=figsize[1],
            category_orders=order,
            color=colors, color_continuous_scale='viridis',
        ))

        if (flip and categorical):
            fig_plot.update_yaxes(type='category')
        elif(not flip and categorical):
            fig_plot.update_xaxes(type='category')

        fig_plot.update_layout(coloraxis_showscale=False)
        fig_plot.update_traces(
            hovertemplate="<br>".join([
                "<b>%{customdata[0]}</b> <br>",
                "%{customdata[3]}: %{customdata[2]}",
                "%{customdata[4]}: %{customdata[1]}",
            ])
        )
        fig_plot.update_layout(margin=go.layout.Margin(l=0, r=25, b=0, t=25))

        return fig_plot

    def line_plots(self, x_axis, y_axis, figsize=(1000, 600)):
        """ Return a figure with several line plots (number of y_axis elements) """

        fig_plot = go.FigureWidget()

        for y in y_axis:
            fig_plot.add_trace(go.Scatter(
                x=x_axis, y=self.selected_data[y], mode='lines+markers', name=y))

        fig_plot.update_layout(width=figsize[0], height=figsize[1])
        fig_plot.update_layout(hovermode="x", xaxis={'visible': False})
        # fig_plot.update_layout(hovermode="x unified", xaxis={'visible':False})
        # fig_plot.update_layout(hovermode="x", xaxis={'visible':False, 'rangeslider':{'visible':True}})
        fig_plot.update_layout(margin=go.layout.Margin(l=0, r=0, b=30, t=0))

        return fig_plot

    def scatter_plot(self, x_axis, y_axis, remove=False, remove_labels=None, figsize=(800, 600), log_y=False):
        """ Returns a Scatter plot
            It may remove elements from the data that are equal to 0 on the selected labels
        """

        colors = self.generate_colors(self.score_color.value, remove=remove, remove_labels=remove_labels)

        df = self.clean_data(self.selected_data, remove=remove, remove_labels=remove_labels)

        fig_plot = go.FigureWidget(px.scatter(
            df, x=x_axis, y=y_axis,
            hover_name='Name', hover_data=['Genre', 'Platform', 'Rating', 'Year_of_Release', 'Publisher', 'Developer'],
            width=figsize[0], height=figsize[1], log_y=log_y,
            color=colors, color_continuous_scale='jet', # viridis
        ))
        
        fig_plot.update_layout(coloraxis_showscale=False)
        fig_plot.update_layout(margin=go.layout.Margin(l=0, r=25, b=0, t=25))

        return fig_plot

    def show(self):
        """ Method that organizes all plots in a specific layout
        """

        self.threshold.value = INITIAL_THRESHOLD
        
        # scatter plot for score variables
        self.white_drop = widgets.Dropdown(
            options=['Genre', 'Platform', 'Rating', 'Publisher'],
            disabled=True,
        )
        self.white_drop.layout.visibility = 'hidden'

        self.layout = VBox([
            self.threshold,
            self.sales_lines,
            HBox([
                VBox([self.white_drop,
                self.cat_tab,]),
                VBox([self.score_color, 
                self.score_tab])
            ])
        ])

        return self.layout


# df = pd.read_csv("filled_Video_Games_Sales.csv")
# # change year_of_release data type of float to int
# df['Year_of_Release'] = df['Year_of_Release'].astype('int')
# app = App(df)
# app.show()
