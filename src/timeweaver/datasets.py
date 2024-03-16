import pandas as pd
from importlib import resources
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataSets:
    @classmethod
    def PRSA(cls):
        """
        Beijing PM2.5 Data: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
        """
        
        try:
            with resources.path('timeweaver.data', 'PRSA.pkl') as filepath:
                data = pd.read_pickle(filepath)
            return data
        except FileNotFoundError:
            print(filepath)
            print("The file was not found.")
            return None
        except Exception as e:
            print(filepath)
            print(f"An error occurred: {e}")
            return None
        
    @classmethod
    def MachineProcess(cls):
        """
        Examplary Machine Process Data. 
        """
        try:
            with resources.path('timeweaver.data', 'MachineProcess.pkl') as filepath:
                data = pd.read_pickle(filepath)
            return data
        except FileNotFoundError:
            print(filepath)
            print("The file was not found.")
            return None
        except Exception as e:
            print(filepath)
            print(f"An error occurred: {e}")
            return None
        
    @classmethod
    def AirQuality(cls):
        """
        Examplary Machine Process Data. 
        """
        try:
            with resources.path('timeweaver.data', 'AirQuality.pkl') as filepath:
                data = pd.read_pickle(filepath)
            return data
        except FileNotFoundError:
            print(filepath)
            print("The file was not found.")
            return None
        except Exception as e:
            print(filepath)
            print(f"An error occurred: {e}")
            return None
        
    @classmethod
    def plot_data(
        cls,
        df: pd.DataFrame,
        title: str = "Dataframe - Feature Overview",
        subplot_num_columns: int = 6,
        subplot_title_font_size: int = 11,
    ) -> go.Figure:
        """
        Plot DataFrame columns as line charts in subplots using Plotly.
        """
        subplot_num_rows = -(-len(df.columns) // subplot_num_columns)  # Ceiling division for partial row

        fig = make_subplots(
            rows=subplot_num_rows,
            cols=subplot_num_columns,
            subplot_titles=df.columns,
            vertical_spacing=0.1,  # Adjusted vertical spacing
        )

        row, col = 1, 1

        for index, column in enumerate(df.columns):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode="lines",
                    name=column,
                ),
                row=row,
                col=col,
            )

            col += 1
            if col > subplot_num_columns:
                col = 1
                row += 1

        fig.update_layout(
            height=300 * subplot_num_rows,
            width=1400,
            title_text=title,
            showlegend=False,
        )

        # Update font size for all subplot annotations
        for annotation in fig.layout.annotations:
            annotation.font.size = subplot_title_font_size

        # Update axes titles for the last row and the first column
        for i in range(1, subplot_num_columns + 1):
            fig.update_xaxes(title_text="Index", row=subplot_num_rows, col=i)
        for i in range(1, subplot_num_rows + 1):
            fig.update_yaxes(title_text="Value", col=1, row=i)

        return fig