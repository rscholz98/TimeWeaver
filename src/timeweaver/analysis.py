import pandas as pd
import numpy as np
import re
from collections import defaultdict
import plotly.graph_objects as go


def get_summary_characters(self, set_regex_pattern=None):
        """
        Generates a summary DataFrame containing various statistics for each column in the original DataFrame.

        :param full_summary: If True, includes additional statistics such as unique values count, most frequent value, and more.
        :type full_summary: bool
        :return: A DataFrame containing summary statistics for each column.
        :rtype: pd.DataFrame

        **Example Usage**:

        .. code-block:: python

            summary_df = tw.get_summary(full_summary=True)
            print(summary_df)
        """
        
        if self.evaluation_dataframe is not None:
            self.log("Use the get_summary() method before the evaluation", overwrite=False)
            return

        summary = defaultdict(lambda: defaultdict(int))
        numeric_pattern = re.compile(r'\d')  # Pattern to match numeric characters

        if set_regex_pattern is None:
            regex_pattern = re.compile(r'\D')
        else:
            regex_pattern = re.compile(set_regex_pattern)

        # Iterate through each column in the DataFrame
        for column in self.df.columns:
            summary[column]
            for value in self.df[column]:
                value_str = str(value)
                # Count non-numeric characters
                matches = regex_pattern.findall(value_str)
                for match in matches:
                    summary[column][match] += 1
                
                # Count numeric characters
                numeric_matches = numeric_pattern.findall(value_str)
                summary[column]['Total Numeric'] += len(numeric_matches)

            total_non_numbers = sum(val for key, val in summary[column].items() if key != 'Total Numeric')
            summary[column]['Total Non-Numeric'] = total_non_numbers
            summary[column]['Data Type'] = str(self.df[column].dtype)

        # Prepare the summary DataFrame
        summary_df = pd.DataFrame([{**{'Column': col}, **chars} for col, chars in summary.items()]).fillna(0)
        
        # Reorder columns for readability
        cols = summary_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index('Data Type')))
        cols.insert(2, cols.pop(cols.index('Total Non-Numeric')))
        cols.insert(3, cols.pop(cols.index('Total Numeric')))
        summary_df = summary_df[cols]

        summary_df = summary_df.reset_index(drop=True)
        return summary_df

def get_summary(self, full_summary: bool = False):
        """
        Creates a summary of various statistics for each column in the DataFrame, including counts of numeric and non-numeric cells, NaNs, and zeros. Optionally includes detailed statistics like unique values count, most frequent value, minimum, maximum, mean, and median.

        :param full_summary: Includes detailed statistics if True. Default is False.
        :type full_summary: bool
        :return: A DataFrame summarizing the statistics for each column.
        :rtype: pd.DataFrame

        **Example Usage**:

        .. code-block:: python

            summary_df = tw.get_summary(full_summary=True)
            print(summary_df)

        **Note**:
        The function calculates basic counts by default. Set `full_summary=True` for an extended set of statistics, applicable primarily to numeric columns.
        """
        # Initialize a dictionary to hold summary data
        summary = {
            'Column': [],
            'Data Type': [],
            'Total Numeric Cells': [],
            'Total Non-Numeric Cells': [],
            'Total NaNs': [],
            'Total Zero Values': []
        }

        # Add extra fields to the summary if full_summary is True
        if full_summary:
            summary.update({
                'Unique Values Count': [],
                'Most Frequent Value': [],
                'Minimum Value': [],
                'Maximum Value': [],
                'Mean': [],
                'Median': []
            })

        # Regex pattern to match numeric values (integers, floats, and negative numbers)
        numeric_pattern = re.compile(r'^-?\d+\.?\d*$')

        # Iterate through each column in the DataFrame
        for column in self.df.columns:
            nan_count = self.df[column].isnull().sum()
            numeric_count = 0
            zero_count = 0
            unique_values = self.df[column].nunique(dropna=True)
            most_frequent_value = self.df[column].mode().iloc[0] if not self.df[column].mode().empty else np.nan

            # Check each cell using the regex pattern
            for cell in self.df[column].astype(str):  # Convert cells to string to apply regex
                if pd.isnull(cell):
                    continue  # Already counted NaNs above
                elif numeric_pattern.match(cell):
                    numeric_count += 1
                    if cell == '0' or cell == '-0' or float(cell) == 0.0:
                        zero_count += 1  # Increment zero_count if cell represents 0

            non_numeric_count = self.df[column].size - numeric_count - nan_count

            # Append the general summary data
            summary['Column'].append(column)
            summary['Total Numeric Cells'].append(numeric_count)
            summary['Total Non-Numeric Cells'].append(non_numeric_count)
            summary['Total NaNs'].append(nan_count)
            summary['Data Type'].append(self.df[column].dtype)
            summary['Total Zero Values'].append(zero_count)

            if full_summary:
                summary['Unique Values Count'].append(unique_values)
                summary['Most Frequent Value'].append(most_frequent_value)

                # Only calculate these stats for numeric columns
                if self.df[column].dtype in ['int64', 'float64']:
                    summary['Minimum Value'].append(self.df[column].min())
                    summary['Maximum Value'].append(self.df[column].max())
                    summary['Mean'].append(self.df[column].mean())
                    summary['Median'].append(self.df[column].median())
                else:
                    # Fill with NaN for non-numeric columns
                    summary['Minimum Value'].append(np.nan)
                    summary['Maximum Value'].append(np.nan)
                    summary['Mean'].append(np.nan)
                    summary['Median'].append(np.nan)

        # Convert the summary dictionary into a DataFrame
        summary_df = pd.DataFrame(summary)
        return summary_df

def get_rate_analysis(self, start_rate: float = 0.1, end_rate: float = 0.5, step: float = 0.01, prints: bool = False):
        """
        Analyzes interpolation method performance across a range of missing data rates.

        :param start_rate: The starting missing rate for evaluation.
        :type start_rate: float
        :param end_rate: The ending missing rate for evaluation.
        :type end_rate: float
        :param step: The step size between evaluated missing rates.
        :type step: float
        :param prints: If True, prints log messages during the analysis.
        :type prints: bool

        **Example Usage**:

        .. code-block:: python

            tw.get_rate_analysis(start_rate=0.1, end_rate=0.5, step=0.05, prints=True)
        """
        missing_rates = np.arange(start_rate, end_rate, step)
        all_results = []

        for rate in missing_rates:
            rate = round(rate, 4)
            self.log(f"Evaluating missing rate: {rate}", overwrite=True)
            self.evaluate(missing_rate=rate, prints=prints)
            self.evaluation_dataframe['missing_rate'] = rate
            all_results.append(self.evaluation_dataframe.copy()) # Ensure to copy the dataframe

        # Reset evaluation_dataframe for future evaluations
        self.evaluation_dataframe = None
        results_df = pd.concat(all_results)

        melted_df = results_df.reset_index().melt(id_vars=['index', 'missing_rate'], var_name='method', value_name='performance')
        features = melted_df['index'].unique()

        for feature in features:
            fig = go.Figure()
            feature_df = melted_df[melted_df['index'] == feature]
            methods = feature_df['method'].unique()
            
            for method in methods:
                method_df = feature_df[feature_df['method'] == method]
                fig.add_trace(go.Scatter(x=method_df['missing_rate'], y=method_df['performance'],
                                         mode='lines+markers',
                                         name=method))

            fig.update_layout(title=f'Performance Development for Feature {feature}',
                              xaxis_title='Missing Rate',
                              yaxis_title='Difference between original and interpolated data',
                              legend_title='Method')
            fig.show()