import pandas as pd
import numpy as np
import sys
from collections import defaultdict
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
import warnings


class TimeWeaver:
    def __init__(self, df: pd.DataFrame, tracking_column: str):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Error: df must be a pandas DataFrame")

        if not isinstance(tracking_column, str):
            raise ValueError("Error: tracking_column must be a string")
        if tracking_column not in df.columns:
            raise ValueError(
                f"Error: tracking_column '{tracking_column}' does not exist in the DataFrame columns"
            )

        # Initial Dataframe
        self.df = df
        # Time / Tracking column
        self.tracking_column = tracking_column
        # Reuslts of Method performance
        self.evaluation_dataframe = None
        # Method worked : True / False
        self.method_success_dataframe = None

    def log(self, message, overwrite=True):

        start_symbol = "➤"
        end_symbol = "✔"
        pretty_message = f"{start_symbol} {message} {end_symbol}"

        if overwrite:
            # Use \r to return to the start of the line, then overwrite with the pretty message
            sys.stdout.write("\r" + pretty_message + " " * (80 - len(pretty_message)))
            # Clear to end of line
            sys.stdout.flush()
        else:
            # Simply print the pretty message on a new line
            print("\n" + pretty_message)

    def get_summary_characters(self, set_regex_pattern=None):
        
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
    
    def preprocess(self):
        """
        Preprocesses the DataFrame by dropping specified columns, converting data to numeric,
        replacing specified characters with NaN, and checking continuity and reindexing.
        """

    def fill_edge_nans(self, df: pd.DataFrame):
        
        for column_name in df.columns:
            # Forward fill (ffill) to handle NaNs at the start
            df[column_name] = df[column_name].ffill()
            
            # Backward fill (bfill) to handle NaNs at the end
            df[column_name] = df[column_name].bfill()
        
        return df

    def evaluate(
            self, missing_rate=0.1, random: bool = True, prints: bool = True, logging: bool = False
        ):
            
            if logging is False:
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)

            methods = [
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
            {"name": "linear"},
            # {"name": "time"}, # used for daily data
            # {"name": "index"}, # index is same as linear in evenly spaced data
            # {"name": "pad"}, # use backfill
            # Passed to scipy.interpolate.interp1d
            {"name": "nearest"},
            {"name": "zero"},
            {"name": "slinear"},
            {"name": "quadratic"},
            {"name": "cubic"},
            #{"name": "Baum"},
            # {"name": "barycentric"},
            {"name": "polynomial", "order": 1},
            {"name": "polynomial", "order": 2},
            {"name": "polynomial", "order": 3},
            {"name": "polynomial", "order": 5},
            {"name": "polynomial", "order": 7},
            {"name": "polynomial", "order": 9},
            # Wrappers around the SciPy interpolation methods of similar names
            # {"name": "krogh"},
            {"name": "piecewise_polynomial"},
            {"name": "spline", "order": 1},
            {"name": "spline", "order": 2},
            {"name": "spline", "order": 3},
            {"name": "spline", "order": 4},
            {"name": "spline", "order": 5},
            {"name": "akima"},
            {"name": "cubicspline"},
            # Refers to scipy.interpolate.BPoly.from_derivatives.
            {"name": "from_derivatives"},
            # Add other methods as needed
            ]

            ### Initialize results dictionary with empty values for each method and column ###
            results = {}
            method_success = {} 
            for method in methods:
                if "order" in method:
                    key = f"{method['name']}_order_{method['order']}"
                else:
                    key = method["name"]
                results[key] = {column: None for column in self.df.columns if column != self.tracking_column}
                method_success[key] = {column: True for column in self.df.columns if column != self.tracking_column} # Initialize all as True

            if prints:
                self.log("Initializing...", overwrite=True)

            for method in methods:
                if prints:
                    self.log(f"Evaluating method: {method['name']}", overwrite=True)
                for column in self.df.columns:

                    if column == self.tracking_column:
                        continue

                    ########## Fill NaNs at the edges ##########
                    temp_df = self.df.copy()
                    temp_df = self.fill_edge_nans(temp_df)
                    ############################################

                    np.random.seed(0)
                    # Only choose valid indices to remove that are not NaN already
                    valid_indices = temp_df[temp_df[column].notna()].index.tolist()

                    if random:
                        remove_indices = np.random.choice(
                            valid_indices,
                            size=int(len(valid_indices) * missing_rate),
                            replace=False,
                        )
                    else:
                        num_indices_to_remove = int(len(valid_indices) * missing_rate)
                        step_size = len(valid_indices) // num_indices_to_remove
                        remove_indices = valid_indices[::step_size][:num_indices_to_remove]

                    # Set Nans for the selected indices
                    temp_df.loc[remove_indices, column] = np.nan
                    temp_df[column] = pd.to_numeric(temp_df[column], errors='coerce')

                    try:
                        if "order" in method:
                            interpolated_series = temp_df[column].interpolate(method=method["name"], order=method["order"])
                            method_key = f"{method['name']}_order_{method['order']}"
                        else:
                            interpolated_series = temp_df[column].interpolate(method=method["name"])
                            method_key = method["name"]

                        if interpolated_series.isna().any():
                            if logging:
                               self.log(f"Method {method['name']} generates NaNs on column {column}", overwrite=False)
                               #print(interpolated_series.isna().sum())
                            results[method_key][column] = np.nan
                            method_success[method_key][column] = False # Mark as False if interpolation breaks
                        
                        else:
                            temp_df[column] = interpolated_series

                            diff = temp_df.loc[remove_indices, column] - self.df.loc[remove_indices, column]
                            results[method_key][column] = round(np.mean(diff.abs()), 4)
                            method_success[method_key][column] = True # Mark as True if interpolation breaks

                    except Exception as e:
                        if logging:
                            self.log(f"Error with method {method['name']}: {str(e)} on column {column}", overwrite=False)
                        method_success[method_key][column] = False # Mark as False if interpolation breaks
                        continue

            self.evaluation_dataframe = pd.DataFrame(results)
            self.method_success_dataframe = pd.DataFrame(method_success) 
            if prints:
                self.log("Evaluation complete.", overwrite=True)
                self.log(f"Evaluated number of methods: {len(methods)}", overwrite=False)

    def highlight_lowest(self, s):
        """
        Highlights the lowest non-negative values in the DataFrame.
        """
        # Filter out negative values and find the minimum of the remaining values
        non_negative_values = s[s >= 0]
        if not non_negative_values.empty:  # Check if there are any non-negative values
            min_value = non_negative_values.min()
        else:
            min_value = None
        
        # Highlight the cell if it is the lowest non-negative value
        return ["background-color: green" if v == min_value else "" for v in s]
    
    def highlight_false(self, s):
   
        return ['background-color: #FF4500' if v is False else '' for v in s]

    def get_evaluation_dataframe(self, style: bool = True):
        """
        Applies styling to the results DataFrame to highlight the lowest non-negative values in each row.
        """
        if self.evaluation_dataframe is not None:
            if style:
                # Apply the highlight_lowest function across rows
                return self.evaluation_dataframe.style.apply(
                    self.highlight_lowest, axis=1  # Ensure function is applied row-wise
                )
            else:
                return self.evaluation_dataframe
        else:
            self.log(
                "Results DataFrame is not available. Please run evaluate() first.",
                overwrite=False,
            )
    
    def get_method_success_dataframe(self, style: bool = True):
        if self.method_success_dataframe is not None:
            if style:
                return self.method_success_dataframe.style.apply(
                    self.highlight_false, axis=1
                )
            else:
                return self.method_success_dataframe
        else:
            self.log(
                "Method Success DataFrame is not available. Please run evaluate() first.",
                overwrite=False,
            )
        
    def get_best(self, optimized_selection: bool = True):
        if self.evaluation_dataframe is not None:

            if optimized_selection is False:
                best_methods_per_index ={}

                for index, row in self.evaluation_dataframe.iterrows():
                    min_value = row.min()
                    best_columns = row[row == min_value].index.tolist()
                    best_methods_per_index[index] = best_columns

                return best_methods_per_index
            else:
                best_methods_per_index = {}
                method_frequency = {}

                # Step 1: Collect all methods tied for the lowest value for each index
                for index, row in self.evaluation_dataframe.iterrows():
                    min_value = row.min()
                    best_columns = row[row == min_value].index.tolist()
                    best_methods_per_index[index] = best_columns
                    
                    for method in best_columns:
                        if method in method_frequency:
                            method_frequency[method] += 1
                        else:
                            method_frequency[method] = 1

                # Step 2: Create a list to store the selected method for each index
                optimized_selection = []

                # Iterate through the best_methods_per_index to fill the optimized_selection
                for index, methods in best_methods_per_index.items():
                    if methods:  # Ensure there are methods to choose from
                        sorted_methods = sorted(methods, key=lambda x: (-method_frequency[x], x))
                        selected_method = sorted_methods[0]
                        optimized_selection.append((index, selected_method))

                # Return a list of tuples (index, selected_method)
                return optimized_selection
        else:
            self.log(
                "Results DataFrame is not available. Please run the evaluate method first.",
                overwrite=False,
            )
            return None
        
    def get_rate_analysis(self, start_rate: float = 0.1, end_rate: float = 0.5, step: float = 0.01, prints: bool = False):
        missing_rates = np.arange(start_rate, end_rate, step)
        all_results = []

        for rate in missing_rates:
            rate = round(rate, 4)
            self.log(f"Evaluating missing rate: {rate}", overwrite=False)
            self.evaluate(missing_rate=rate, prints=prints)
            self.evaluation_dataframe['missing_rate'] = rate
            all_results.append(self.evaluation_dataframe.copy()) # Ensure to copy the dataframe

        # Reset evaluation_dataframe for future evaluations
        self.evaluation_dataframe = None
        results_df = pd.concat(all_results)

        # Process results for plotting
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
    





