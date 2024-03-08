import pandas as pd
import numpy as np
import sys
from collections import defaultdict
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go


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

        self.df = df
        self.tracking_column = tracking_column
        self.results_df = None
        self.evaluation_dataframe = None

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
        
        if hasattr(self, 'evaluation_dataframe') and self.evaluation_dataframe is not None:
            # Assuming self.log is a method defined elsewhere in your class for logging messages
            self.log("Use the get_summary() method before the evaluation", overwrite=False)
            return

        summary = defaultdict(lambda: defaultdict(int))
        numeric_pattern = re.compile(r'\d')  # Pattern to match numeric characters

        # Use the provided regex pattern for non-numeric characters or default to matching any non-digit character
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
    
    def get_summary(self):
        # Initialize a dictionary to hold summary data
        summary = {
            'Column': [],
            'Total Numeric Cells': [],
            'Total Non-Numeric Cells': [],
            'Total NaNs': [],
            'Data Type': []
        }

        # Iterate through each column in the DataFrame
        for column in self.df.columns:
            numeric_count = 0
            non_numeric_count = 0
            nan_count = 0

            # Iterate through each cell in the column
            for cell in self.df[column]:
                if pd.isnull(cell):
                    nan_count += 1
                elif isinstance(cell, (int, float, np.number)):
                    numeric_count += 1
                else:
                    non_numeric_count += 1

            # Append the counts and column details to the summary
            summary['Column'].append(column)
            summary['Total Numeric Cells'].append(numeric_count)
            summary['Total Non-Numeric Cells'].append(non_numeric_count)
            summary['Total NaNs'].append(nan_count)
            summary['Data Type'].append(self.df[column].dtype)

        # Convert the summary dictionary into a DataFrame
        summary_df = pd.DataFrame(summary)
        return summary_df

    def preprocess(self):
        """
        Preprocesses the DataFrame by dropping specified columns, converting data to numeric,
        replacing specified characters with NaN, and checking continuity and reindexing.
        """

    def evaluate(
        self, missing_rate=0.1, random: bool = True, top_n: int = 3, prints: bool = True
    ):
        """
        Evaluates various interpolation methods by introducing missing data and comparing
        the interpolated results with the original data. Calculates the mean absolute difference.
        """
        methods = [
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
            {"name": "linear"},
            # {"name": "time"} # used for daily data
            # {"name": "index"}, # index is same as linear in evenly spaced data
            # {"name": "pad"}, # use backfill
            # Passed to scipy.interpolate.interp1d
            {"name": "nearest"},
            {"name": "zero"},
            {"name": "slinear"},
            {"name": "quadratic"},
            {"name": "cubic"},
            {"name": "Baum"},
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
        ##################################################################################
        results = {}
        for method in methods:
            if "order" in method:
                key = f"{method['name']}_order_{method['order']}"
            else:
                key = method["name"]
            results[key] = {
                column: -1
                for column in self.df.columns
                if column != self.tracking_column
            }
        if prints:
            self.log("Initializing...", overwrite=True)
        ##################################################################################

        for method in methods:
            if prints:
                self.log(f"Evaluating method: {method['name']}", overwrite=True)
            for column in self.df.columns:
                if column == self.tracking_column:
                    continue

                temp_df = self.df.copy()
                np.random.seed(0)

                # Get indices that are not NaN to ensure we don't replace existing NaNs
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

                temp_df.loc[remove_indices, column] = np.nan
                temp_df.head()

                try:
                    if "order" in method:
                        temp_df[column] = temp_df[column].interpolate(
                            method=method["name"], order=method["order"]
                        )
                        method_key = f"{method['name']}_order_{method['order']}"
                    else:
                        temp_df[column] = temp_df[column].interpolate(
                            method=method["name"]
                        )
                        method_key = method["name"]

                    diff = (
                        temp_df.loc[remove_indices, column]
                        - self.df.loc[
                            remove_indices, column
                        ] 
                    )
                    results[method_key][column] = round(np.mean(diff.abs()), 4)

                except Exception as e:
                    error_message = f"Error with method {method['name']}: {str(e)} on column {column}"
                    if prints:
                        self.log(error_message, overwrite=False)
                    continue

                self.evaluation_dataframe = pd.DataFrame(results)
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

    def display_dataframe(self, style: bool = True):
        """
        Applies styling to the results DataFrame to highlight the lowest values.
        """
        if self.evaluation_dataframe is not None:
            if style:
                return self.evaluation_dataframe.style.apply(
                    self.highlight_lowest, axis=1
                )
            else:
                return self.evaluation_dataframe
        else:
            self.log(
                "Results DataFrame is not available. Please run evaluate() first.",
                overwrite=False,
            )

    def display_list(self, n: int = 1):
        if self.evaluation_dataframe is not None:
            # Initialize an empty list to store the top n methods for each column
            best_methods_per_column = []

            # Transpose the DataFrame for easier iteration over columns (metrics)
            results_transposed = self.evaluation_dataframe.T

            # Iterate over each column (originally rows in `self.results_df`)
            for column in results_transposed.columns:
                # Sort the column to get the top n lowest values (best performance)
                top_methods = results_transposed[column].nsmallest(n).index.tolist()
                best_methods_per_column.append({column: top_methods})

            return best_methods_per_column
        else:
            self.log(
                "Results DataFrame is not available. Please run the evaluate method first.",
                overwrite=False,
            )
            return None


