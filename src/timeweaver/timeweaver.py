import pandas as pd
import numpy as np
import sys
import pandas as pd
import re
import warnings
from timeweaver.preprocessing import DropColumnTransformer, CharactersToNaNTransformer, ContinuityReindexTransformer, EdgeNaNFillerTransformer, ColumnInterpolateTransformer, PreProcessor
from timeweaver.analysis import get_summary_characters, get_summary, get_rate_analysis
from sklearn.pipeline import Pipeline


class TimeWeaver:
    """
    Facilitates data preprocessing, analysis, and evaluation of interpolation methods on pandas DataFrames, particularly for time-series data. 
    It summarizes data characteristics, preprocesses data, and evaluates various interpolation methods under different conditions.

    :param df: The DataFrame to analyze and process.
    :type df: pd.DataFrame
    :param tracking_column: The column name used for tracking, often representing time.
    :type tracking_column: str
    
    **Attributes**:
    - `df` (pd.DataFrame): The initial DataFrame.
    - `tracking_column` (str): The tracking column name.
    - `evaluation_dataframe` (pd.DataFrame or None): Stores interpolation method evaluation results.
    - `method_success_dataframe` (pd.DataFrame or None): Records success or failure of each interpolation method for each column.

    **Example Usage**:

    .. code-block:: python

        import pandas as pd
        # Example DataFrame
        data = {
            'date': pd.date_range(start='1/1/2020', periods=5),
            'value': [1, 2, np.nan, 4, 5]
        }
        df = pd.DataFrame(data)
        tw = TimeWeaver(df, 'date')
        tw.evaluate()

    .. note:: 
        This class is designed to work with pandas DataFrames and requires the pandas library.
    """
    def __init__(self, df: pd.DataFrame, tracking_column: str, frequency: float = 1.0, columns_to_drop : list = [], logging: bool = False):
        """
        Initializes the TimeWeaver class with a DataFrame and a tracking column name.

        :param df: The DataFrame containing the data to be analyzed and processed.
        :type df: pd.DataFrame
        :param tracking_column: The column name used for tracking, often time.
        :type tracking_column: str
        """

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
        # Processed Dataframe
        self.df_processed = None
        # Pipeline
        self.pipeline = None
        # Time / Tracking column
        self.tracking_column = tracking_column
        # Frequency for Continuity Reindexing
        self.frequency = frequency
        # Columns to drop
        self.columns_to_drop = columns_to_drop
        # Reuslts of Method performance
        self.evaluation_dataframe = None
        # Method worked : True / False
        self.method_success_dataframe = None
        # Logging
        self.logging = logging

    def log(self, message, overwrite=True):
        """
        Prints a formatted log message to the console. Can overwrite the current line or print on a new line.

        :param message: The message to be logged.
        :type message: str
        :param overwrite: If True, the message overwrites the current line in the console. If False, the message is printed on a new line.
        :type overwrite: bool
        """

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
    
    def preprocess(self, prints: bool = True):
        """
        Preprocesses the DataFrame by replacing non-numeric characters with NaNs and ensuring continuity in the tracking column.
        """

        if prints:
            self.log("Pre-Processing starting...", overwrite=True)
        
        if self.logging is False:
                warnings.filterwarnings("ignore", category=FutureWarning)

        pipeline = Pipeline([
        ('drop_columns', DropColumnTransformer(columns_to_drop=self.columns_to_drop)),
        ('characters_to_nan', CharactersToNaNTransformer(value=np.nan)),
        ('edge_nan_filler', EdgeNaNFillerTransformer()),
        #('continuity_reindex', ContinuityReindexTransformer(time_column=self.tracking_column, frequency=self.frequency)),
        ])

        pipeline.fit(self.df)
        df_processed = pipeline.transform(self.df)

        if prints:
            self.log("Pre-Processing Done.", overwrite=False)

        return df_processed
    
    def build_PreProcessor(self, prints: bool = True):
        """
        Preproccesor Object
        """

        if self.evaluation_dataframe is not None:
        
            method_dict = self.get_best(optimized_selection=True)

            pipeline = Pipeline([
            ('drop_columns', DropColumnTransformer(columns_to_drop=self.columns_to_drop)),   
            ('characters_to_nan', CharactersToNaNTransformer(value=np.nan)),
            ('continuity_reindex', ContinuityReindexTransformer(tracking_column=self.tracking_column, frequency=self.frequency)),
            ('edge_nan_filler', EdgeNaNFillerTransformer()),
            ('column_interpolate', ColumnInterpolateTransformer(interpolation_methods=method_dict)),
            ])

            self.pipeline = pipeline

            return PreProcessor(pipeline)
    
        else:
                self.log(
                    "Results DataFrame is not available. Please run evaluate() first.",
                    overwrite=False,
                )
    
    def get_summary(self, full_summary: bool = False):
        return get_summary(self, full_summary)
    
    def get_summary_characters(self):
        return get_summary_characters(self)
    
    def get_rate_analysis(self, start_rate: float = 0.1, end_rate: float = 0.5, step: float = 0.01, prints: bool = False):
        return get_rate_analysis(self, start_rate, end_rate, step, prints)

    def evaluate(
            self, missing_rate=0.1, random: bool = True, prints: bool = True, logging: bool = False
        ):
            """
            Evaluates various interpolation methods on the DataFrame, under specified conditions.

            :param missing_rate: Proportion of values to be randomly removed from each column for interpolation testing.
            :type missing_rate: float
            :param random: Determines if missing values are randomly selected or evenly spaced.
            :type random: bool
            :param prints: Controls printout of log messages during the evaluation process.
            :type prints: bool
            :param logging: Enables logging of warning messages for interpolation methods that result in errors or NaNs.
            :type logging: bool

            **Example Usage**:

            .. code-block:: python

                tw.evaluate(missing_rate=0.2, random=True, prints=True, logging=True)
            """
            
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

            if self.df_processed is None:
                self.preprocess(prints=prints)

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
                            # print(method ["name"] + " " + str(method["order"]))
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
        Highlights the lowest non-negative value in a Series for visualization.

        :param s: Series of values to be highlighted.
        :type s: pd.Series
        :return: List of styles to be applied for highlighting.
        :rtype: list

        **Example Usage**:

        .. code-block:: python

            styled_df = df.style.apply(tw.highlight_lowest, axis=1)
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

        """
        Highlights False values in a Series, used primarily for visualizing method success or failure.

        :param s: Series of boolean values indicating success (True) or failure (False).
        :type s: pd.Series
        :return: List of styles to be applied for highlighting False values.
        :rtype: list

        **Example Usage**:

        .. code-block:: python

            styled_df = method_success_df.style.apply(tw.highlight_false, axis=1)
        """
   
        return ['background-color: #FF4500' if v is False else '' for v in s]

    def get_evaluation_dataframe(self, style: bool = True):
        """
        Retrieves the evaluation DataFrame with optional styling to highlight the lowest values.

        :param style: Applies styling if set to True.
        :type style: bool
        :return: Evaluation DataFrame, optionally styled.
        :rtype: pd.DataFrame or Styler

        **Example Usage**:

        .. code-block:: python

            eval_df = tw.get_evaluation_dataframe(style=True)
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
        """
        Retrieves the method success DataFrame with optional styling to highlight False values.

        :param style: Applies styling if set to True.
        :type style: bool
        :return: Method success DataFrame, optionally styled.
        :rtype: pd.DataFrame or Styler

        **Example Usage**:

        .. code-block:: python

            success_df = tw.get_method_success_dataframe(style=True)
        """
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
        """
        Identifies the best interpolation method(s) for each column based on the evaluation results.

        :param optimized_selection: Selects methods based on performance and frequency of being best if True. 
                                    Otherwise, selects the method(s) with the lowest error.
        :type optimized_selection: bool
        :return: Mapping each column to the best interpolation method(s) or a list of (column, method) tuples.
        :rtype: dict or list

        **Example Usage**:

        .. code-block:: python

            best_methods = tw.get_best(optimized_selection=True)
            print(best_methods)
        """
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
                    if methods: 
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