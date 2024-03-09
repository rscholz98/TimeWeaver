TimeWeaver Class
================

The ``TimeWeaver`` class facilitates data preprocessing, analysis, and evaluation of interpolation methods on pandas DataFrames, especially for time-series data.

Class Documentation
-------------------

.. automodule:: timeweaver
    :members:
    :undoc-members:
    :show-inheritance:


Usage Example
-------------

.. code-block:: python

   import pandas as pd
   from your_module_name import TimeWeaver  # Replace with your actual module name

   # Sample DataFrame
   data = {'time': pd.date_range(start='2021-01-01', periods=5, freq='D'),
           'value': [1, 2, np.nan, 4, 5]}
   df = pd.DataFrame(data)

   # Initialize TimeWeaver
   tw = TimeWeaver(df, 'time')

   # Get summary characters
   print(tw.get_summary_characters())

   # Evaluate interpolation methods
   tw.evaluate()


