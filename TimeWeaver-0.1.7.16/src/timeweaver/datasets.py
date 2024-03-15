import pandas as pd
from importlib import resources 

class DataSets:
    @classmethod
    def PRSA(cls):
        """
        Beijing PM2.5 Data: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
        """
        
        try:
            with resources.path('timeweaver', 'PRSA_data_2010.1.1-2014.12.31.csv') as filepath:
                data = pd.read_csv(filepath)
            return data
        except FileNotFoundError:
            print(filepath)
            print("The file was not found.")
            return None
        except Exception as e:
            print(filepath)
            print(f"An error occurred: {e}")
            return None