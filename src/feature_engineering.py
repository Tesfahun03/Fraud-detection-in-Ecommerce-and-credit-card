import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os
sys.path.append(os.path.abspath('..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='../logs/feature_engineering.log'
)

logging.info(
    '****************************Logging started for Feature Engineering module****************************')


class FeatureEngineering:
    def __init__(self, data):
        self.data = data

    def perform_insertion(self, column, new_column_name, value):
        """A method used to insert a new column to the DataFrame.

        Args:
            column (pd.series): pandas dataframe column to insert the new column after.
            new_column_name (): new column name to insert.
            value (pd.series): value to insert in the new column.

        Returns:
            pd.Daaframe: pandas dataframe with the new column inserted.
        """
        logging.info(
            'Inserting new column to the column')
        column_index = self.data.columns.get_loc(column)
        self.data.insert(column_index + 1, new_column_name, value)
        logging.info(f'Column {new_column_name} inserted successfully')
        return self.data

    def get_purchase_weekday(self):
        """
        Create a new column 'purchase_weekday' based on the 'purchase_time' column.
        This method creates a new column 'purchase_weekday' based on the 'purchase_time' column.
        The 'purchase_weekday' column contains the weekday (Monday=0, Sunday=6) of the purchase date.
        Returns:
            pandas.DataFrame: A DataFrame with the new 'purchase_weekday' column.
        """
        try:
            logging.info(f'Creating purchase_weekday column')
            purchase_weekday = self.data['purchase_time'].dt.dayofweek
            self.perform_insertion(
                'purchase_time', 'purchase_weekday', purchase_weekday)
            return self.data
        except Exception as e:
            logging.error("Error creating purchase_weekday column: %s", str(e))
            raise

    def get_purchase_hour(self):
        try:
            logging.info(f'creating purchase hour based on purchase time')
            day_of_hr = self.data['purchase_time'].dt.hour
            self.perform_insertion('purchase_time', 'purchase_hour', day_of_hr)
            return self.data
        except Exception as e:
            logging.error(
                f'canot extract hout of the day from purchase-time  :: {e}')

    def transaction_frequency(self):
        """
        Create a new column 'transaction_frequency' based on the number of transactions per user.
        This method creates a new column 'transaction_frequency' based on the number of transactions per user.
        The 'transaction_frequency' column contains the number of transactions made by each user.
        Returns:
            pandas.DataFrame: A DataFrame with the new 'transaction_frequency' column.
        """
        try:
            logging.info('Creating transaction_frequency column')
            transaction_freq = self.data.groupby(
                'user_id')['user_id'].transform('count')
            self.perform_insertion(
                'user_id', 'transaction_frequency', transaction_freq)
            return self.data
        except Exception as e:
            logging.error(
                "Error creating transaction_frequency column: %s", str(e))
            raise

    def velocity_check(self):
        """
        Create a new column 'velocity_check' based on the time difference between signup and purchase.
        This method creates a new column 'velocity_check' based on the time difference between signup and purchase.
        The 'velocity_check' column contains the time difference in seconds between the signup and purchase time.
        Returns:
            pandas.DataFrame: A DataFrame with the new 'velocity_check' column.
        """
        try:
            logging.info('Creating velocity_check column')
            velocity = (self.data['purchase_time'] -
                        self.data['signup_time']).dt.total_seconds()
            self.perform_insertion('purchase_time', 'velocity_check', velocity)
            return self.data
        except Exception as e:
            logging.error("Error creating velocity_check column: %s", str(e))
            raise
