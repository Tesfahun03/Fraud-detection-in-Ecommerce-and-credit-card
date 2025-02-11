import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(
    filename='../logs/encoding.logs',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info(
    '****************************Logging started for Ecoding Data module****************************')


class DataProcessing:
    """
    DataProcessing class for handling data encoding, correlation calculation, and standardization.
    Attributes:
        data (pd.DataFrame): The input data to be processed.
    Methods:
        encode_data():
            Encodes categorical columns into numerical values using LabelEncoder.
                pd.DataFrame: Encoded dataframe.
        corr_with_target(target):
            Calculates the correlation of numerical columns with the target column.
                target (str): The target column name.
                pd.Series: Correlation values with the target column.
        standardize_data(dataframe):
            Standardizes the numerical columns of the dataframe using StandardScaler.
                dataframe (pd.DataFrame): The dataframe to be standardized.
                pd.DataFrame: Standardized dataframe.
    """

    def __init__(self, data):
        self.data = data
        logging.info("DataProcessing instance created.")

    def encode_data(self):
        """_encodes catagorical coloumns into sum randomly assigned numbers for regression purpose_

        Returns:
            _DataFrame_: _encoded_dataframe_
        """
        logging.info("Starting encoding of categorical columns.")
        columns_label = self.data.select_dtypes(
            include=['object']).columns
        df_lbl = self.data.copy()
        try:
            for col in columns_label:
                label = LabelEncoder()
                label.fit(list(self.data[col].values))
                df_lbl[col] = label.transform(df_lbl[col].values)
            logging.info("Encoding completed.")
            return df_lbl
        except Exception as e:
            logging.error(f'Error while trying to encode data:: {e}')
            raise

    def corr_with_target(self, target):
        logging.info(f"Calculating correlation with target: {target}")
        numericals = self.data.select_dtypes(include=['int64', 'float64'])
        corr = numericals.corr()
        corr_with_target = corr[target]
        logging.info("Correlation calculation completed.")
        return corr_with_target

    def standardize_data(self, dataframe):
        """_standrardize the dataset columns_

        Args:
            dataframe ( Pd.DataFrame ): _pandas_dataframe(endoded data)_

        Returns:
            Pd.Dataframe: standardize_dataframe
        """
        logging.info("Starting standardization of dataframe.")
        column_scaler = dataframe.select_dtypes(
            include=['object', 'float64', 'int64']).columns
        df_standard = dataframe.copy()
        standard = StandardScaler()
        try:
            for col in column_scaler:

                df_standard[column_scaler] = standard.fit_transform(
                    df_standard[column_scaler])
            logging.info("Standardization completed.")
            return df_standard
        except Exception as e:
            logging.error(
                f'Error occured while standardizing data :: Erorr :- {e}')
            raise
