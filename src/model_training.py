from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(
    filename='../logs/model-training.logs',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info(
    '****************************Logging started for Model Training module****************************')


class SplitData:
    """
    A class used to split the dataset into training and testing sets.

    Attributes:
    ----------
    x : pd.DataFrame
        The feature columns of the dataset.
    y : pd.DataFrame
        The target column of the dataset.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def split_data(self):
        """
        Splits the data into 80% training data and 20% testing data.

        Returns:
        -------
        tuple
            A tuple containing the training and testing data.
        """
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=0.2, random_state=42)
            logging.info("Data split with [ --- 80%|20% --- ] successfully.")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise


class TrainData:
    """
    A class for training multiple machine learning algorithms on the same training and testing sets.

    Attributes:
    ----------
    x_train : pd.DataFrame
        The feature columns of the training dataset.
    y_train : pd.DataFrame
        The target column of the training dataset.
    """

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def decision_tree_regressor(self):
        """
        Initializes the Decision Tree Regressor model and fits it to the training data.

        Returns:
        -------
        DecisionTreeRegressor
            A Decision Tree Regressor model fitted on the training data.
        """
        try:
            logging.info('initializing decision tree')
            decision_tree_model = DecisionTreeRegressor(random_state=42)
            logging.info(
                'fitting train set to --- [DecisionTree Regressor] ---')
            decision_tree_model.fit(self.x_train, self.y_train)
            logging.info("Decision Tree Regressor model trained successfully.")
            return decision_tree_model
        except Exception as e:
            logging.error(f"Error training Decision Tree Regressor model: {e}")
            raise

    def random_forest(self):
        """
        Initializes the Random Forest model and fits it to the training data.

        Returns:
        -------
        RandomForestRegressor
            A Random Forest model fitted on the training data.
        """
        try:
            logging.info('initializing random forest with all CPUs')
            random_forest_model = RandomForestRegressor(
                n_estimators=100, n_jobs=-1)
            logging.info(
                'fitting train set to --- [RandomForest Regressor] ---')
            random_forest_model.fit(self.x_train, self.y_train)
            logging.info("Random Forest model trained successfully.")
            return random_forest_model
        except Exception as e:
            logging.error(f"Error training Random Forest model: {e}")
            raise

    def xgboost(self):
        """
        Initializes the XGBoost model and fits it to the training data.

        Returns:
        -------
        XGBRegressor
            An XGBoost model fitted on the training data.
        """
        try:
            xg_model = XGBRegressor(random_state=42)
            logging.info(
                'fitting train set to --- [XGBRegressor Regressor] ---')
            xg_model.fit(self.x_train, self.y_train)
            logging.info("XGBoost model trained successfully.")
            return xg_model
        except Exception as e:
            logging.error(f"Error training XGBoost model: {e}")
            raise


class EvaluateModel:
    """
    A class for evaluating the accuracy of a given model.

    Methods:
    -------
    evaluate_model(model, x_test, y_test)
        Evaluates the errors of the model using accuracy metrics.
    """

    def evaluate_model(self, model, x_test, y_test):
        """
        Evaluates the errors of the model using accuracy metrics.

        Parameters:
        ----------
        model : object
            The regression model to measure its accuracy.
        x_test : pd.DataFrame
            The feature columns of the testing dataset.
        y_test : pd.DataFrame
            The target column of the testing dataset.

        Returns:
        -------
        tuple
            A tuple containing the Mean Absolute Error, Mean Squared Error, R-squared score, and predicted values.
        """
        try:
            y_pred = model.predict(x_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info("Model evaluated successfully.")
            return mae, mse, r2, y_pred
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise
