from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
            decision_tree_model = DecisionTreeClassifier(random_state=42)
            logging.info(
                'fitting train set to --- [DecisionTree Classifier] ---')
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
            random_forest_model = RandomForestClassifier(
                n_estimators=100, n_jobs=-1, class_weight="balanced")
            logging.info(
                'fitting train set to --- [RandomForest Classifier] ---')
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
            # Adjust scale_pos_weight based on imbalance
            xg_model = XGBClassifier(random_state=42, scale_pos_weight=49)
            logging.info(
                'fitting train set to --- [XGBRegressor Classifier] ---')
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
            A tuple containing the Accuracy, precision, recall, f1, roc_auc and predicted values.
        """
        try:
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            logging.info("Model evaluated successfully.")
            return accuracy, precision, recall, f1, roc_auc, y_pred
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise
