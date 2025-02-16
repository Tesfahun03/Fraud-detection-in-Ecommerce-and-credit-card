import shap
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    filename='../logs/model-explainability.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class ModelExplainability:
    """
    A class to explain machine learning models using SHAP and LIME.
    """

    def __init__(self, model, x_train, x_test, feature_names):
        """
        Initialize the explainability class with model, training and test data.

        Parameters:
        ----------
        model : object
            The trained machine learning model.
        x_train : pd.DataFrame
            The feature columns of the training dataset.
        x_test : pd.DataFrame
            The feature columns of the test dataset.
        feature_names : list
            List of feature names.
        """
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.feature_names = feature_names

    def explain_with_shap(self):
        """
        Explain the model using SHAP values and generate plots.
        """
        try:
            logging.info('Initializing SHAP explainer...')
            explainer = shap.Explainer(self.model, self.x_train)
            shap_values = explainer(self.x_test, check_additivity=False)

            logging.info('Generating SHAP plots...')
            plt.figure()
            feature_names_array = np.array(
                self.feature_names)  # Convert to NumPy array

            shap.summary_plot(shap_values, self.x_test,
                              feature_names=feature_names_array)
            plt.savefig('../plots/shap_summary_plot.png')
            logging.info('SHAP summary plot saved.')

            plt.figure()
            # shap.dependence_plot(0, shap_values.values,
            #                      self.x_test, feature_names=feature_names_array)
            # plt.savefig('../plots/shap_dependence_plot.png')
            # logging.info('SHAP dependence plot saved.')

        except Exception as e:
            logging.error(f"Error generating SHAP explanations: {e}")
            raise

    def explain_with_lime(self, instance_index=0):
        """
        Explain an individual prediction using LIME.

        Parameters:
        ----------
        instance_index : int, optional
            The index of the test instance to explain (default is 0).
        """
        try:
            logging.info('Initializing LIME explainer...')
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(self.x_train),
                feature_names=self.feature_names,
                class_names=['Fraud', 'Not Fraud'],
                mode='regression'
            )

            logging.info(
                f'Generating LIME explanation for instance {instance_index}...')
            exp = explainer.explain_instance(
                data_row=self.x_test.iloc[instance_index],
                predict_fn=self.model.predict
            )

            exp.save_to_file(
                f'../plots/lime_explanation_{instance_index}.html')
            logging.info(
                f'LIME explanation saved as HTML for instance {instance_index}.')

        except Exception as e:
            logging.error(f"Error generating LIME explanations: {e}")
            raise
