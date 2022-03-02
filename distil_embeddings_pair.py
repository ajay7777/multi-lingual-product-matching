from typing import List
import numpy as np
import pandas as pd
import json
import pathlib
import py_entitymatching as em
from sklearn import preprocessing

from util import compute_metrics, prep_data_pair, output_and_store_results, prep_data_pair_mallegan, \
    create_config_key
from argparse import ArgumentParser

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

# Define string constants
SMALL = "small"
MEDIUM = "medium"
LARGE = "large"
XLARGE = "xlarge"
LOGIT = "logistic-regression"
RAFO = "random-forest"
SVM = "svm"
MAGELLAN = "magellan"
STANDARD = "standard"
HARD = "hard"
COOC = "cooc"
LASER = "laser"


def run_baseline_pair(input_path: str, setting_keys: List[str] = None):
    # Read settings file
    with open(f'{input_path}') as file:
        settings = json.load(file)

    for setting_key, setting_data in settings.items():
        # Only run the setting if the key is in the list of settings or no setting_keys are provided
        if setting_keys is None:
            pass
        elif setting_keys is not None and setting_key not in setting_keys:
            continue

        # Get name of settings
        settings_name = create_config_key(setting_data)
        # Get the relevant data from the settings
        model = setting_data.get("model")
        vectorization = setting_data.get("vectorization")
        dataset_size = setting_data.get("dataset_size")
        use_description = setting_data.get("use_description")
        train_langs = setting_data.get("train_lang")
        test_langs = setting_data.get("eval_lang")
        category = setting_data.get("category")

        # Create a string of the train languages
        train_langs_str = ", ".join(train_langs)

        # Process the categories separately
        dataset_p = pathlib.Path(input_path).parent.joinpath("datasets")

        train_data_p = dataset_p.joinpath("embeddings_distilled_train.csv")
        test_data_p = dataset_p.joinpath("embeddings_distilled_test.csv")
        # Read the data
        train_data = pd.read_csv(train_data_p)    
        test_data = pd.read_csv(test_data_p)

        features = list(train_data.columns)
        features.remove('label')
        train_data_embeddings = np.array(train_data[features].values)

        # Fit the models
        if model == LOGIT:
            est = LogisticRegression()
            parameters = {
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'class_weight': ['balanced'],
                'max_iter': [5000],
                'n_jobs': [-2]
            }

        elif model == RAFO:
            est = RandomForestClassifier()
            parameters = {
                'n_estimators': [100],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [2, 4, 7, 10],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'class_weight': ['balanced_subsample'],
                'n_jobs': [-2]
            }

        elif model == SVM:
            est = SVC()
            parameters = {
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'class_weight': ['balanced'],
                'kernel': ['rbf','linear','poly'],
                'degree':[0, 1, 2, 3, 4, 5, 6],
                'gamma':['scale', 'auto']
            }
        else:
            # Other models are not implemented
            raise AssertionError

        print(model)
        # Define grid search and fit model
        rs = GridSearchCV(estimator=est, param_grid=parameters, scoring="f1_macro", cv=2,
                                n_jobs=-1, verbose=10, refit=True)
        rs.fit(train_data_embeddings, train_data["label"].astype(int))

        # Generate list for scores
        scores_per_lang = {}

        ## Run predictions
        # Run predictions for cooc feature
        if vectorization == COOC:
            for lang in test_langs:
                # Subset the test data
                test_data_lang = test_data.loc[test_data['lang_1'] == lang]

                # Retrieve representations & word co-occurence vectors for test set
                cv_content1_test = cv.transform(test_data_lang['content_1']).toarray()
                cv_content2_test = cv.transform(test_data_lang['content_2']).toarray()
                test_data_embeddings_lang = np.multiply(cv_content1_test, cv_content2_test)

                # Prediction and computation of metrics to measure performance of model
                pred = rs.best_estimator_.predict(test_data_embeddings_lang)
                scores_per_lang[lang] = compute_metrics({"labels": test_data_lang["label"], "predictions": pred}).get(
                    "f1")
                output_and_store_results(setting_data, settings_name, category, train_langs_str, lang,
                                         scores_per_lang[lang], "", str(rs.best_params_), input_path, pred)

        # Run predictions for Magellan features
        elif vectorization == MAGELLAN:
            for lang in test_langs:
                # Subset the test data
                test_data_lang = test_data.loc[test_data['lang_1'] == lang]

                # Retrieve tables A,B,G
                A, B, G = prep_data_pair_mallegan(test_data_lang, use_description)

                # Generate features
                # feature_table = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)
                H = em.extract_feature_vecs(G,
                                            feature_table=feature_table,
                                            attrs_before=attrs_from_table,
                                            attrs_after='label',
                                            show_progress=False)

                # Replace NA values
                H.fillna(-1, inplace=True)

                # Retrieve features
                test_data_embeddings_lang = H.drop(columns=attrs_to_be_excluded)

                # Normalize Features
                test_data_embeddings_lang = normalizer.transform(test_data_embeddings_lang)

                # Prediction and computation of metrics to measure performance of model
                pred = rs.best_estimator_.predict(test_data_embeddings_lang)
                scores_per_lang[lang] = compute_metrics(
                    {"labels": test_data_lang["label"], "predictions": pred}).get("f1")
                output_and_store_results(setting_data, settings_name, category, train_langs_str, lang,
                                         scores_per_lang[lang], "", str(rs.best_params_), input_path, pred)
        elif vectorization == LASER:
                    for lang in test_langs:
                        # Subset the test data
                        test_data_laser = pd.read_csv("embeddings_distilled_test.csv")
                        test_data_lang = test_data_laser.loc[test_data_laser['lang_1'] == lang]
                        
                        features = list(test_data_lang.columns)
                        features.remove('label')
                        test_data_embeddings_lang = np.array(test_data_lang[features].values)
                        # Normalize Features
                        #test_data_embeddings_lang = normalizer.transform(test_data_embeddings_lang)

                        # Prediction and computation of metrics to measure performance of model
                        pred = rs.best_estimator_.predict(test_data_embeddings_lang)
                        scores_per_lang[lang] = compute_metrics(
                            {"labels": test_data_lang["label"], "predictions": pred}).get("f1")
                        output_and_store_results(setting_data, settings_name, category, train_langs_str, lang,
                                                scores_per_lang[lang], "", str(rs.best_params_), input_path, pred)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input
    run_baseline_pair(input_path)