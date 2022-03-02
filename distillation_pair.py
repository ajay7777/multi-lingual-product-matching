from typing import List
import numpy as np
import pandas as pd
import json
import pathlib
import py_entitymatching as em
from sklearn import preprocessing
from torch import embedding

from util import compute_metrics, output_and_store_results_emb, prep_data_pair, output_and_store_results, prep_data_pair_mallegan, \
    create_config_key
from argparse import ArgumentParser

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import svm

np.random.seed(246)

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
EMBEDDINGMODEALL = "all"
EMBEDDINGMODEABSDIFF="absolute-difference"

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
        mode="distillation"
        #vectorization = setting_data.get("vectorization")
        #dataset_size = setting_data.get("dataset_size")
        #use_description = setting_data.get("use_description")
        train_langs = setting_data.get("train_lang")
        test_langs = setting_data.get("eval_lang")
        category = setting_data.get("category")
        embedding_mode = setting_data.get("embeddings_mode")
        use_cross_lingual_pairs = setting_data.get("use_cross_lingual_pairs")
        print(embedding_mode)
        # Create a string of the train languages
        train_langs_str = ", ".join(train_langs)

        # Process the categories separately
        dataset_p = pathlib.Path(input_path).parent.joinpath("datasets")

        train_data_p = dataset_p.joinpath(f"{mode}_embeddings_train_{category}.csv")
        test_data_p = dataset_p.joinpath(f"{mode}_embeddings_test_{category}.csv")
        # Read the data
        train_data = pd.read_csv(train_data_p)    

        # Filter the train data:
        if use_cross_lingual_pairs:
            train_data = train_data.loc[(train_data['lang_1'].isin(train_langs)) & (train_data['lang_2'].isin(train_langs))]
        else:
            train_data = train_data.loc[train_data['lang_1'].isin(train_langs)]
            train_data = train_data.loc[train_data['lang_1']==train_data['lang_2']]
        if set(embedding_mode)== set(["abs_diff"]):
            print("abs_diff")
            train_data = train_data[train_data.columns.drop(list(train_data.filter(regex='emb1_')))]
            train_data = train_data[train_data.columns.drop(list(train_data.filter(regex='emb2_')))]
        else:
            print("all embeddings")
        print(len(train_data))
        features = list(train_data.columns)
        features.remove('label')
        features.remove('Unnamed: 0')
        features.remove('lang_1')
        features.remove('lang_2')
        if mode=="laser":
            x = train_data[features].values
            min_max_scaler = preprocessing.StandardScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            train_data_embeddings = x_scaled
        else:  
            train_data_embeddings = np.array(train_data[features].values)

        # Fit the models
        if model == LOGIT:
            est = LogisticRegression(class_weight="balanced",max_iter=10000, n_jobs=-1)
            parameters = {
                'C': [1, 10],
            }

        elif model == RAFO:
            est = RandomForestClassifier()
            parameters = {
                'n_estimators': [100],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [2, 4, 7, 10,100],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'class_weight': ['balanced_subsample'],
                'n_jobs': [-1]
            }

        elif model == SVM:
            est = svm.SVC(kernel="linear", class_weight="balanced")
            parameters = {
                'C': [1, 10]
            }
        else:
            # Other models are not implemented
            raise AssertionError

        print("Model= "+ model)
        # Define grid search and fit model
        rs = GridSearchCV(estimator=est, param_grid=parameters, scoring="f1_macro", cv=3,
                                n_jobs=-1, verbose=10, refit=True)
        rs.fit(train_data_embeddings, list(train_data["label"].astype(int)))

        # Generate list for scores
        scores_per_lang = {}

        for lang in test_langs:
            # Subset the test data
            test_data_laser = pd.read_csv(test_data_p)
            if set(embedding_mode) == set(["abs_diff"]):
                print(EMBEDDINGMODEABSDIFF)
                test_data_laser = test_data_laser[test_data_laser.columns.drop(list(test_data_laser.filter(regex='emb1_')))]
                test_data_laser = test_data_laser[test_data_laser.columns.drop(list(test_data_laser.filter(regex='emb2_')))]
            else:
                print(EMBEDDINGMODEALL)
            test_data_lang = test_data_laser.loc[test_data_laser['lang_1'] == lang]
            
            features = list(test_data_lang.columns)
            features.remove('label')
            features.remove('Unnamed: 0')
            features.remove('lang_1')
            features.remove('lang_2')
            test_data_embeddings_lang = np.array(test_data_lang[features].values)
            # Normalize Features
            #test_data_embeddings_lang = normalizer.transform(test_data_embeddings_lang)

            # Prediction and computation of metrics to measure performance of model
            pred = rs.best_estimator_.predict(test_data_embeddings_lang)
            scores_per_lang[lang] = compute_metrics(
                {"labels": test_data_lang["label"], "predictions": pred}).get("f1")
            output_and_store_results_emb(setting_data, settings_name, category, train_langs_str, lang,
                                    scores_per_lang[lang], "", str(rs.best_params_), input_path, pred, embedding_mode, use_cross_lingual_pairs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input
    run_baseline_pair(input_path)