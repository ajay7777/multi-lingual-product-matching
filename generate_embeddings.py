from typing import List
import numpy as np
import pandas as pd
import pathlib

from tqdm import tqdm
from util import augment_train_phone_language_pairs, prep_data_pair
from sentence_transformers import SentenceTransformer
from laserembeddings import Laser

# Define string constants
SMALL = "small"
MEDIUM = "medium"
LARGE = "large"
XLARGE = "xlarge"
DISTILLATION_MODE = "distillation"
LASER_MODE = "laser"


def embeddings_gen():

    for category in ["watch"]:
        # Get the relevant data from the settings
        dataset_size = "medium"
        use_description = True
        # Process the categories separately
        dataset_p = pathlib.Path().parent.resolve().joinpath("datasets")
        if dataset_size == SMALL:
            train_data_p = dataset_p.joinpath(
                f'pairwise_train_set_{category}_{SMALL}.csv')
            test_data_p = dataset_p.joinpath(
                f'pairwise_test_set_{category}.csv')
        elif dataset_size == MEDIUM:
            train_data_p = dataset_p.joinpath(
                f'pairwise_train_set_{category}_{MEDIUM}.csv')
            test_data_p = dataset_p.joinpath(
                f'pairwise_test_set_{category}.csv')
        elif dataset_size == LARGE:
            train_data_p = dataset_p.joinpath(
                f'pairwise_train_set_{category}_{LARGE}.csv')
            test_data_p = dataset_p.joinpath(
                f'pairwise_test_set_{category}.csv')
        elif dataset_size == XLARGE:
            train_data_p = dataset_p.joinpath(
                f'pairwise_train_set_{category}_{XLARGE}.csv')
            test_data_p = dataset_p.joinpath(
                f'pairwise_test_set_{category}.csv')

        # Read the data
        train_data = pd.read_csv(train_data_p)
        test_data = pd.read_csv(test_data_p)

        if category == "phone":
            train_data = augment_train_phone_language_pairs(train_data)

        # Prepare the train and test data for the experiments and get the mapping of the labels
        train_data, test_data = prep_data_pair(
            train_data, test_data, use_description)

        generate_embeddings(type=LASER_MODE, data=train_data,
                            category=category, train=True, dataset_p=dataset_p)
        generate_embeddings(type=LASER_MODE, data=test_data,
                            category=category, train=False, dataset_p=dataset_p)
        generate_embeddings(type=DISTILLATION_MODE, data=train_data,
                            category=category, train=True, dataset_p=dataset_p)
        generate_embeddings(type=DISTILLATION_MODE, data=test_data,
                            category=category, train=False, dataset_p=dataset_p)


def generate_embeddings(type, data, category, train, dataset_p):
    d_list = []
    laser = Laser()
    model_st = SentenceTransformer(
        'distiluse-base-multilingual-cased-v2')
    for row in tqdm(data.itertuples()):
        d = {}
        if type == "distillation":
            print("distillation generation started")
            sent_emb1 = model_st.encode(
                sentences=row.content_1, normalize_embeddings=True)
            sent_emb2 = model_st.encode(
                sentences=row.content_2, normalize_embeddings=True)
        else:
            print(row.lang_1)
            print(row.lang_2)
            sent_emb1 = laser.embed_sentences(
                sentences=row.content_1, lang=row.lang_1)
            sent_emb1 = sent_emb1[0]  # lang is only used for tokenization
            sent_emb2 = laser.embed_sentences(
                sentences=row.content_2, lang=row.lang_2)
            sent_emb2 = sent_emb2[0]
        abs_diff = np.absolute(np.array(sent_emb1) - np.array(sent_emb2))

        i, j, k = 1, 1, 1
        for emb_1 in sent_emb1:
            d["emb1_"+str(i)] = emb_1
            i = i+1
        for emb_2 in sent_emb2:
            d["emb2_"+str(j)] = emb_2
            j = j+1
        for emb_diff in abs_diff:
            d["emb_abs_diff_"+str(k)] = emb_diff
            k = k+1
        d["lang_1"] = row.lang_1
        d["lang_2"] = row.lang_2
        d["label"] = row.label
        d_list.append(d)

    df_emb = pd.DataFrame(d_list)
    if train:
        df_emb.to_csv(dataset_p.joinpath(
            f'{type}_embeddings_train_{category}.csv'))
        print("saved_train_csv")
    else:
        df_emb.to_csv(dataset_p.joinpath(
            f'{type}_embeddings_test_{category}.csv'))
        print("saved_test_csv")


if __name__ == "__main__":
    embeddings_gen()
