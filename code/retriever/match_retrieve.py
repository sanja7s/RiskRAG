
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process, utils

import pickle
import requests
from copy import deepcopy
import ast
import os
import argparse
import string

from get_embeddings import compute_embeddings

MODEL = ["BAAI/bge-large-en-v1.5",
         "Linq-AI-Research/Linq-Embed-Mistral",
         "Salesforce/SFR-Embedding-2_R"]
TOP_K = 5
OUTPUT_FOLDER = "data/new_data/results_v2/"

# Flag to set if AIID should be used in train set
use_aiid_full = True
# Flag to set if top N downloaded models to be used as test set, need to be true for user_study samples
use_best_test_set = True
# Percentage of the data to be used as the test set
percent = 0.1
# Flag to set to retrieve risks for user study
use_user_study_samples = False
user_study_file = "data/new_data/sample_models_for_user_study_v2.csv"

# Flag to set for cross validation
use_cv = False


# Load all risk generated data
extracted_risks = pd.read_json(
    "data/new_data/llm_analysis/LLM_risk_analysis_results_gppt4o.json")
extracted_risks.set_index('cardID', inplace=True)

# process risks to contain clean strings
risk_ref_list = []
mit_ref_list = []
risk_list = []
mit_list = []

for i, row in extracted_risks.iterrows():
    risk_sentences = ''
    mit_sentences = ''
    risks = ''
    mits = ''
    if "risks_and_mitigations" in row["section"].keys():
        for sentence in row["section"]['risks_and_mitigations']:
            if sentence["type"] == "RISKS" or sentence["type"] == ["RISKS"]:
                risk_sentences += sentence["reference_text"] + ' ** '
            if sentence["type"] == "MITIGATIONS" or sentence["type"] == ["MITIGATIONS"]:
                mit_sentences += sentence["reference_text"] + ' ** '

            if "splits" in sentence.keys():
                for split in sentence["splits"]:
                    if split["type"] == "RISKS" or split["type"] == ["RISKS"]:
                        risks += split["unique_risk_mitigation"] + ' ** '
                    if split["type"] == "MITIGATIONS" or split["type"] == ["MITIGATIONS"]:
                        mits += split["unique_risk_mitigation"] + ' ** '

    risk_ref_list.append(risk_sentences)
    mit_ref_list.append(mit_sentences)
    risk_list.append(risks)
    mit_list.append(mits)

#  assign it to the dataframe
extracted_risks["risk_reference"] = risk_ref_list
extracted_risks["mitigation_reference"] = mit_ref_list
extracted_risks["risks"] = risk_list
extracted_risks["mitigations"] = mit_list

# extracted_risks.drop(["Incident ID", "rating_of_section",
#  "reasoning"], axis=1, inplace=True)
extracted_risks.drop(["Incident ID"], axis=1, inplace=True)
nan_or_empty_risk_indices = extracted_risks[extracted_risks["risks"].isna() | (extracted_risks["risks"] == '')].index
extracted_risks = extracted_risks.drop(index=nan_or_empty_risk_indices)

# data file containing aiid descriptions and risks
aiid = pd.read_json("data/llm_analysis/aiid/LLM_risks_aiid.json")
aiid.set_index("incidentID", inplace=True)
aiid["risks"] = aiid["risks"].apply(lambda x: '** '.join(x))
aiid_descriptions = aiid["description"]

# file containing classes from taxonomy for each risk 
risk_taxonomy = pd.read_csv("data/new_data/llm_analysis/LLM_risk_taxonomy_gppt4o.csv")
risk_taxonomy.drop("Unnamed: 0", axis=1)

# # file containing classes from taxonomy for each aiid risk 
# risk_taxonomy_aiid = pd.read_csv("data/new_data/llm_analysis/LLM_risk_taxonomy_aiid.csv")
# risk_taxonomy_aiid.drop("Unnamed: 0", axis=1)


def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation)).strip()

def find_close_risk(risk, all_risks, threshold=59):
    close_risk, fuzzscore, _ = process.extractOne(risk, all_risks, scorer = fuzz.token_sort_ratio, processor=utils.default_process)
    #TODO: originally near duplicates risks are removed using fuzzy matching and embedding model. Here it is matched back using only fuzzy matching. Hence there are some discrepancies in matching. For now, this is resolved with low fuzz score (>59)
    if fuzzscore > threshold: return close_risk 
    else: print (risk, close_risk, fuzzscore)

def remove_closest_risks(all_risks, return_list=True, remove_punct=True):
    if remove_punct:
        all_risks = [remove_punctuation(i) for i in all_risks.split('** ') if len(remove_punctuation(i))>0]
    else: all_risks = [i for i in all_risks.split('** ') if len(i)>0]
    if len(all_risks) == 1 : return all_risks
    slim_risks = []
    for risk in all_risks:
        mask = [i for i in all_risks if i!=risk]
        if not(mask): return list(set(all_risks))
        close_risk, fuzzscore, _ = process.extractOne(risk, mask, scorer = fuzz.token_sort_ratio, processor=utils.default_process)
        if fuzzscore > 80: slim_risks.append(risk if len(risk)>len(close_risk) else close_risk)
        else: slim_risks.append(risk)
    if return_list: return list(set(slim_risks))
    else: return '** '.join(list(set(slim_risks)))


# Get the indices of the top k similar texts for each test data point
def retrieve_topk_matches(cosine_sim, k=4):
    return np.argsort(cosine_sim, axis=1)[:, -(k):][:, ::-1]

def user_study_samples(file_name):
    sampled_unique_risks = pd.read_csv(file_name)
    sampled_unique_risks.set_index('cardID', inplace=True)
    mask = risk_cards.index.isin(sampled_unique_risks.index)
    rest = risk_cards[~mask]
    return sampled_unique_risks, rest

def get_taxonomy_classes(risk, taxonomy_data=risk_taxonomy):
    risk = pd.Series(risk, name="risk")
    # Merge the dataframes
    pred_taxonomy = pd.merge(taxonomy_data, risk, on='risk', how='right')
    if pred_taxonomy.empty:
        print (risk)
    # Create a mapping DataFrame
    mapping = risk.to_frame().copy()
    mapping['closest_risk'] = mapping['risk'].apply(lambda x: find_close_risk(x, taxonomy_data['risk'].tolist()))

    # Merge the mapping DataFrame with taxonomy_data to get full details of closest matches
    mapping = pd.merge(mapping, taxonomy_data, left_on='closest_risk', right_on='risk')

    # Fill NaN values in pred_taxonomy using the mapping DataFrame
    for col in ["axis1_target_of_analysis", "axis2_risk_area_main", "axis3_module_main"]:
        pred_taxonomy[col] = pred_taxonomy[col].fillna(mapping[col])
    return pred_taxonomy["axis1_target_of_analysis"].tolist(), pred_taxonomy["axis2_risk_area_main"].tolist(), pred_taxonomy["axis3_module_main"].tolist()


for model in MODEL:
    print("Model: ", model)
    if model == "BAAI/bge-large-en-v1.5":
        max_len = 512
    else:
        max_len = 1024
    
    # data file containing model card descriptions
    # this is read in every loop because the embeddings are calculated for descriptions before dropping nan_or_empty_risk_indices
    risk_cards = pd.read_json(
        "data/new_data/unique_risks_without_model_cards_v3.json")
    # dropping this model card as there was a problem generating risks with GPT4-o
    risk_cards.drop(15688, axis=0, inplace=True)
    # risk_cards.set_index('cardID', inplace=True)
    passages = risk_cards["model_description"]

    #  Split model cards description embeddings into train and test data, no validation for now
    df_X_train, df_X_test, corpus, queries = train_test_split(
        risk_cards, passages.tolist(), test_size=0.2, random_state=42)
    embeddings_corpus, embeddings_queries = compute_embeddings(
        corpus, queries, model=model, max_len=max_len, batch_size=8, corpus_embedding_available=False, data_name="new_mc_data")

    # Use the indices from the precomputed embeddings
    risk_cards_indices = np.array(risk_cards.index)

    # Create a DataFrame to hold all embeddings with correct indices
    all_embeddings_df = pd.DataFrame(
        index=risk_cards.index, columns=range(embeddings_corpus.shape[1]))
    all_embeddings_df.loc[df_X_train.index] = embeddings_corpus
    all_embeddings_df.loc[df_X_test.index] = embeddings_queries

    # dropping the model cards which do not have any risks extracted
    risk_cards = risk_cards.drop(index=nan_or_empty_risk_indices)
    # risk_cards.drop(index=nan_or_empty_risk_indices, axis=0, inplace=True)
    print (extracted_risks.shape)
    print (risk_cards.shape)

    if use_aiid_full:
        aiid_description_embeddings = compute_embeddings(corpus=None, queries=aiid_descriptions, model=model, max_len=max_len,
                                                         batch_size=8, data_name="aiid_descriptions", corpus_embedding_available=False, use_full_dataset=False, no_retrieval=True)
        
    # results = []
    #
    # for i, test_query in enumerate(queries):
    #     # get the df indices from the rangeindex in topk_indices
    #     similar_indices = df_X_train.index[topk_indices[i]]
    #     test_index = df_X_test.index[i]
    #     test_risks = extracted_risks.loc[test_index]['risks']
    #     matched_description = df_X_train.loc[similar_indices]['model_description'].tolist()
    #     matched_risks = extracted_risks.loc[similar_indices]['risks'].tolist()
    #     matched_mitigations = extracted_risks.loc[similar_indices]['mitigations'].tolist()
    #     print("Model Card: ", test_index)
    #     if use_aiid_full:
    #         similar_indices_aiid = aiid.index[topk_indices_aiid[i]]
    #         matched_description_aiid = aiid.loc[similar_indices_aiid]["description"].tolist()
    #         matched_risks_aiid = aiid.loc[similar_indices_aiid]['risks'].tolist()
    #         results.append({
    #             'test_index': test_index,
    #             'test_description': test_query,
    #             'similar_indices': similar_indices.tolist(),
    #             'matched_description': matched_description,
    #             'matched_risks': '** '.join(list(set(matched_risks))),
    #             'matched_mitigations': '** '.join(list(set(matched_mitigations))),
    #             'test_risks': test_risks,
    #             'test_mitigations': extracted_risks.loc[test_index]['mitigations'],
    #             'similar_indices_aiid': similar_indices_aiid.tolist(),
    #             'matched_description_aiid': matched_description_aiid,
    #             'matched_risks_aiid': '** '.join(list(set(matched_risks_aiid))),
    #         })
    #     else:
    #         results.append({
    #             'test_index': test_index,
    #             'test_description': test_query,
    #             'similar_indices': similar_indices.tolist(),
    #             'matched_description': matched_description,
    #             'matched_risks': '** '.join(list(set(matched_risks))),
    #             'matched_mitigations': '** '.join(list(set(matched_mitigations))),
    #             'test_risks': test_risks,
    #             'test_mitigations': extracted_risks.loc[test_index]['mitigations']
    #             # 'rephrased_risks': '** '.join(ast.literal_eval(response[2]['content'])['risks'])
    #         })

    # results_df = pd.DataFrame(results)
    # results_df.to_csv("data/new_data/matched_risks_aiid_{}.csv".format(model.split('/')[-1]))

    if use_best_test_set:
        # Sort the dataframe by the 'downloads' column in descending order
        risk_cards_sorted = risk_cards.sort_values(
            by='downloads', ascending=False)

        # Select the top N% rows
        best_risk_cards = risk_cards_sorted.head(
            int(np.round(percent*len(risk_cards_sorted))))
        rest_risk_cards = risk_cards_sorted.iloc[int(
            np.round(percent*len(risk_cards_sorted))):]

        if use_user_study_samples: 
            best_risk_cards, rest_risk_cards = user_study_samples(user_study_file)

        # Get the embeddings for the current train and test
        embeddings_train = all_embeddings_df.loc[rest_risk_cards.index].values.astype(
            float)
        embeddings_best_test = all_embeddings_df.loc[best_risk_cards.index].values.astype(
            float)

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(
            embeddings_best_test, embeddings_train)
        print('*'*10, similarity_matrix.shape)
        topk_indices = retrieve_topk_matches(similarity_matrix, k=TOP_K)
        results = []

        if use_aiid_full:

            similarity_matrix_aiid = cosine_similarity(
                embeddings_best_test, aiid_description_embeddings)
            print('*'*10, similarity_matrix_aiid.shape)
            topk_indices_aiid = retrieve_topk_matches(
                similarity_matrix_aiid, k=TOP_K)

        for i, test_query in enumerate(best_risk_cards["model_description"]):
            # get the df indices from the rangeindex in topk_indices
            similar_indices = rest_risk_cards.index[topk_indices[i]]
            test_index = best_risk_cards.index[i]
            test_risks = extracted_risks.loc[test_index]['risks']
            print("Model Card: ", test_index)
            matched_description = rest_risk_cards.loc[similar_indices]['model_description'].tolist()
            matched_risk_sections = rest_risk_cards.loc[similar_indices]['risks_limitations_bias'].tolist()
            matched_risks = remove_closest_risks('** '.join((extracted_risks.loc[similar_indices]['risks'].tolist())))
            if (len(matched_risks)==0):
                topk_indices = retrieve_topk_matches(similarity_matrix, k=TOP_K+TOP_K)
                similar_indices = rest_risk_cards.index[topk_indices[i]]
                matched_description = rest_risk_cards.loc[similar_indices]['model_description'].tolist()
                matched_risk_sections = rest_risk_cards.loc[similar_indices]['risks_limitations_bias'].tolist()
                matched_risks = remove_closest_risks('** '.join((extracted_risks.loc[similar_indices]['risks'].tolist())))
                # if pd.notna(best_risk_cards.loc[test_index]['card_data_base_model']):
                #     model_id = best_risk_cards.loc[test_index]['card_data_base_model']
                #TODO Other strategies like risks from the base model can be used but the base model may not have risk sections and hence not in our pool.
            matched_taxonomy = get_taxonomy_classes(matched_risks)
            matched_axis1 = matched_taxonomy[0]
            matched_axis2 = matched_taxonomy[1]
            matched_axis3 = matched_taxonomy[2]
            matched_mitigations = extracted_risks.loc[similar_indices]['mitigations'].tolist()
            # $$ is added to matched with no risks to maintain order with the risk sections. these are added to enable ranking
            # matched_risks = [extracted_risks.loc[idx, 'risks'] if pd.notnull(
                # extracted_risks.loc[idx, 'risks']) and extracted_risks.loc[idx, 'risks'] else " $$ " for idx in similar_indices]
            # matched_mitigations = [extracted_risks.loc[idx, 'risks'] if pd.notna(
                # extracted_risks.loc[idx, 'mitigations']) and extracted_risks.loc[idx, 'mitigations'] else " $$ " for idx in similar_indices]
            #TODO Reranking the risk sections 
            
            if use_aiid_full:
                similar_indices_aiid = aiid.index[topk_indices_aiid[i]]
                matched_description_aiid = aiid.loc[similar_indices_aiid]["description"].tolist()
                matched_risks_aiid = remove_closest_risks('** '.join((aiid.loc[similar_indices_aiid]['risks'].tolist())))
                # TODO Risk taxonomy for AIID risks.
                # matched_taxonomy_aiid = get_taxonomy_classes(matched_risks_aiid, taxonomy_data=risk_taxonomy_aiid)
                # matched_axis1_aiid = matched_taxonomy_aiid[0]
                # matched_axis2_aiid = matched_taxonomy_aiid[1]
                # matched_axis3_aiid = matched_taxonomy_aiid[2]
                # matched_risks_aiid = [aiid.loc[idx, 'risks'] if pd.notnull(
                #     aiid.loc[idx, 'risks']) and aiid.loc[idx, 'risks'] else " $$ " for idx in similar_indices_aiid]
                results.append({
                    'test_index': test_index,
                    'test_description': test_query,
                    'similar_indices': similar_indices.tolist(),
                    'matched_description': matched_description,
                    'matched_risks': '** '.join(matched_risks),
                    'matched_axis1_target_of_analysis': '** '.join(matched_axis1),
                    'matched_axis2_risk_area_main': '** '.join(matched_axis2),
                    'matched_axis3_module_main': '** '.join(matched_axis3),
                    'matched_mitigations': '** '.join(matched_mitigations),
                    'matched_risk_sections': '**|sep|** '.join(matched_risk_sections),
                    'test_risks': test_risks,
                    'test_mitigations': extracted_risks.loc[test_index]['mitigations'],
                    'test_risk_sections': best_risk_cards.loc[test_index]['risks_limitations_bias'],
                    'similar_indices_aiid': similar_indices_aiid.tolist(),
                    'matched_description_aiid': matched_description_aiid,
                    'matched_risks_aiid': '** '.join(matched_risks_aiid),
                    # 'matched_axis1_target_of_analysis_aiid': '** '.join(matched_axis1_aiid),
                    # 'matched_axis2_risk_area_main_aiid': '** '.join(matched_axis2_aiid),
                    # 'matched_axis3_module_main_aiid': '** '.join(matched_axis3_aiid)
                })
            else:
                results.append({
                    'test_index': test_index,
                    'test_description': test_query,
                    'similar_indices': similar_indices.tolist(),
                    'matched_description': matched_description,
                    'matched_risks': '** '.join(matched_risks),
                    'matched_axis1_target_of_analysis': '** '.join(matched_axis1),
                    'matched_axis2_risk_area_main': '** '.join(matched_axis2),
                    'matched_axis3_module_main': '** '.join(matched_axis3),
                    'matched_mitigations': '** '.join(matched_mitigations),
                    'matched_risk_sections': '** '.join(matched_risk_sections),
                    'test_risks': test_risks,
                    'test_mitigations': extracted_risks.loc[test_index]['mitigations'],
                    'test_risk_sections': best_risk_cards.loc[test_index]['risks_limitations_bias'],
                })

        results_df = pd.DataFrame(results)
        if use_user_study_samples: d = 'user_study' 
        else: d = 'best_test'
        # results_df.to_csv(OUTPUT_FOLDER+"matched_risks_{}_aiid_{}_top{}_{}percent_taxonomy.csv".format(d,
        #     model.split('/')[-1], TOP_K, int(percent*100)))
        # results_df.to_csv(OUTPUT_FOLDER+"matched_risks_user_study_aiid_{}_top{}_taxonomy.csv".format(
        #     model.split('/')[-1], TOP_K))

    if use_cv:
        # Cross validation
        # Number of folds for cross-validation
        n_splits = 5

        # Cross validation using ShuffleSplit
        # ss = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        ss = KFold(n_splits=n_splits)

        all_fold_results = []

        for fold, (train_index, test_index) in enumerate(ss.split(risk_cards_indices)):
            print(f"Fold {fold}:")
            # Map indices to actual train and test splits
            train_indices_fold = risk_cards_indices[train_index]
            test_indices_fold = risk_cards_indices[test_index]

            # Get the embeddings for the current fold
            embeddings_train_fold = all_embeddings_df.loc[train_indices_fold].values.astype(
                float)
            embeddings_test_fold = all_embeddings_df.loc[test_indices_fold].values.astype(
                float)

            # Compute the cosine similarity between test and train embeddings
            similarity_matrix = cosine_similarity(
                embeddings_test_fold, embeddings_train_fold)
            topk_indices = retrieve_topk_matches(similarity_matrix, k=TOP_K)

            if use_aiid_full:
                similarity_matrix_aiid = cosine_similarity(
                    embeddings_test_fold, aiid_description_embeddings)
                print('*'*10, similarity_matrix_aiid.shape)
                topk_indices_aiid = retrieve_topk_matches(
                    similarity_matrix_aiid, k=TOP_K)

            # Create a DataFrame to store the results for this fold
            fold_results = []

            for i, test_index in enumerate(test_indices_fold):
                similar_indices = train_indices_fold[topk_indices[i]]
                matched_description = risk_cards.loc[similar_indices]['model_description'].tolist(
                )
                matched_risks = extracted_risks.loc[similar_indices]['risks'].tolist(
                )
                matched_mitigations = extracted_risks.loc[similar_indices]['mitigations'].tolist(
                )
                print("Model Card: ", test_index)
                if use_aiid_full:
                    similar_indices_aiid = aiid.index[topk_indices_aiid[i]]
                    matched_description_aiid = aiid.loc[similar_indices_aiid]["description"].tolist(
                    )
                    matched_risks_aiid = aiid.loc[similar_indices_aiid]['risks'].tolist(
                    )
                    fold_results.append({
                        'fold': fold,
                        'test_index': test_index,
                        'test_description': risk_cards.loc[test_index]['model_description'],
                        'similar_indices': similar_indices.tolist(),
                        'matched_description': matched_description,
                        'matched_risks': '** '.join(list(set(matched_risks))),
                        'matched_mitigations': '** '.join(list(set(matched_mitigations))),
                        'test_risks': extracted_risks.loc[test_index]['risks'],
                        'test_mitigations': extracted_risks.loc[test_index]['mitigations'],
                        'similar_indices_aiid': similar_indices_aiid.tolist(),
                        'matched_description_aiid': matched_description_aiid,
                        'matched_risks_aiid': '** '.join(list(set(matched_risks_aiid)))
                    })
                else:
                    fold_results.append({
                        'fold': fold,
                        'test_index': test_index,
                        'test_description': risk_cards.loc[test_index]['model_description'],
                        'similar_indices': similar_indices.tolist(),
                        'matched_description': matched_description,
                        'matched_risks': '** '.join(list(set(matched_risks))),
                        'matched_mitigations': '** '.join(list(set(matched_mitigations))),
                        'test_risks': extracted_risks.loc[test_index]['risks'],
                        'test_mitigations': extracted_risks.loc[test_index]['mitigations']
                    })

            fold_results_df = pd.DataFrame(fold_results)
            all_fold_results.append(fold_results_df)

        # Concatenate all fold results into a single DataFrame
        results_df = pd.concat(all_fold_results).reset_index(drop=True)
        # results_df.to_csv(
            # "data/new_data/results/matched_risks_aiid_{}_cv_kfold_top{}.csv".format(model.split('/')[-1], TOP_K))
