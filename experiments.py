import numpy as np
import sklearn
import pandas as pd
from sklearn.decomposition import NMF as NMF_p
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

def NMF(A, r):
    mod =  NMF_p(r, max_iter=500)
    return mod.fit_transform(A), mod.components_

from stratified_nmf import stratified_nmf
from estratified_nmf import estratified_nmf


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


def prep_data():
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                       min_df=0.05,
                                       stop_words="english")
    
    data = []
    indices = [0]
    for team in ["michigan", "osu", "psu", "msu", "ucla", "usc", "iowa", "wisconsin"]:
    
        teamdata = np.load(f"data/{team}_articles_out.npy", allow_pickle=True)
        teamdata = [art['text'] for art in teamdata if len(art['text']) > 400]
        np.save(f"data/{team}_text.npy", np.array(teamdata))
        indices.append(len(teamdata) + indices[-1] +1)
        data.extend(teamdata)


    all_m = tfidf_vectorizer.fit_transform(data).toarray()
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return all_m, [all_m[indices[i]:indices[i+1]] for i in range(len(indices)-1)], feature_names, indices


def top_v_newsgroup_words(V, feature_names, num_words):
    """Gets the words with highest value in V for each stratum

    Args:
        V: List of vectors (strata features)
        feature_names: Array of words
        num_words: Number of words to return

    Returns:
        Two dimensional array of the top words for each stratum
    """
    max_nonzero_in_group = np.max(np.count_nonzero(V, axis=1))
    if num_words > max_nonzero_in_group:
        print("num_words exceeds the maximum of non-zero entries in V(i) for some i.")
        print(f"Using num_words = {max_nonzero_in_group} instead")
    top_indices = np.argsort(V, axis=1)[:, -num_words:]

    top_words = [feature_names[top_indices[i]] for i in range(V.shape[0])]
    return np.array(top_words)

def compare(data=None, rank=3, n_out = 10, s_rank = 2,reg=1, tex_out=False):
    np.set_printoptions(linewidth=200)
    if data is None:
        A_both, A_strat, feature_names, indices = prep_data()
    else:
        A_both, A_strat, feature_names = data
    
    W, H = NMF(A_both, rank)
    V_s, W_s, H_s, loss_array = stratified_nmf(A_strat, rank, 20)
    V_e, X_e, W_e, H_e, loss_array = estratified_nmf(A_strat, s_rank, rank, 20, reg=reg)
    
    print("Combined NMF:")
    nmf_tops = top_v_newsgroup_words(H, feature_names, n_out)
    print(nmf_tops)
    print()
    print("-"*50)
    print("Stratified NMF:\n")
    print("Strata features V:")
    stratified_features = top_v_newsgroup_words(V_s, feature_names, n_out)
    print(stratified_features)
    print("\nTopics H: ")
    stratified_topics = top_v_newsgroup_words(H_s, feature_names, n_out)
    
    count_shape = list(stratified_topics.shape)
    count_shape[1] += len(A_strat) 
    stratified_counts = np.zeros(tuple(count_shape), dtype='object')
    stratified_counts[:, :stratified_topics.shape[1]] = stratified_topics
    for j,a in enumerate(stratified_topics):
        counts = [np.count_nonzero(np.argmax(Wa, axis=1) == j) for Wa in W_s]
        print(j, a, counts)
        stratified_counts[j, stratified_topics.shape[1]:] = counts
        
    print()
    print("-"*50)
    print("Extended Stratified NMF:\n")
    strata_topics = []
    for i in range(len(A_strat)):
        print(f"Strata topic X[{i}]:")
        strata_topic = top_v_newsgroup_words(X_e[i], feature_names, n_out)
        print(strata_topic)
        strata_topics.append(strata_topic)
            
    print("\nTopics H: ")
    global_topics = top_v_newsgroup_words(H_e, feature_names, n_out)
     
    count_shape = list(global_topics.shape)
    count_shape[1] += len(A_strat)
    estratified_counts = np.zeros(tuple(count_shape), dtype='object')
    estratified_counts[:, :global_topics.shape[1]] = global_topics
    
    for j,a in enumerate(global_topics):
        counts = [np.count_nonzero(np.argmax(Wa, axis=1) == j) for Wa in W_e]
        print(j, a, counts)
        estratified_counts[j, global_topics.shape[1]:] = counts
        
    
    if tex_out:

        return nmf_tops, stratified_features, stratified_counts, strata_topics, estratified_counts
    
    return V_e, X_e, W_e, H_e, loss_array

def quantitative_results(data=None, rank=10, s_rank = 3, n_trials=10):
    np.set_printoptions(linewidth=200)
    
    if data is None:
        A_both, A_strat, feature_names, indices = prep_data()
    else:
        A_both, A_strat, feature_names = data
    
    standard_deviations = np.zeros((3, n_trials))
    data_fidelity = np.zeros((3, n_trials))
    
    for trial in tqdm(range(n_trials)):
        W, H = NMF(A_both, rank)
        V_s, W_s, H_s, loss_array = stratified_nmf(A_strat, rank, 20, hide_bar=True)
        V_e, X_e, W_e, H_e, eloss_array = estratified_nmf(A_strat, s_rank, rank, 20,hide_bar=True)
        
        data_fidelity[0, trial] = np.linalg.norm(A_both - W @ H, ord='fro')
        data_fidelity[1, trial] = loss_array[-1]
        data_fidelity[2, trial] = eloss_array[-1]
        
        
        count_shape = (rank, len(A_strat))
        classical_counts = np.zeros(tuple(count_shape))
        stratified_counts = np.zeros(tuple(count_shape))
        estratified_counts = np.zeros(tuple(count_shape))
        
        Wsbystrata = [W[indices[i]:indices[i+1]] for i in range(len(A_strat))]
        for i in range(rank):
            
            counts = [np.count_nonzero(np.argmax(Wa, axis=1) == i) for Wa in Wsbystrata]
            classical_counts[i] = counts
        
        
        for i in range(rank):
            counts = [np.count_nonzero(np.argmax(Wa, axis=1) == i) for Wa in W_s]
            stratified_counts[i] = counts
        
        for i in range(rank):
            counts = [np.count_nonzero(np.argmax(Wa, axis=1) == i) for Wa in W_e]
            estratified_counts[i] = counts

        standard_deviations[0, trial] = np.median(np.std(classical_counts, axis=1))
        standard_deviations[1, trial] = np.median(np.std(stratified_counts, axis=1))
        standard_deviations[2, trial] = np.median(np.std(estratified_counts, axis=1))
    
    return data_fidelity, standard_deviations
    

    
    
    
    
