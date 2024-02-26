import os
import pandas as pd
import csv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr 




def statistic1d(x, y):
    return spearmanr(x, y)


# Function to calculate Spearman correlation coefficient for a given permutation
def permuted_spearman(A, B):
    permuted_B = np.random.permutation(B)
    return spearmanr(A, permuted_B).correlation

def compute_correlation(list1, list2):
    num_permutations=10000
    correlation = np.corrcoef(list1, list2)
    sp, pvalue = spearmanr(list1, list2)
    print("Pearson correlation coefficient:\n", correlation[0,1])
    print("Speraman correlation coefficient:\n", sp)
    print("Sperman p-value: ", pvalue)
    
    #permutation_p_value = permutation_test(list1, list2, paired=True, method='approximate', seed=0, num_rounds=1000)
    #print(f"P-value from permutation test 1: {permutation_p_value}")
    
    permuted_correlations = [permuted_spearman(list1, list2) for _ in range(num_permutations)]

    # Calculate the p-value by comparing observed correlation to the permuted distribution
    p_value_permutation = (np.abs(permuted_correlations) >= np.abs(sp)).mean()
    print(f"P-value from permutation test: {p_value_permutation}")

    #permutation_p_value_m = sciperm((list1, list2), statistic1d, permutation_type='pairings',n_resamples=10000, alternative="two-sided")  
    #print(f"Spearman and P-value from permutation test: {permutation_p_value_m.statistic} , {permutation_p_value_m.pvalue}")

    permuted_correlations = [permuted_spearman(list1, list2) for _ in range(num_permutations)]
   
    #ref = permutation_test((list1,), statistic, alternative='greater', permutation_type='pairings')
    #print("P-value with permutation test: ", ref)
    print()

    return correlation[0,1], sp, pvalue


#--------------------------------------------------------------------#
#Function to split elements in buckets according to percentiles 
#--------------------------------------------------------------------#
def split_into_buckets_percentiles(dataset, column, percentiles, perc):
    """
    split_into_buckets_percentiles function splits elments in buckets 
    according to percentiles.

    :param dataset: input dataset containing elements to be split
    :param column: specify which column to consider to split elements into buckets. 
        It can be "annotations" or "occurrences". In this paper we used only "occurrences"
        i.e., the number of web occurrences for a given element
    :param percentiles: percentiles boundaries
    :param perc: number of percentiles
    :return: buckets containing elements split (list of list)
    """ 

    if column=='annotations':
        column_num = 2
    else: 
        column_num=3
    buckets = [[] for _ in range(perc)]
    #print(len(buckets))
    dataset_sorted = dataset.sort_values(column)
    dataset_sorted = dataset_sorted.values.tolist()
    d = 0
    #print(percentiles[0][-1])
    for element in dataset_sorted:
        #print(element[column_num])
        if element[column_num] >= percentiles[d][0] and element[column_num]<=percentiles[d][-1]:
            buckets[d].append(element)
        else:
            d+=1    
    return buckets

#-------------------------------------------------------------------------------#
#Function that comput percentiles on a list of elements
#-------------------------------------------------------------------------------#
def subdivide_into_percentile(lst, perc):
    """
    subdivide_into_percentiles function that compute {perc} percentiles on a 
    list {lst} of unique elements.

    :param lst: list of elements
    :param perc: number of percentiles to compute
    :return: percentile boundaries
    """ 

    # Remove duplicates while preserving the original order
    unique_elements = []
    seen = set()
    for item in lst:
        if item not in seen:
            seen.add(item)
            unique_elements.append(item)

    # Sort the unique elements
    sorted_elements = sorted(unique_elements)

    # Calculate the number of elements in each decile
    num_elements = len(sorted_elements)
    elements_per_decile = num_elements // perc

    # Initialize deciles
    deciles = [[] for _ in range(perc)]

    # Distribute elements into deciles
    for i, element in enumerate(sorted_elements):
        decile_index = min(i // elements_per_decile, perc-1)  
        deciles[decile_index].append(element)

    return deciles


def get_annotations_occurrences_of(e, dataset):
    """
    get_annotations_occurrences_of function extracts the number of annotations 
    and occurrences from the dataset for a specific element {e} 

    :param e: input element
    :param dataset: the dataset
    :return: the number of annotations and occurrences of {e} in {dataset}
    """ 
    selected_rows = dataset[dataset.iloc[:, 0] == e]
    if selected_rows.empty:
        annotations = -1
        occurrences = -1
    else:
        annotations = selected_rows.iloc[0]['annotations']
        occurrences = selected_rows.iloc[0]['occurrences']
        #print(annotations)
    return annotations, occurrences 

def extract_number_of_unique_predictions(dataset):
    """
    extract_number_of_unique_predictions function extracts the number of 
    unique predictions in {dataset} 

    :param dataset: the dataset
    :return: the number of unique predictions in {dataset}
    """ 
    l = list()
    predictions = dataset['prediction']
    for p in predictions:
        l.append(p)
    return len(set(l))

def extract_k_most_predicted_labels(dataset, k):
    """
    extract_k_most_predicted_labels function extracts the {k} most predicted
    labels in {dataset} 

    :param dataset: the dataset
    :k: the number of most predicted labels
    :return: the {k} most predicted labels in {dataset}
    """ 
    to_return = list()
    predictions = dataset['prediction']
    # Count occurrences of each number
    number_counts = Counter(predictions)
    #print(number_counts)
    # Get the top N most frequent numbers
    top_k_frequencies = number_counts.most_common(k)
    #print(top_k_frequencies)
    for e in top_k_frequencies:
        annotations, occurrences =  get_annotations_occurrences_of(e[0], dataset)
        v = [e[0], e[1], annotations, occurrences]
        to_return.append(v)

    return to_return

paths = [os.path.join("..", "Dataset", "pythia12B_go_dataset_with_predictions_prompt2.csv"), 
         os.path.join("..", "Dataset", "gpt-35-go_dataset_with_predictions.csv"), 
         os.path.join("..", "Dataset", "gpt4-go-predictions.csv")]


to_plot = []
models_name=["Pythia12B", "GPT-3.5", "GPT-4"]
for ip, path in enumerate(paths):
    dataset = pd.read_csv(path, sep=";", encoding="utf-8", header=None)
    column_names=["ID","label",'annotations','occurrences', 'prediction']
    dataset.columns = column_names

    perc = 50
    k = 500
    to_analyze = "occurrences"
    percentiles = subdivide_into_percentile(dataset[to_analyze], perc)
    buckets = split_into_buckets_percentiles(dataset, to_analyze, percentiles, perc)

    number_of_unique_predictions = extract_number_of_unique_predictions(dataset)
    print(f"[{models_name[ip]}] number_of_unique_predictions: ", number_of_unique_predictions)

    k_most_predicted_labels = extract_k_most_predicted_labels(dataset, k)
    #print(k_most_predicted_labels[0])

    #print(len(buckets))
    res = [0]*len(buckets)
    resprob = [0]*len(buckets)


    for e in k_most_predicted_labels:
        for i, b in enumerate(buckets):
            for b2 in b:
                if e[0] == b2[0]:
                    res[i] += 1
                    break

    #print(k_most_predicted_labels)
    sum = 0
    for r in res:
        sum+=r
    #print(sum)


    expected = []
    delta = []
    n = 42854
    #n = 15543
    for b, r in zip(buckets, res):
        expected.append((float(len(b))/n)*k)

    #for e in expected:
    #    print(e)

    for i in range(0, len(buckets)):
        delta.append(res[i] / expected[i])

    #for d in delta:
    #    print(d)

    to_plot.append(delta)


bl = list()
for i in range(1, perc+1):
    bl.append(f"B{i}")
titles = ["Pythia12B", "GPT-3.5", "GPT-4"]
plt.plot(range(0, len(res)),to_plot[0],  marker='o', label=titles[0])
plt.plot(range(0, len(res)),to_plot[1],  marker='o', label=titles[1])
plt.plot(range(0,len(res)),to_plot[2],  marker='o', label=titles[2])

selected_labels_indices = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]  # Adjust the indices as needed
selected_labels = [str(bl[i]) for i in selected_labels_indices]  # Convert to strings
plt.xticks(selected_labels_indices, labels=selected_labels, fontsize=20)

plt.xlabel('Bucket number', fontsize=20)
#plt.xticks(range(1, len(bl) + 1), labels=bl, fontsize=20)
plt.ylabel("Actual over expected top-500 repeated IDs per bucket", fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()

#compute_correlation(range(1, len(res)+1), to_plot[0])
#compute_correlation(range(1, len(res)+1), to_plot[1])
#compute_correlation(range(1, len(res)+1), to_plot[2])