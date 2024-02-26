import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.metrics import jaccard_distance
from nltk.tokenize import word_tokenize
import re
import math
from collections import Counter
from scipy.stats import spearmanr
#from mlxtend.evaluate import permutation_test
from scipy.stats import spearmanr
#from scipy.stats import permutation_test as sciperm
#from scipy.stats import t

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


#---------------------------------------------------------------------#
#Function that calculates the performance on the bucket passed as input
#---------------------------------------------------------------------#
def compute_performance(dataset):
    """
    compute_performance function compute the accuracy of the model on 
    the bucket passed as input 

    :param dataset: input dataset containing the bucket to analyze
    :return: the accuracy obtained in that dataset
    """ 
    correct = 0
    total = 0
    for row in dataset:
        if row[0] == row[4]:
            correct +=1
        total+=1
    accuracy = correct/total*100
    return accuracy

#--------------------------------------------------------------#
#Function that return the mean of a list of values
#--------------------------------------------------------------#
def compute_mean(dataset, column):
    """
    compute_mean function compute the mean of a list of values 

    :param dataset: input dataset containing the elements to analyze
    :param column: specify which column to consider. 
        It can be "annotations" or "occurrences". In this paper we used only "occurrences"
        i.e., the number of web occurrences for a given element
    :return: the mean of the elemnents present in the dataset at the spicified column
    """ 
    if column == "annotations":
        c = 2
    else:
        c = 3
    elements = list()
    for row in dataset:
        elements.append(row[c])
    return np.mean(elements)

#--------------------------------------------------------------#
#Plot performance
#--------------------------------------------------------------#
def plot_performance(performance, x):
    """
    plot_performance function plots the elements passed as input 

    :param performance: performance to plot
    :param x: values to plot
    :return: None
    """ 
    plt.plot(performance, x, marker='o')
    plt.title('List Plot')
    plt.xlabel('Mean of the bucket')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

#--------------------------------------------------------------#
#Error Analysis
#--------------------------------------------------------------#
def jaccard_similarity(str1, str2):
    """
    jaccard_similarity function computes the jaccard similarity of the strings passed as input 

    :param str1: the first string
    :param str2: the second string
    :return: the Jaccard similarity value between str1 and str2
    """ 
    # Tokenize the strings
    tokens1 = set(word_tokenize(str1.lower()))
    tokens2 = set(word_tokenize(str2.lower()))
    jaccard_sim = 1 - jaccard_distance(tokens1, tokens2)    # Calculate Jaccard similarity
    return jaccard_sim

def levenshtein_distance(word1, word2):
    """
    levensthein_distance computes the Levenshtein distance of the IDs passed as input 

    :param word1: the first ID
    :param word2: the second ID
    :return: the Levensthein distance between word1 and word2
    """ 
    m, n = len(word1), len(word2)    # Create a matrix to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): # Initialize the first row and column
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1): # Fill in the matrix using dynamic programming
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                           dp[i][j - 1] + 1,  # Insertion
                           dp[i - 1][j - 1] + substitution_cost)  # Substitution
    return dp[m][n]

def cosine_similarity(s1, s2):
    """
    cosine_similarity computes the cosine similarity of the two strings passed as input 

    :param s1: the first string
    :param s2: the second string
    :return: the cosine similarity between s1 and s2
    """ 
    words_s1 = word_tokenize(s1)
    words_s2 = word_tokenize(s2)
    vector1 = Counter(words_s1)
    vector2 = Counter(words_s2)
    all_words = set(vector1.keys()) | set(vector2.keys())
    dot_product = sum(vector1[word] * vector2[word] for word in all_words)
    magnitude1 = math.sqrt(sum(vector1[word] ** 2 for word in all_words))
    magnitude2 = math.sqrt(sum(vector2[word] ** 2 for word in all_words))
    if magnitude1 == 0 or magnitude2 == 0:    # Check if either magnitude is zero before dividing
        return 0.0  
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity

def extract_errors_in_buckets(dataset, bucket):
    """
    extract_errors_in_buckets extracts the wrong predictions of the model in a given bucket 

    :param dataset: datased to analyze
    :param bucket: bucket to analyze
    :return: a list containing the mean Jaccard similarity, the mean cosine similarity and 
        the mean Levenshtein distance in that bucket
    """ 
    errors = list()
    jaccard_l = list()
    lev_l = list()
    cos_l = list()
    for row in bucket:
        if row[0] != row[4]:
            filtered_df = dataset[dataset['ID'] == row[4]]
            if not filtered_df.empty:
                i = filtered_df.iloc[0, 0]
                l = filtered_df.iloc[0, 1]
                a = filtered_df.iloc[0, 2]
                o = filtered_df.iloc[0, 3] 
            else:
                i=""
                l=""
                a=""
                o=""
            jaccard =  jaccard_similarity(row[1], l)
            jaccard_l.append(jaccard)
            levensthein = levenshtein_distance(row[0], i)
            lev_l.append(levensthein)
            cos = cosine_similarity(row[1], l)
            cos_l.append(cos)
            #errors.append([row[0], row[1], row[2], row[3], row[4], l, a, o, jaccard])
            errors.append([row[0], row[1], row[2], row[3], row[4], l, a, o])
    #print(np.mean(jaccard_l))
    return np.mean(jaccard_l), errors, np.mean(lev_l), np.mean(cos_l)

def get_valid_numbers(numbers, e):
    """
    get_valid_number is an auxiliary function that substitute NaN elements in a list with a default value 

    :param numbers: a list of numbers
    :param e: the default value
    :return: a list where the NaN elements are substituted with "e"
    """         
    valid_numbers = list()
    for n in numbers:
        if not np.isnan(n):
            valid_numbers.append(n)
        else: 
            valid_numbers.append(e)
    return valid_numbers

#--------------------------------------------------------------#
#Main code to reproduce the results of this paper
#--------------------------------------------------------------#
paths_go_occ = [os.path.join("..", "Dataset", "pythia12B_go_dataset_with_predictions_prompt2.csv"), #Pythia predictions on the GO (sorted by occurrences)
         os.path.join("..", "Dataset", "gpt-35-go_dataset_with_predictions.csv"),  #GPT-3.5 predictions on the GO (sorted by occurrences)
         os.path.join("..", "Dataset", "gpt4-go-predictions.csv")] #GPT-4 predictions on the GO (sorted by occurrences)

paths_ub = [os.path.join("..", "Dataset", "sorted_by_occurrences_pythia12B_UBERON_dataset_prompt2.csv"), #Pythia predictions on the Uberon (sorted by occurrences)
         os.path.join("..", "Dataset", "sorted_by_occurrences_gpt35_uberon.csv"), #GPT-3.5 predictions on the Uberon (sorted by occurrences)
         os.path.join("..", "Dataset", "sorted_by_occurrences_gpt4_uberon.csv")] #GPT-4 predictions on the Uberon (sorted by occurrences)

to_analyze = "occurrences"
perc = 50
#perc = 3
to_print_x = list()
to_print_y = list()
to_print_j = list()
to_print_c = list()
to_print_l = list()

models = ["Pythia12B", "GPT-3.5", "GPT-4"]
dataset_name = ""
for counter, path in enumerate(paths_go_occ):
    if "uberon" in str(path.lower()): 
        dataset_name="UBERON ONTOLOGY"
    else:
        dataset_name="GENE ONTOLOGY"
    print("\n#--------------------------------------------------------#")
    print(f"#--- Model {models[counter]} on {dataset_name}")
    print("#--------------------------------------------------------#")
    #----------------------------------------------------------------------------------#
    print("--- Performance per bucket: ")
    dataset = pd.read_csv(path, sep=";", encoding="utf-8", header=None)
    #print(dataset)
    column_names=["ID","label",'annotations','occurrences', 'prediction']
    dataset.columns = column_names
    percentiles = subdivide_into_percentile(dataset[to_analyze], perc)

    buckets = split_into_buckets_percentiles(dataset, to_analyze, percentiles, perc)
    accuracy_per_bucket = list()
    mean_per_bucket = list()
    for b in buckets:
        accuracy = compute_performance(b)
        accuracy_per_bucket.append(accuracy)
        mean = compute_mean(b, to_analyze)
        mean_per_bucket.append(mean)

    for i, a in enumerate(accuracy_per_bucket):
        print("Bucket ", i, " has accuracy: ", a)

    correct = 0
    total_accuracy = compute_performance(dataset.values.tolist()[1:])
    print("Total accuracy: ", total_accuracy)

    #plot_performance(mean_per_bucket, accuracy_per_bucket)
    #for i, b in enumerate(buckets):
    #    print("Bucket ", i, " has size: ", len(b))
    
    to_print_x.append(mean_per_bucket)
    to_print_y.append(accuracy_per_bucket)

    print("\n--- correlation between number of occurrences per bucket and accuracy per buket:")
    compute_correlation(mean_per_bucket, accuracy_per_bucket)

    print("--- Error analysis per buket (it may take some time):")
    jaccard_m_l = list()
    levensthein_m_l = list()
    cos_m_l = list()
    mean_per_bucket_2 = list()
    for i, b in enumerate(buckets):
        mj, errors, ml, mc = extract_errors_in_buckets(dataset, b)
        jaccard_m_l.append(mj)
        levensthein_m_l.append(ml)
        cos_m_l.append(mc)
        mean_per_bucket_2.append(mean_per_bucket[i])
        print(f"Jaccard similarity Bucket {i}: {mj}")
        print(f"Levenshtein distance Bucket {i}: {ml}")
        print(f"Cosine similarity Bucket {i}: {mc}")
        #jaccard_m_l.append(mj)
        
    to_print_j.append(jaccard_m_l)
    to_print_l.append(levensthein_m_l)
    to_print_c.append(cos_m_l)

#mjd, errorsd, mld, mcd = extract_errors(dataset)
valid_numbers = [num for num in jaccard_m_l if not np.isnan(num)]
print("Jaccard: ",np.mean(valid_numbers))
valid_numbers = [num for num in levensthein_m_l if not np.isnan(num)]
print("Lev: ", np.mean(valid_numbers))
valid_numbers = [num for num in cos_m_l if not np.isnan(num)]
print("Cosine: ", np.mean(valid_numbers))
#print(len(jaccard_m_l), len(mean_per_bucket_2))
#plot_performance(mean_per_bucket_2, jaccard_m_l)
#plot_performance(mean_per_bucket_2, levensthein_m_l)
#plot_performance(mean_per_bucket_2, cos_m_l)


titles = ["Pythia12B", "GPT-3.5", "GPT-4"]
#plt.plot(to_print_x[0], to_print_y[0],  marker='o', label=titles[0])
#plt.plot(to_print_x[1], to_print_y[1],  marker='o', label=titles[1])
#plt.plot(to_print_x[2], to_print_y[2],  marker='o', label=titles[2])
plt.plot(range(1, 51), to_print_y[0],  marker='o', label=titles[0])
plt.plot(range(1, 51), to_print_y[1],  marker='o', label=titles[1])
plt.plot(range(1, 51), to_print_y[2],  marker='o', label=titles[2])

#plt.title(titles[ip])
plt.xlabel("Bucket number in dataset sorted by occurrences")
plt.ylabel("Average accuracy per bucket")
plt.grid(True)
plt.legend()
plt.show()

'''
titles = ["Pythia12B", "GPT-3.5", "GPT-4"]
yl = ["Jaccard similarity between gold and predicted labels", "Cosine similarity between gold and predicted labels", "Levensthein distance between gold ID and predicted ID"]

plt.plot(range(1, 51), get_valid_numbers(to_print_j[0], 1),  marker='o', label=titles[0])
plt.plot(range(1, 51), get_valid_numbers(to_print_j[1], 1),  marker='o', label=titles[1])
plt.plot(range(1, 51), get_valid_numbers(to_print_j[2], 1),  marker='o', label=titles[2])

#plt.title(titles[ip])
plt.xlabel("Bucket number")
plt.ylabel(yl[0])
plt.grid(True)
plt.legend()
plt.show()
'''
print("- Pythia12B: correlation between number of occurrences per bucket and Jaccard similarity per buket:")
compute_correlation(mean_per_bucket, get_valid_numbers(to_print_j[0], 1))
print("- GPT-3.5: correlation between number of occurrences per bucket and Jaccard similarity per buket:")
compute_correlation(mean_per_bucket, get_valid_numbers(to_print_j[1], 1))
print("- GPT-4: correlation between number of occurrences per bucket and Jaccard similarity per buket:")
compute_correlation(mean_per_bucket, get_valid_numbers(to_print_j[2], 1))

'''
plt.plot(range(1, 51), get_valid_numbers(to_print_c[0], 1),  marker='o', label=titles[0])
plt.plot(range(1, 51), get_valid_numbers(to_print_c[1], 1),  marker='o', label=titles[1])
plt.plot(range(1, 51), get_valid_numbers(to_print_c[2], 1),  marker='o', label=titles[2])
#plt.title(titles[ip])
plt.xlabel("Bucket number")
plt.ylabel(yl[1])
plt.grid(True)
plt.legend()
plt.show()
'''

print("- Pythia12B: correlation between number of occurrences per bucket and Cosine similarity per buket:")
compute_correlation(mean_per_bucket, get_valid_numbers(to_print_c[0], 1))
print("- GPT-3.5: correlation between number of occurrences per bucket and Cosine similarity per buket:")
compute_correlation(mean_per_bucket, get_valid_numbers(to_print_c[1], 1))
print("- GPT-4: correlation between number of occurrences per bucket and Cosine similarity per buket:")
compute_correlation(mean_per_bucket, get_valid_numbers(to_print_c[2], 1))

'''
plt.plot(range(1, 51), get_valid_numbers(to_print_l[0], 0),  marker='o', label=titles[0])
plt.plot(range(1, 51), get_valid_numbers(to_print_l[1], 0),  marker='o', label=titles[1])
plt.plot(range(1, 51), get_valid_numbers(to_print_l[2], 0),  marker='o', label=titles[2])
#plt.title(titles[ip])
plt.xlabel("Bucket number")
plt.ylabel(yl[2])
plt.grid(True)
plt.legend()
plt.show()
'''
print("- Pythia12B: correlation between number of occurrences per bucket and Levenhstein distance per buket:")
compute_correlation(mean_per_bucket, get_valid_numbers(to_print_l[0], 0))
print("- GPT-3.5: correlation between number of occurrences per bucket and Levenhstein distance per buket:")
compute_correlation(mean_per_bucket, get_valid_numbers(to_print_l[1], 0))
print("- GPT-4: correlation between number of occurrences per bucket and Levenhstein distance per buket:")
compute_correlation(mean_per_bucket, get_valid_numbers(to_print_l[2], 0))
