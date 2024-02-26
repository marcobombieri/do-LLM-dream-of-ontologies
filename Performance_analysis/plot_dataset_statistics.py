import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print(percentiles[0][-1])
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
#Main
#--------------------------------------------------------------#
paths_go_occ = [os.path.join("..", "Dataset", "pythia12B_go_dataset_with_predictions_prompt2.csv"), 
         os.path.join("..", "Dataset", "gpt-35-go_dataset_with_predictions.csv"), 
         os.path.join("..", "Dataset", "gpt4-go-predictions.csv")]

paths_go_ann = [os.path.join("..", "Dataset", "sorted_by_annotations_pythia_prompt_2_dataset.csv"), 
         os.path.join("..", "Dataset", "sorted_by_annotations_dataset.csv"), 
         os.path.join("..", "Dataset", "sorted_by_annotations_gpt4_go.csv")]

paths_ub = [os.path.join("..", "Dataset", "sorted_by_occurrences_pythia12B_UBERON_dataset_prompt2.csv"), 
         os.path.join("..", "Dataset", "sorted_by_occurrences_uberon_dataset.csv"), 
         os.path.join("..", "Dataset", "sorted_by_occurrences_gpt4_uberon.csv")]

to_analyze = "occurrences"
perc = 50

path=paths_go_occ[1]
#----------------------------------------------------------------------------------#
dataset = pd.read_csv(path, sep=";", encoding="utf-8", header=None)
print(dataset)
column_names=["ID","label",'annotations','occurrences', 'prediction']
dataset.columns = column_names
percentiles = subdivide_into_percentile(dataset[to_analyze], perc)
occurrences =  dataset.occurrences
occurrences = np.sort(occurrences)
buckets = split_into_buckets_percentiles(dataset, to_analyze, percentiles, perc)
accuracy_per_bucket = list()
elements_per_bucket = list()
mean_per_bucket = list()
for b in buckets:
    accuracy = compute_performance(b)
    accuracy_per_bucket.append(accuracy)
    mean = compute_mean(b, to_analyze)
    mean_per_bucket.append(mean)
    elements_per_bucket.append(len(b))

bl = list()
for i in range(1, perc+1):
    bl.append(f"B{i}")
fig, ax1 = plt.subplots() #Create the main plot
ax1.plot(bl, elements_per_bucket, 'bo-', label='Number of elements') # Plot the first y-axis data
ax1.set_xlabel('Bucket number', fontsize=25)
ax1.set_ylabel('Number of elements', color='black', fontsize=25)
ax1.set_xticklabels(bl, fontsize=25)
ax1.tick_params('y', colors='b', labelsize=25)
#ax1.set_yticklabels(elements_per_bucket, fontsize=16, rotation=90)
ax2 = ax1.twinx() # Create a second y-axis
ax2.plot(bl, mean_per_bucket, 'ro-', label='Average number of web occurrences')
ax2.set_ylabel('Average number of web occurrences', color='black', fontsize=25)
ax2.tick_params('y', colors='r', labelsize=25)
ax1.legend(loc='upper left',  bbox_to_anchor=(0.4, 1.0), fontsize=25) # Add a legend
ax2.legend(loc='upper left',  bbox_to_anchor=(0.4, 0.93), fontsize=25)
#plt.yticks(fontsize=16)
xticks_positions = [bl[i] for i in [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]]
ax1.set_xticks(xticks_positions)
#ax1.set_xticks(bl)
ax1.tick_params(axis='x', labelsize=25)
#plt.title('Number of elements and average number of web occurrences per bucket', fontsize=16)
plt.grid()
plt.show()

  