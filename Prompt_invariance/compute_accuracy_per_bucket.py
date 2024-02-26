import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np


def compute_performance(file, repetitions, golds):
    """
    compute_perfromance computes the prompt invariance metrics

    :param file: input file to analyze
    :param repetitions: number of prompt repetition per label
    :param golds: true IDs
    :return: list of prompt invariance per bucket and accuracy per bucket
    """
    with open(file, mode="r", encoding="utf-8") as file:
        elements = file.readlines()
        single_elements_list = [elements[i:i+repetitions] for i in range(0, len(elements), repetitions)] #size = 1000
        bucket_elements_list = [single_elements_list[i:i+20] for i in range(0, len(single_elements_list), 20)] #size = 50
    
        number_of_different_results = 0
        count = 0
        same = 0 
        PI_per_bucket = list()
        ACC_per_bucket = list()
        j=0
        for i, e in enumerate(bucket_elements_list): 
            for e2 in e:
                number_of_different_results += len(set(e2))
                s = 1 - ((number_of_different_results-1) / (repetitions-1))
                count+=s
                number_of_different_results = 0        
                for e3 in e2:
                    #print(e3[:-1], golds[j])
                    if e3[:-1] == golds[j]:
                        same = same+1
                j=j+1
            #print(same)
            accuracy = same/(repetitions*20)
            MPI = count / 20
            print(f"Mean prompt invariance {i}: ", MPI)
            PI_per_bucket.append(MPI)
            print(f"Mean accuracy invariance {i}: ", accuracy, "\n")
            ACC_per_bucket.append(accuracy)
            count = 0
            MPI=0
            same=0

    return PI_per_bucket, ACC_per_bucket

    
    

dataset_sp = os.path.join(".", "1_same_prompt_output.txt")
dataset_dt = os.path.join(".", "2_different_temperatures_output.txt")
dataset_dl = os.path.join(".", "3_different_languages_output.txt")
dataset_labels = os.path.join("extracted_labels.csv")

datasets = [dataset_sp, dataset_dt, dataset_dl]

golds = list() # contiene la lista delle gold ID (1000, 20 per bucket)
with open(dataset_labels, mode="r", encoding="utf-8") as file_labels:
    gold_labels = file_labels.readlines()
    for g in gold_labels:
        fields = g.split(";")
        golds.append(fields[0])

PI_sp, Ac_sp = compute_performance(dataset_sp, 10, golds)
PI_dt, Ac_dt = compute_performance(dataset_dt, 11, golds)
PI_dl, Ac_dl = compute_performance(dataset_dl, 5, golds)

titles = ["PI-1","PI-2","PI-3"]
plt.plot(range(1, 51), PI_sp, marker='o', label=titles[0])
plt.plot(range(1, 51), PI_dt, marker='o', label=titles[1])
plt.plot(range(1, 51), PI_dl, marker='o', label=titles[2])
plt.xlabel("Bucket number", fontsize=20)
plt.ylabel("AvPI", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()

titles = ["PI-1","PI-2","PI-3"]
plt.plot(range(1, 51), Ac_sp, marker='o', label=titles[0])
plt.plot(range(1, 51), Ac_dt, marker='o', label=titles[1])
plt.plot(range(1, 51), Ac_dl, marker='o', label=titles[2])
plt.xlabel("Bucket number",fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()

correlation = np.corrcoef(PI_sp, Ac_sp)
sp, pvalue = spearmanr(PI_sp, Ac_sp)
print(correlation)
print(sp, pvalue)

correlation = np.corrcoef(PI_dt, Ac_dt)
sp, pvalue = spearmanr(PI_dt, Ac_dt)
print(correlation)
print(sp, pvalue)

correlation = np.corrcoef(PI_dl, Ac_dl)
sp, pvalue = spearmanr(PI_dl, Ac_dl)
print(correlation)
print(sp, pvalue)