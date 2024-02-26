import os

k = 1000
buckets = 50


def compute_prompt_invariance(dataset):
    with open(dataset, mode="r", encoding="utf-8") as d:
        elements = d.readlines()
        single_elements_list = [elements[i:i+26] for i in range(0, len(elements), 26)] #size = 1000
        bucket_elements_list = [single_elements_list[i:i+20] for i in range(0, len(single_elements_list), 20)] #size = 50
        
        #print(len(bucket_elements_list))
        #print(len(bucket_elements_list[0][0]))
        
        number_of_different_results = 0
        count = 0
        for i, e in enumerate(bucket_elements_list): 
            for e2 in e:
                number_of_different_results += len(set(e2))
                s = 1 - ((number_of_different_results-1) / (26-1))
                count+=s
                number_of_different_results = 0
                #MPI = count/20
            
            MPI = count / 20
            print(f"Mean prompt invariance {i}: ", MPI)
            
            count = 0
            MPI=0
   

dataset=os.path.join(".", f"SAMPLES_n_RANDOM_ELEMENTS_PER_BUCKET.txt")
compute_prompt_invariance(dataset)







