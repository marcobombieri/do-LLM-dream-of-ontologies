import requests
import re
import os
import csv
from itertools import islice
import time

def get_search_results_count(query, api_key, cse_id):
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query
#        "num": 10,  # Number of results per page (max: 10)
    }

    total_results = 0
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    try:
        total_results = data.get("queries").get('request')[0]['totalResults']
    except KeyError:
        total_results=0
    except requests.exceptions.RequestException:
        total_results=0

    return total_results



if __name__ == "__main__":
    api_key = "YOUR API KEY"
    cse_id = "YOUR CSE ID"
    #input_data = os.path.join("go_with_ids_labels_annotations.csv")
    input_data = os.path.join("your_dataset.csv")
    
    start_row = 15457  # The row index to start iterating (zero-based)
    end_row = 15543    # The row index to end iterating (zero-based)
    
    #output = os.path.join("google_search_ids_labels_annotations_["+str(start_row)+"_"+str(end_row)+"].csv")
    output = os.path.join("your_dataset_["+str(start_row)+"_"+str(end_row)+"].csv")



    results = list()
    with open(input_data, 'r', newline='', encoding='utf-8') as csvfile, open(output, "w", encoding="utf-8") as outputfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        
        # Skip rows until we reach the starting row
        for _ in range(start_row):
            next(csv_reader, None)
        for index, word in enumerate(csv_reader, start=start_row):
            # Stop iterating if we've reached the end row
            if index >= end_row:
                break

            keyword = '"' + word[1] + '"' + " " + word[0]
            results_count = get_search_results_count(keyword, api_key, cse_id)
            #print(keyword, results_count)
        
            if results_count is None:
                results_count = 0

            #results.append(word[0]+";"+word[1]+";"+str(results_count)+";"+word[3])    
            #print(word[0]+";"+word[1]+";"+str(results_count)+";"+word[3])
            results.append(word[0] + ";" + word[1] + ";" + word[2] + ";" + str(results_count) )
            print(keyword)
            print(index, word[0] + ";" + word[1] + ";" + word[2] + ";" + str(results_count))
            outputfile.write(word[0] + ";" + word[1] + ";" + word[2] + ";" + str(results_count) +"\n")
            print()
            time.sleep(1)
    
'''
    with open(output, "w", encoding="utf-8") as outputfile:
        for item in results:
            print
            outputfile.write(item+"\n")
'''