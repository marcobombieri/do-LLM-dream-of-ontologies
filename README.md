# Do LLM dream of ontologies?

Repository containing code and dataset of the paper "Do LLM Dream Of Ontologies?" <br>

## Repository description
The repository is composed by 3 sub-directories:

__Dataset__ <br>
This directory contains the Gene and Uberon datasets. In particular:
  - go_dataset.csv: it contains for each GO concept its ID, label, number of annotations in ontologies (not considered in this paper) and the number of web occurrences
  - uberon_dataset.csv: it contains for each UBERON concept its ID, label, number of annotations in ontologies (not considered in this paper) and the number of web occurrences  
  - pythia12B_go_dataset_with_predictions_prompt2.csv: it contains the Pythia12B predictions (5th field) of the GO elements
  - gpt-35-go_dataset_with_predictions.csv: it contains the GPT-3.5 predictions (5th field) of the GO elements 
  - gpt4-go_predictions.csv: it contains the GPT-4 predictions (5th field) of the GO elements
  - sorted_by_occurrences_pythia12B_UBERON_dataset_prompt2.csv: it contains the Pythia12B predictions (5th field) of the UBERON elements
  - sorted_by_occurrences_gpt35_uberon.csv: it contains the GPT-3.5 predictions (5th field) of the UBERON elements
  - sorted_by_occurrences_gpt4_uberon.csv: it contains the GPT-4 predictions (5th field) of the UBERON elements

__Perfromance_analysis__ <br>
This directory contains the code to reproduce the results of the submitted paper. In particular:
  - google_search_final.py: code to compute the number of web occurrences for a given <ID;label> string. Used to produce the datasets in the Dataset directory
  - compute_performance.py: code to compute the correlation between performance and number of occurrences and to do the error Perfromance_analysis
  - extract_k_most_predicted_labels.py: code to extract the k most predicted labels and to plot their distribution in buckets
  - plot_dataset_statistics.py: code to plot the datasets statistics
  - prompt_from_label_to_ID_GO_Pythia12B.py: Pythia prompt to GO
  - prompt_from_label_to_ID_GO_GPT-35.py: GPT-3.5 prompt to GO
  - prompt_from_label_to_ID_GO_GPT-4.py: GPT-4 prompt to GO
  - prompt_from_label_to_ID_UBERON_Pythia12B.py: Pythia prompt to UBERON
  - prompt_from_label_to_ID_UBERON_GPT-35.py: GPT-3.5 prompt to UBERON
  - prompt_from_label_to_ID_UBERON_GPT-4.py: GPT-4 prompt to UBERON


__Prompt invariance__ <br>
This directory contains the code to reproduce the results of the Prompt Invariance proposed method. In particular:
  - extracted_labels.csv: it contains random elements extracted from the GO
  - prompt_invariance_experiments.py: code for prompting PI-1, PI-2 and PI-3
  - compute_accuracy_per_bucket.py: code to plot AvPI and Accuracy per buckets
  - evaluate_per_bucket.py: code to compute the prompt invariance
  - 1_same_prompt_output.txt, 2_different_temperatures_output.txt, 3_different_languages_output.txt and SAMPLES_n_RANDOM_ELEMENTS_PER_BUCKETS.txt are output files
