from transformers import GPTNeoXForCausalLM, AutoTokenizer
import os
import csv
#import torch

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-12b-deduped",
  revision="step143000",
  cache_dir="./pythia-12b-deduped/step143000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-12b-deduped",
  revision="step143000",
  cache_dir="./pythia-12b-deduped/step143000",
)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

dataset=os.path.join("..", "Dataset", "go_dataset.csv")
with open(dataset, mode="r", encoding="utf-8") as file:
    labels = list()
    csv_reader = csv.reader(file, delimiter=';')
    for row in csv_reader:
        #print(row[1])
        labels.append(row[1])

start = 0
end =   42854
output_name = "Pythia_PROMPT_2_go_dataset_predictions"+str(start)+"_"+str(end)+".txt"
output_path = os.path.join("..", "Dataset", output_name)


with open(output_path, mode="w", encoding="utf-8") as output:
    for l in labels[start:end]:
        #inputs = tokenizer(f"The GO ID of the label '{l}' is:", return_tensors="pt")
        inputs = tokenizer(f"In the Gene Ontology, the GO ID of the label '{l}' is GO:", return_tensors="pt")
        tokens = model.generate(**inputs, max_new_tokens=5)
        o = tokenizer.decode(tokens[0])
        print(o)
        output.write(o+"\n")
