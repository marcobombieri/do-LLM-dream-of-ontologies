import openai
import os
import csv
import time

#from dotenv import load_dotenv, find_dotenv
#_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = 'YOUR_KEY_HERE'



def get_completion(prompt, model="gpt-4"):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0, # this is the degree of randomness of the model's output
            request_timeout=30,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print("Exeception ", e)
        return None


dataset=os.path.join("..", "Dataset", "go_dataset.csv")

with open(dataset, mode="r", encoding="utf-8") as file:
    labels = list()
    csv_reader = csv.reader(file, delimiter=';')
    for row in csv_reader:
        #print(row[1])
        labels.append(row[1])

#for l in labels:
#    print(l)

start = 0
end =   42854
output_name = "gpt4_output_"+str(start)+"_"+str(end)+".txt"
output_path = os.path.join("..", "Output", output_name)

with open(output_path, mode="w", encoding="utf-8") as output:
    for i, l in enumerate(labels[start:end]):
        while True:
            prompt = f"""
            Provide the GO ID for the label '''{l}'''. In the answer write only the corresponding GO ID.
            """
            response = get_completion(prompt)
            if response:
                print(i, " ", l, ": ", response)
                output.write(response+"\n")
                #time.sleep(1)
                break
            else:
                print("Timeout error: retrying after 10 seconds")
                time.sleep(180)
    



