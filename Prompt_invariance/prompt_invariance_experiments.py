import os
import csv
import numpy as np
import pandas as pd

import re
import math
from collections import Counter
import random
import time
import openai

openai.api_key  = 'YOUR_API_KEY'


#-------------------------------------------------------------------------------#
#Prompt
#-------------------------------------------------------------------------------#
def get_completion(prompt, model="gpt-3.5-turbo", temp=0.0):
    """
    get_completion function sends a prompt to an OpenAI LLM

    :param prompt: your prompt
    :param model: OpenAI model
    :temp: temperature
    :return: the response
    """ 
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temp, # this is the degree of randomness of the model's output
            request_timeout=30,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print("Exeception ", e)
        return None

def do_prompt_different_temperature(l, temp_list, output):
    """
    do_prompt_different_temperature function prompts a query using different temperatures

    :param l: label
    :param temp_list: list of temperature levels to test
    :param output: output file
    :return: None
    """
    i = 0
    while True:
        prompt = f"""
        Provide the GO ID for the label '''{l}'''. In the answer write only the corresponding GO ID.
        """
        #response = get_completion(prompt, temp=temp_list[i])
        response = get_completion(prompt, temp=temp_list[i])
        if response:
            i+=1
            print(l, ": ", response)
            output.write(response+"\n")
            if i==len(temp_list):
                break
        else:
            print("Timeout error: retrying after 10 seconds")
            time.sleep(10)


def do_prompt_different_languages(l, output):
    """
    do_prompt_different_languages function prompts a query using different languages

    :param l: label
    :param output: output file
    :return: None
    """    
    #ENG
    while True:
        prompt = f"""
        Provide the GO ID for the label '''{l}'''. In the answer write only the corresponding GO ID.
        """
        #response = get_completion(prompt, temp=temp_list[i])
        response = get_completion(prompt, temp=0.0)
        if response:
            print(l, ": ", response)
            output.write(response+"\n")
            break
        else:
            print("Timeout error: retrying after 10 seconds")
            time.sleep(10)
    
    #ITA
    while True:
        prompt = f"""
        Fornire la GO ID per l'etichetta '''{l}'''. Nella risposta scrivi solo la GO ID corrispondente.
        """
        #response = get_completion(prompt, temp=temp_list[i])
        response = get_completion(prompt, temp=0.0)
        if response:
            #i+=1
            print(l, ": ", response)
            output.write(response+"\n")
            #if i==len(temp_list):
            break
        else:
            print("Timeout error: retrying after 10 seconds")
            time.sleep(10)
    
    #FRA
    while True:
        prompt = f"""
        Fournissez le GO ID pour le libellé '''{l}'''. Dans la réponse, écrivez uniquement le GO ID correspondant.
        """
        #response = get_completion(prompt, temp=temp_list[i])
        response = get_completion(prompt, temp=0.0)
        if response:
            #i+=1
            print(l, ": ", response)
            output.write(response+"\n")
            #if i==len(temp_list):
            break
        else:
            print("Timeout error: retrying after 10 seconds")
            time.sleep(10)
    
    #DEU
    while True:
        prompt = f"""
        Geben Sie die GO-ID für das Label '''{l}''' an. Schreiben Sie in die Antwort nur die entsprechende GO-ID.
        """
        #response = get_completion(prompt, temp=temp_list[i])
        response = get_completion(prompt, temp=0.0)
        if response:
            #i+=1
            print(l, ": ", response)
            output.write(response+"\n")
            #if i==len(temp_list):
            break
        else:
            print("Timeout error: retrying after 10 seconds")
            time.sleep(10)
    
    #ESP
    while True:
        prompt = f"""
        Proporcione el ID de GO para la etiqueta '''{l}'''. En la respuesta escriba solo el GO ID correspondiente.
        """
        #response = get_completion(prompt, temp=temp_list[i])
        response = get_completion(prompt, temp=0.0)
        if response:
            #i+=1
            print(l, ": ", response)
            output.write(response+"\n")
            #if i==len(temp_list):
            break
        else:
            print("Timeout error: retrying after 10 seconds")
            time.sleep(10)

random_elements_per_bucket = list()

with open("extracted_labels.csv", mode="r", encoding="utf-8") as random_labels:
    lines = random_labels.readlines()
    for l in lines[:]:
        fields = l.split(";")
        random_elements_per_bucket.append(fields[1])
        #print(fields[1]) 


with open("samples_of_n_random_elements_per_bucket.txt", mode="w", encoding="utf-8") as output:
    for r2 in random_elements_per_bucket:
        #for r2 in r:
        do_prompt_different_temperature(r2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], output)
        do_prompt_different_temperature(r2, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], output)
        do_prompt_different_languages(r2, output)
        output.write("\n")

