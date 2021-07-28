# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:43:54 2021

@author: MJH
"""
import pandas as pd
import json
from tqdm import tqdm



def load_data(path, sep = '\t'):
    
    data = []
    with open(path, 'r', encoding = 'utf-8') as f: 
        for datum in tqdm(f):
            data.append(json.loads(datum))
            
    dataframe = pd.DataFrame(data)
    dataframe.dropna(inplace = True)
    
    return dataframe




class Summarize:
    
    def __init__(self, tokenizer, summarize_model):
        self.tokenizer = tokenizer
        self.summarize_model = summarize_model


    def summarize(self, text):
        
        text_encoding = self.tokenizer.encode_plus(
            text,
            max_length = 512,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = 'pt'
            )
        
        generated_ids = self.summarize_model.model.generate(
            input_ids = text_encoding.input_ids,
            attention_mask = text_encoding.attention_mask,
            max_length = 192,
            num_beams = 8,
            repetition_penalty = 2.5,
            length_penalty = 2.0,
            early_stopping = True
            )
        
        predicted_text = [
            self.tokenizer.decode(generation_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for generation_id in generated_ids
            ]
        
        return ''.join(predicted_text)
