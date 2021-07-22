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