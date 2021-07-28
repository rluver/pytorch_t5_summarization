from model import Summarizer
from auxiliary import Summarize
from transformers import T5TokenizerFast as T5Tokenizer

import torch




if  __name__ == '__main__':
    
    tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
    
    summarize_model = Summarizer.load_from_checkpoint(r'checkpoints\best-checkpoint.ckpt')
    summarize_model.freeze()
    
    summarize = Summarize(tokenizer, summarize_model)
    
    text = ''
    summarize.summarize(text)    