from model import Summarizer
from auxiliary import load_data
from transformers import T5TokenizerFast as T5Tokenizer

import torch



def summarize(text):
    
    text_encoding = tokenizer.encode_plus(
        text,
        max_length = 512,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = 'pt'
        )
    
    generated_ids = summarize_model.model.generate(
        input_ids = text_encoding.input_ids,
        attention_mask = text_encoding.attention_mask,
        max_length = 196,
        num_beams = 8,
        repetition_penalty = 2.5,
        length_penalty = 2.0,
        early_stopping = True
        )
    
    predicted_text = [
        tokenizer.decode(generation_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for generation_id in generated_ids
        ]
    
    return ''.join(predicted_text)




if  __name__ == '__main__':
    
    tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
    
    summarize_model = Summarizer.load_from_checkpoint(r'checkpoints\best-checkpoint.ckpt')
    summarize_model.freeze()
    
    print('input text: ')
    summarize(input())