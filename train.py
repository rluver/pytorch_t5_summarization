# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:19:04 2021

@author: MJH
"""

from auxiliary import load_data
from model import Summarizer

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast as T5Tokenizer




class NewsSummaryDataset(Dataset):
    
    def __init__(
            self, 
            data: pd.DataFrame, 
            tokenizer: T5Tokenizer, 
            text_max_token_length: int = 512, 
            summary_max_token_length: int = 192
            ):
        
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_length = text_max_token_length
        self.summary_max_token_length = summary_max_token_length
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index: int):
        
        data_row = self.data.iloc[index]

        encoded_article = tokenizer(
            ' '.join(data_row.article_original),
            max_length = self.text_max_token_length, 
            padding = 'max_length', 
            truncation = True, 
            return_attention_mask = True, 
            add_special_tokens = True, 
            return_tensors = 'pt'
            )
        
        
        encoded_summarized_article = tokenizer(
            data_row.abstractive,
            max_length = self.summary_max_token_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = 'pt'
            )
        
        
        labels = encoded_summarized_article['input_ids']
        labels[labels == 0] = -100
        
        
        return dict(
            original_article = ' '.join(data_row['article_original']),
            summary = data_row['abstractive'],
            text_input_ids = encoded_article['input_ids'].flatten(),
            text_attention_mask = encoded_article['attention_mask'].flatten(),
            labels = labels.flatten(),
            labels_attention_mask = encoded_summarized_article['attention_mask'].flatten()
            )
    
    
    
    
class NewsSummaryDataModule(pl.LightningDataModule):
    
    def __init__(            
        self,
        train_dataframe: pd.DataFrame,
        test_dataframe: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        text_max_token_length: int = 512,
        summary_max_token_length: int = 192
    ):
    
        super().__init__()
        
        self.train_dataframe = train_dataframe
        self.test_dataframe = test_dataframe
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_length = text_max_token_length,
        self.summary_max_token_length = summary_max_token_length
        
        self.setup()
    
    
    def setup(self, stage = None):
        self.train_dataset = NewsSummaryDataset(
            self.train_dataframe,
            self.tokenizer,
            self.text_max_token_length,
            self.summary_max_token_length
            )
        
        self.test_dataset = NewsSummaryDataset(
            self.test_dataframe,
            self.tokenizer,
            self.text_max_token_length,
            self.summary_max_token_length
            )
        
        
    def train_dataloader(self):        
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True
            )
    
    
    def val_dataloader(self):        
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )
    
    
    def test_dataloader(self):        
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )
        
    

def main():
        
    EPOCHS = 10
    BATCH_SIZE = 8
    
    train = load_data('문서요약 텍스트/1.Training/신문기사/train.jsonl')
    valid = load_data('문서요약 텍스트/2.Valid/신문기사/dev.jsonl')
    
    tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
    
    
    data_module = NewsSummaryDataModule(train, valid, tokenizer, batch_size = BATCH_SIZE)    
    model = Summarizer()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = 'checkpoints',
        filename = 'best-checkpoint',
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'
        )

    logger = TensorBoardLogger('lightning_logs', name = 'news_summary')
  
    trainer = pl.Trainer(
        logger = logger,
        checkpoint_callback = checkpoint_callback,
        max_epochs = EPOCHS,
        gpus = 2,
        accelerator = 'dp',
        progress_bar_refresh_rate = 1
        )
    
    trainer.fit(model, data_module)
    
    
    
    
if __name__ == '__main__':
    main()
