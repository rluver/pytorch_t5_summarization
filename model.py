# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:16:30 2021

@author: MJH
"""


import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AdamW, MT5ForConditionalGeneration




class Summarizer(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base', return_dict = True, max_length = 512)
        
        
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels = None):
        
        output = self.model(
            input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = decoder_attention_mask
            )
       
        return output.loss, output.logits
    
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
            )
        
        self.log('train_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
            )
        
        self.log('val_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
            )
        
        self.log('test_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = 1e-4)