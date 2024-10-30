import matplotlib.pyplot as plt
from tqdm import tqdm
import time
# import re
import os
import seaborn as sns
# from functools import partial
import boto3
# from  utils import *

#################################################
##################################################
# import numpy as np
import torch
# import torch.nn as nn
# from torch.optim import AdamW
# import torch.nn.functional as F
#################################################
##################################################
from transformers import  AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
# from torch.utils.data import DataLoader, Dataset, RandomSampler
# from datasets import load_dataset
# from torch.utils.data import Subset, Dataset
# from torch.nn.utils.rnn import pad_sequence

from torch.cuda.amp import GradScaler, autocast
# from torch.amp import GradScaler, autocast
#################################################
##################################################
import warnings
import logging
import sys
# import json
# from dotenv import load_dotenv

sys.path.append('./src') #link to GTL libraries
import src.guided_transfer_learning as gtl


# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda:0')
#     # print(f" DEVICE: {DEVICE}. GPU: {torch.cuda.get_device_name(0)} is available.")
# else:
#     DEVICE = 'cpu'
#     # print(f" DEVICE: {DEVICE}. No GPU available. Training will run on CPU.")
########################################################################################################################
def download_from_s3(bucket_name, s3_file_path, local_file_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, s3_file_path, local_file_path)
    print(f'File s3://{bucket_name}/{s3_file_path} downloaded to {local_file_path}')

def upload_to_s3(local_file_path, bucket_name, s3_file_path):
    s3 = boto3.client('s3')
    s3.upload_file(local_file_path, bucket_name, s3_file_path)
    # print(f'File {local_file_path} uploaded to s3://{bucket_name}/{s3_file_path}')
########################################################################################################################


def plot_multiple_results(results_list, title="Train and Validation Results (Accuracy and Loss)",legend_fontsize=9, nb_epochs_toplot=None):
    fig, axs = plt.subplots(1, 2, figsize=(40, 10))
    styles = [{'linestyle': (0,(1,1)), 'marker': 'o', 'color': '#2ca02c'},{'linestyle': 'dashdot', 'marker': 'x', 'color': '#1f77b4'},
        {'linestyle': (0,(5,1)), 'marker': 's', 'color': '#d62728'},{'linestyle': (5,(10,3)), 'marker': 'd', 'color': '#ff7f0e'} ]
    for i, results in enumerate(results_list):
        train_loss, train_acc = results['train_results']
        val_loss, val_acc = results['validation_results']
        if nb_epochs_toplot is None:
            max_epochs = max(max(len(train_acc) for results in results_list),max(len(val_acc) for results in results_list),max(len(train_loss) for results in results_list),max(len(val_loss) for results in results_list))
        else:
            max_epochs = nb_epochs_toplot
        train_loss , train_acc , val_loss, val_acc = train_loss[:max_epochs], train_acc[:max_epochs], val_loss[:max_epochs+1], val_acc[:max_epochs+1],
        style = styles[i % len(styles)]
        # Plot the training and validation accuracy
        axs[0].plot(range(1,len(train_acc) + 1), train_acc, marker=style['marker'], markersize=3, color=style['color'], label=f'Training Accuracy Selection {i+1}')
        axs[0].plot(val_acc, linestyle=style['linestyle'], marker=style['marker'],  markersize=3, color=style['color'], label=f'Validation Accuracy Selection {i+1}')
        # Plot the training and validation loss
        axs[1].plot(range(1,len(train_loss) + 1), train_loss, marker=style['marker'], markersize=3,color=style['color'], label=f'Training Loss Selection {i+1}')
        axs[1].plot(val_loss, linestyle=style['linestyle'], marker=style['marker'], markersize=3,color=style['color'], label=f'Validation Loss Selection {i+1}')
    for ax in axs:
        ax.set_xticks(range(0, max_epochs + 1, 2))
    # Set labels and titles
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].legend(fontsize=legend_fontsize)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training and Validation Loss')
    axs[1].legend(fontsize=legend_fontsize)
    fig.suptitle(title)
    plt.show()

    
########################################################################################################################
# Iterate over all parameters in the model
def check_trainable_layers(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is || TRAINABLE || ")
        else:
            print(f"{name} is || NOT TRAINABLE || ")
# Iterate over all parameters in the model

def list_model_layers(model):
    for name, param in model.named_parameters():
        print(f"{name}")

def count_trainable_layers(model):
    trainable_layers = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_layers += 1
    return trainable_layers

def count_parameters(model):
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

def unfreeze_BERT_layers(bert, nb_layer_to_unfreeze):
    if (0 < nb_layer_to_unfreeze <= len(bert.encoder.layer)):
        for layer in bert.encoder.layer[-nb_layer_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in bert.encoder.layer[:len(bert.encoder.layer) - nb_layer_to_unfreeze]:
            for param in layer.parameters():
                param.requires_grad = False
    else:
        print(" Number of unfreeze layers is larger than number of layers or less than 1.")

def unfreeze_all_Llama_layers(model ):
    for param in model.parameters():
        param.requires_grad = True

# Iterate over all parameters in the model
def count_trainable_layers(model):
    trainable_layers = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_layers += 1
    return trainable_layers

def count_parameters(model):
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

def plot_results(results, title="Train and Validation Loss"):
    train_loss = results['train_results'][0]
    val_loss = results['validation_results'][0]
    fig, axs = plt.subplots(figsize=(8, 3))
    # Plot the training and validation loss
    axs.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    axs.plot(val_loss, label='Validation Loss')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.set_title(title)
    axs.legend()
    plt.show()

def rescale_matrix_layer_by_layer(matrix):
    for layer in matrix:
        min_layer = torch.amin(matrix[layer])
        max_layer = torch.amax(matrix[layer])
        scale =  1/(max_layer - min_layer)
        if scale is not torch.inf:
            matrix[layer] = scale *  matrix[layer]

def get_cuda_summary():
    print(torch.cuda.memory_summary())

def print_model_layers(module, prefix=''):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(full_name)
        print_model_layers(child, full_name)

   
    
########################################################################################################################
# Define the training function
def train_model(model, optimizer, epochs, train_loader, validation_loader, device, with_early_stopping=True, train_verbose=True, eval_verbose=True, with_GTL=False, guidance_matrix=None, with_autocast=False,patience=3, with_lr_sceduler=False, time_to_sleep=5, empty_cache=False):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    skip = 2000 if epochs > 10000 else 300 if epochs > 1000 else 30 if epochs > 100 else 5 if epochs > 10 else 1
    model.train()
    train_loss_vals = [];  val_loss_vals = [];

    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps) if with_lr_sceduler  else None
    
    grad_scaler = GradScaler('cuda') if with_autocast else None

    print(f"# Epoch: 0  (Before start training) ")
    val_loss = evaluate_model(model, validation_loader, device, verbose=eval_verbose) if validation_loader is not None else 0
    val_loss_vals.append(val_loss) if validation_loader is not None else None
    print("================================================================================================")
    for epoch in range(epochs):
        epoch_loss = 0.0; epoch_loss_sum = 0.0;
        if train_verbose and ((epoch + 1) % skip == 0 or epoch == 0 or epoch == epochs - 1):
            print(f"# Epoch: {epoch+1} (Learning rate: {optimizer.param_groups[0]['lr']}) ", end='')

        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Use autocast for mixed precision training
            if with_autocast:
                with autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs['loss']
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs['loss']

            model.zero_grad()

            if with_autocast:
                grad_scaler.scale(loss).backward() # Scales the loss and computes gradients
                if with_GTL and guidance_matrix is None:
                    print('Guidance matrix is not provided')
                    return None
                if with_GTL and guidance_matrix is not None:
                    gtl.apply_guidance(model, guidance_matrix)

                grad_scaler.step(optimizer) # Applies the scaled gradients to the optimizer.
                grad_scaler.update()        # Adjusts the scaling factor for the next iteration.
            else:
                loss.backward()             # Calculate the gradients for the loss w.r.t the parameters
                if with_GTL and guidance_matrix is None:
                    print('Guidance matrix is not provided')
                    return None
                if with_GTL and guidance_matrix is not None:
                    gtl.apply_guidance(model, guidance_matrix)
                optimizer.step()            # Update the values of parameters  with the gradients

            epoch_loss_sum += loss.item()

            if with_lr_sceduler:
                lr_scheduler.step()

            if empty_cache:
                torch.cuda.empty_cache()

        epoch_loss = epoch_loss_sum / len(train_loader)
        train_loss_vals.append(epoch_loss)
        val_loss = evaluate_model(model, validation_loader, device, verbose=False)
        val_loss_vals.append(val_loss)

        if empty_cache:
            torch.cuda.empty_cache()

        if with_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == patience:
                    print(f'Early stopping after {epochs_without_improvement+1} epochs without improvement.')
                    print(f'- Training loss: {epoch_loss:.2f} || Validation loss: {val_loss:.2f} \n ')
                    break
        if validation_loader is not None:
            if train_verbose and ((epoch + 1) % skip == 0 or epoch == 0 or epoch == epochs - 1):
                print(f' Training loss: {epoch_loss:.2f} || Validation loss: {val_loss:.2f}')
            time.sleep(time_to_sleep)
    return {'train_results': [train_loss_vals], 'validation_results': [val_loss_vals]}

# Define the evaluation function


def evaluate_model(model, loader, device, verbose=True, dataset_category="Validation", with_autocast=False):
    model.eval()
    total_loss = 0
    if verbose:
        print(f"# Evaluating on {dataset_category} Dateset: ", end='')
    with torch.no_grad():
        for batch in tqdm(loader):
        # for batch in loader:
            input_ids = batch['input_ids'].to(device=device)
            attention_mask = batch['attention_mask'].to(device=device)  # mask for attention

            if with_autocast:
                with autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs['loss']
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs['loss']
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    if verbose:
        print(f'# {dataset_category} loss: {avg_loss:.2f} ')
    return avg_loss

# time.sleep(1000)
########################################################################################################################

def plot_textDataset_distribution(dataset,title,xlabel_title, ylabel_title,xlim,kde=False):
    plt.figure(figsize=(10, 3))
    plt.xlim(0, xlim)
    sns.histplot(dataset['Context_length'], bins='auto', kde=kde)
    plt.title(title)
    plt.xlabel(xlabel_title)
    plt.ylabel(ylabel_title)
    plt.show()

# We tokenize the item when getting it from the dataset

########################################################################################################################
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text,
                                max_length=self.max_length,
                                truncation=True,
                                padding=False,
                                return_tensors="pt")
        return {"input_ids": tokens['input_ids'].squeeze(0), "attention_mask": tokens['attention_mask'].squeeze(0)}

    
class CustomDataset_tokenize_onLoad(torch.utils.data.Dataset):
    # Pre-tokenization (faster, more RAM usage)
    def __init__(self, texts, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_texts = [self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt") for text in texts]

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokenized_text = self.tokenized_texts[idx]
        # print("tokenized_text length:", tokenized_text['input_ids'].size(1))  # size(1) gets the sequence length dimension
        return {
            "input_ids": tokenized_text['input_ids'].squeeze(0),
            "attention_mask": tokenized_text['attention_mask'].squeeze(0)
            }
########################################################################################################################