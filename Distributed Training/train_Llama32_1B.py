# import random
from utils import (
    download_from_s3, 
    upload_to_s3, 
    plot_multiple_results, 
    check_trainable_layers, 
    list_model_layers, 
    count_trainable_layers, 
    count_parameters, 
    unfreeze_BERT_layers, 
    unfreeze_all_Llama_layers, 
    plot_results, 
    rescale_matrix_layer_by_layer, 
    get_cuda_summary, 
    print_model_layers, 
    train_model, 
    evaluate_model, 
    plot_textDataset_distribution, 
    save_model, 
    save_model_to_s3, 
    CustomDataset
)
import argparse
import matplotlib.pyplot as plt
# from tqdm import tqdm
# import time
# import re
import os
import seaborn as sns
from functools import partial
import boto3

# import numpy as np
import torch
# import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
# from utils import *

from transformers import  AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, RandomSampler
from datasets import load_dataset
from torch.utils.data import Subset, Dataset
from torch.nn.utils.rnn import pad_sequence
# from torch.cuda.amp import GradScaler, autocast
# from torch.amp import GradScaler, autocast

import warnings
import logging
import sys
# import json
from dotenv import load_dotenv 

import random
from torch.utils.data import random_split

###################################################################
###################################################################
# load_dotenv()
# HF_TOKEN = os.getenv('TF_TOKEN')
HF_TOKEN = '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'

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
logging.getLogger("transformers").setLevel(logging.ERROR)
plt.style.use('dark_background')

###################################################################
## Added:
# import smdistributed.dataparallel.torch.distributed as dist
# from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



from torch.utils.data.distributed import DistributedSampler


# dist.init_process_group()
dist.init_process_group(backend='nccl')  # You can use 'nccl' for GPU or 'gloo' for CPU/multi-node

# Here's what it does:
# Initialization: It sets up the communication backend for distributed training. This function needs to be called before creating any distributed objects, such as distributed data loaders or distributed models.
# Process Group: It creates a group of processes that will communicate with each other during training. This group defines how processes will communicate, whether it's through shared memory, TCP sockets, or other communication methods.
# Initialization Parameters: The dist.init_process_group() function can take parameters to specify the backend (like 'gloo', 'nccl', 'mpi'), the rank of the current process, the total number of processes, and other configuration options depending on the backend being used.
# In summary, dist.init_process_group() is a crucial step in setting up distributed training in PyTorch, enabling communication and coordination between processes running on different devices or nodes in a distributed environment.

rank = dist.get_rank()
world_size = dist.get_world_size()
###################################################################
###################################################################




#################################################################################

###################################################################
###################################################################
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch  Example')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',help='input batch size for training (default: 1)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0e-5, metavar='LR',help='learning rate (default: 1.0e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--save_model', action='store_true', default=False,help='For Saving the current Model')
    parser.add_argument('--verbose', action='store_true', default=False,help='For displaying SMDataParallel-specific logs')
    parser.add_argument('--data_path', type=str, default='/tmp/data', help='Path for downloading the  dataset')
    parser.add_argument('--data_type', type=str, default=torch.bfloat16, help='Model Data Type')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of Samples to keep from dataset')
    parser.add_argument('--max_context_length', type=int, help='LLM context length')
    parser.add_argument('--saving_dir', type=str, default=os.environ['SM_saving_dir'], help='SM Saving Dir')
    
    parser.add_argument('--save_to_s3', type=bool, default=True, help='Whether to save the model to S3')
    parser.add_argument('--s3_path', type=str, help='Path to S3 to save the model')
    

    args = parser.parse_args()

    args.world_size = dist.get_world_size()
    args.rank =  dist.get_rank()
    
    # args.local_rank =  dist.get_local_rank()
    args.local_rank  = int(os.environ['LOCAL_RANK'])
    
    args.batch_size //= args.world_size  # 8
    args.batch_size = max(args.batch_size, 1)
    
    if args.verbose:
        print(f"Rank: {rank}, World Size: {args.world_size}, Local Rank: {args.local_rank}")  
    if not torch.cuda.is_available():
        raise Exception("Must run SMDataParallel on CUDA-capable devices")
    torch.manual_seed(args.seed)
    
        # Load the model
    model_name = "meta-llama/Llama-3.2-1B"
    llama32_1B_tokenizer = AutoTokenizer.from_pretrained(model_name,token=HF_TOKEN)
    llama32_1B_tokenizer.pad_token = llama32_1B_tokenizer.eos_token
    llama32_1B_tokenizer.padding_side = 'right'
    print("Tokenizer Loaded.")
    
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        input_ids_padded = pad_sequence(input_ids,batch_first=True, 
                                        padding_value=llama32_1B_tokenizer.pad_token_id)
        attention_masks_padded = pad_sequence(attention_masks,batch_first=True, padding_value=0)
        return {'input_ids': input_ids_padded, 'attention_mask': attention_masks_padded }
    
    dataset_name = "EleutherAI/pile"
    initial_dataset = load_dataset(dataset_name, 'enron_emails')    
    
    # Select a small subset of samples for all ranks
    if args.num_samples==-1:
        subset_mini = initial_dataset['train']
    else:
        subset_mini = initial_dataset['train'].select(range(args.num_samples)) 
         
    train_size = int(0.8 * len(subset_mini))
    val_size = len(subset_mini) - train_size
    train_subset, val_subset = random_split(subset_mini, [train_size, val_size])          
    
    train_dataset = CustomDataset(
        train_subset['text'], 
        tokenizer=llama32_1B_tokenizer, 
        max_length=args.max_context_length
    )
    
    # Distributed Sampler ensures each process gets a different part of the data
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=args.world_size, 
        rank=args.rank
    )    
    
    # Create the DataLoader using the sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  
        sampler=train_sampler,
        collate_fn=collate_fn ,
        num_workers=4,   
        pin_memory=True
    )
    # **Only on Rank 0 (Primary GPU): Create Validation Loader**
    if rank == 0:
        val_dataset = CustomDataset(
           val_subset['text'],  # Adjust this based on actual dataset
            tokenizer=llama32_1B_tokenizer,
            max_length=args.max_context_length
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            collate_fn=collate_fn ,
            num_workers=8,  # Adjust based on environment
            pin_memory=True
        )
    else:
        val_loader = None  # Other ranks don't need the validation loader

    
    DEVICE = torch.device(f"cuda:{args.local_rank}")
    llama32_1B_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=args.data_type, 
        token=HF_TOKEN
    ).to(DEVICE)
    
    llama32_1B_model = DDP(llama32_1B_model, device_ids=[args.local_rank], output_device=args.local_rank)
    optimizer=AdamW(llama32_1B_model.parameters(),lr=args.lr)
    
    for param in llama32_1B_model.parameters():
        if param.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Training loop
    llama32_1B_losses = train_model(
        llama32_1B_model, 
        optimizer=optimizer, 
        device=DEVICE,
        epochs=args.epochs,
        train_loader=train_loader, 
        validation_loader=val_loader if rank == 0 else None,  # Only use val_loader for rank 0
        patience=2,  # Early stopping patience
        train_verbose=True, 
        eval_verbose=True,
        with_autocast=False,  # Enable mixed precision if needed
        with_early_stopping=False,  # Enable early stopping
        with_GTL=False,  # Set to True if you want to use guided training
        guidance_matrix=None,  # Pass the guidance matrix if needed
        with_lr_sceduler=False,  # Enable learning rate scheduling
        time_to_sleep=5  # Add delay between epochs (optional)
    )
        
    if rank == 0:  # Only save the model from rank 0
        plot_results(llama32_1B_losses)  
        print("Saving the model...")
        save_model(llama32_1B_model, args.saving_dir)
        
        if args.save_to_s3:
            print("Saving the model to S3 Bucket...")
            save_model_to_s3(os.path.join(args.saving_dir, 'model.pth'), args.s3_path)
        
if __name__ == "__main__":
    main()