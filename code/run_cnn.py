# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import gc
import logging
import os
import random
import torch


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys

sys.path.append('../DataAugVuln/')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json


from tqdm import tqdm
import multiprocessing


from models.model import Model

from AugVuln import trainCL,train,test

from data_process.ProcessedData import ProcessedData
from models.CVAE.CVAE_model import *
from CGE import CVAESynthesisData,Pipeline
from data_process.textdata import read_answers,read_predictions,calculate_scores,TextDataset
cpu_cont = multiprocessing.cpu_count()
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):          


# Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training and Evaluation
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.local_rank == 0:
            torch.distributed.barrier()

        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        
        if args.VCEorCGE:
            train_dataset_CL,sample_weight = trainCL(args, train_dataset, model, tokenizer)
            # train_dataset_CL = TextDataset(tokenizer, args, args.train_process_data_file)
            train_dataset_CL2 = train_dataset_CL
            
            pl = Pipeline(train_dataset_CL)
            train_dataset_CL = pl.run(args)
            train_dataset= train_dataset_CL.update(train_dataset_CL2)
            
            model.to(args.device)
            train(args, train_dataset, model,tokenizer,sample_weight)
        
        else:
            train_dataset_cvae = train_dataset
            train_dataset_cvae2 = train_dataset_cvae
            pl = Pipeline(train_dataset_cvae)
            train_dataset_cvae = pl.run(args)
            train_dataset = train_dataset_cvae.update(train_dataset_cvae2)
            train_dataset_CL,sample_weight = trainCL(args, train_dataset, model, tokenizer)

            model.to(args.device)
            train(args, train_dataset_CL, model,tokenizer,sample_weight)
                

    results = {}
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer)

    return results

def setup_args():
    args = argparse.Namespace()
    
    args.do_train =  False
    args.do_test = True
    
    # path
    args.language = "c"
    args.train_data_file = '../dataset/cdata/CL/basedata/train_cdata.jsonl'
    args.test_data_file = '../dataset/cdata/CL/basedata/test_javadata.jsonl'
    
    #output
    args.train_process_data_file = "../results/cnn_CVAE+CL_epo2+8_kf2_basedata/train_process_data_file.jsonl"
    args.output_dir = "../results/cnn_CVAE+CL_epo2+8_kf2_basedata"
    args.output_dir_before = "../results/cnn_CVAE+CL_epo2+8_kf2_basedata/before"
    args.pkl_file = '../dataset/cdata/CL/basedata/short_3path_cdata_nobalance.pkl'

    # models
    args.model_type = "roberta"
    args.tokenizer_name = '../models/codebert'
    args.config_name = "../models/codebert"
    args.model_name_or_path = '../models/codebert'

    # Training and evaluation configuration
    args.threshold = 0.5
    args.train_batch_size = 24
    args.eval_batch_size = 64
    args.epoch = 2
    args.cl_epoch = 8
    args.splits_num = 2# 折数过低会导致结果异常
    
    #cvae
    args.encoder_layer_sizes = 0
    args.decoder_layer_sizes = 0
    args.latent_size = 5
    args.conditional = True
    args.lr = 0.005
    
    #setting
    args.VCEorCGE = False
    # Others
    args.d_size = 128
    args.block_size = 400
    args.mlm = False  # Default for boolean flags is typically False unless specified with action='store_true'
    args.mlm_probability = 0.15
    args.cache_dir = ""
    args.evaluate_during_training = True
    args.do_lower_case = False
    args.gradient_accumulation_steps = 1
    args.learning_rate = 2e-5
    args.weight_decay = 0.0
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 1.0
    args.num_train_epochs = 1.0
    args.max_steps = -1
    args.warmup_steps = 0
    args.logging_steps = 50
    args.save_steps = 50
    args.save_total_limit = None
    args.eval_all_checkpoints = False
    args.no_cuda = False
    args.overwrite_output_dir = False
    args.overwrite_cache = False
    args.seed = 123456
    args.fp16 = False
    args.fp16_opt_level = 'O1'
    args.local_rank = -1
    args.server_ip = ''
    args.server_port = ''
    args.cnn_size = 128
    args.filter_size = 3
    
    return args
    
if __name__ == "__main__":
    args = setup_args()
    main(args)


