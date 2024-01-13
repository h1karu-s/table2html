### vearsion_2: update embedding size of LayoutLMv3 (patchsize: 32, 48 etc.) for high resolutiron image

import argparse
import json
from PIL import Image, ImageDraw
import os
import math
import warnings
import random
import gc

import torch
from torch.optim import AdamW
import torch.nn.functional as F
import pickle5 as pickle
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoConfig, BartTokenizer
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import copy
import numpy as np
from sklearn.metrics import classification_report
import datetime
from transformers import get_constant_schedule_with_warmup
import time
import torch.cuda.amp as amp
import torch.distributed as dist
from utils.logger import create_logger
from utils.utils import *
from utils.build_data import build_loader, CreateDataset
from data.preprocessing_layoutlmv3_en_large_truncation import preprocessing_data
from model.layoutlmv3_bart import DonutConfig, DonutModel
# from model.donut_layoutLMv3_2 import DonutConfig, DonutModel
warnings.simplefilter('ignore')


def save_prediction(pred, label, save_path):
    with open(save_path, "w") as f:
        for p, l in zip(pred, label):
            f.write(f"HTML: {str(l)}"+"\n")
            f.write(f"PREDICTION: {str(p)}"+"\n")
            f.write(f"----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
 

### load data and preprocessing shape of [{"input_ids": [], "bbox": [], "label": [], ..}, {}, ...]
def create_dataset(args, model):
    global decoder_tokenizer
    dataset_train = None
    dataset_test = None
    
    if args.preprocessing_path is not None:
        logger.info(f"DATA: get preprocessing_data")
        logger.info(f"DATA: loading ... {args.preprocessing_path} ")
        t = time.time()
        data = load_file(args.preprocessing_path, logger)
        logger.info(f"loading time: {datetime.timedelta(seconds=int(time.time()-t))})")
        if args.datasize is not None:
            data = data[:args.datasize]
    else:
        logger.info(f"preprocesing Data from scratch")
        t = time.time()
        data, data_val,  data_test = preprocessing_data(args, model, logger)
        logger.info(f"loading time: {datetime.timedelta(seconds=int(time.time()-t))}")
    logger.info(f"DATA: length is {len(data)}, test_data is {len(data_test)}")
    
    # for d in data:
    #     if d["encoding_inputs"]["input_ids"].shape[0] > 200:
    #         print("input_ids", d["encoding_inputs"]["input_ids"].shape)
    #     elif  d["encoding_inputs"]["attention_mask"].shape[0] > 200:
    #         print("atten", d["encoding_inputs"]["attention_mask"].shape)
    if data_val is None:
    ###train と validに分ける
        n_train = math.floor(len(data) * args.ratio_train)
        dataset_train = data[:n_train]
        dataset_val = data[n_train:]
    else:
        dataset_train = data
        dataset_val = data_val
        
    dataset_test = data_test
    logger.info(f"train_dataset length:{len(dataset_train)}, valid dataset length: {len(dataset_val)}, test_dataset length: {len(dataset_test)}")
    
    return dataset_train, dataset_val, dataset_test

class collate_fn:
    def __init__(self, encoder_processor):
        self.encoder_processor = encoder_processor
  
    def __call__(self, batch):
        ####for high resolution pixels , preprocessing in collate_fn
        encoder_inputs = {}
        batch_encoder = []
        for b in batch:
            image = Image.open(b["encoding_inputs"]["image_path"])
            encoder_encoding = self.encoder_processor(image, b["encoding_inputs"]["ocr_tokens"], bboxes=b["encoding_inputs"]["ocr_bboxes"])
            encoder_encoding["bbox"] = (encoder_encoding["bbox"]*1000).to(torch.int32)
            batch_encoder.append(encoder_encoding)
        for key in ["input_ids", "bbox", "attention_mask", "pixel_values"]:
            encoder_inputs[key] = torch.cat([b[key] for b in batch_encoder])
        
        decoder_inputs = {}
        for key in ["input_ids", "attention_mask"]: 
            decoder_inputs[key] = torch.cat([b["encoding_html"][key][:, :-1] for b in batch])
        labels = torch.cat([b["encoding_html"]["input_ids"][:, 1:] for b in batch])
        return encoder_inputs, decoder_inputs, labels

class collate_fn_test:
    def __init__(self, encoder_processor):
        self.encoder_processor = encoder_processor
  
    def __call__(self, batch):
        ####for high resolution pixels , preprocessing in collate_fn
        encoder_inputs = {}
        batch_encoder = []
        for b in batch:
            image = Image.open(b["encoding_inputs"]["image_path"])
            encoder_encoding = self.encoder_processor(image, b["encoding_inputs"]["ocr_tokens"], bboxes=b["encoding_inputs"]["ocr_bboxes"])
            encoder_encoding["bbox"] = (encoder_encoding["bbox"]*1000).to(torch.int32)
            batch_encoder.append(encoder_encoding)
        for key in ["input_ids", "bbox", "attention_mask", "pixel_values"]:
            encoder_inputs[key] = torch.cat([b[key] for b in batch_encoder])
        
        labels = torch.cat([b["encoding_html"]["input_ids"][:, 1:] for b in batch])
        image_paths = [b["image_path"] for b in batch]
        return encoder_inputs, labels, image_paths
      

def main(args):
    # "global variable"
    global decoder_tokenizer
    global best_model
    global best_val_score
    global id2label
    global store
    best_val_score = {"score": 1000000, "steps": 0, "epoch": 0}

    ##model
    donut_config = DonutConfig(
      encoder_max_length=args.encoder_max_length,
      decoder_max_length=args.decoder_max_length,
      encoder_layer=args.num_encoder_layer,
      encoder_name_or_path=args.encoder_name_or_path,
      decoder_name_or_path=args.decoder_name_or_path,
      add_vocab_list=add_vocab_list,
      )

    # donut_config = DonutConfig(encoder_max_length=200, encoder_name_or_path=args.encoder_name_or_path, decoder_name_or_path=args.decoder_name_or_path)
    if args.model_path is not None:
        logger.info(f"Model: loadding checkpoint of {args.mdoel_path}")
        model = DonutModel(donut_config)
    else:
        logger.info(f"Model: trianing from scratch")
        model = DonutModel(donut_config)
    
    ##count parameter
    num_parameter = count_prameters(model.encoder.model)
    logger.info(f"MODEL:encoder parameters is {num_parameter}(M)")
    # logger.info(f"MODEL_CONFIG: {model.config}")
    donut_config.to_json_file(f"{args.output}/model_config.json")
    model.decoder.model.config.to_json_file(f"{args.output}/model_decoder_config.json")
    model.encoder.model.config.to_json_file(f"{args.output}/model_encoder_config.json")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"device:{device}")
     
    dataset_train, dataset_val, dataset_test = create_dataset(args, model)
    train_dataloader, valid_dataloader, _ = build_loader(args, dataset_train, dataset_val, None, collate_fn=collate_fn(model.encoder.prepare_input))
    _, _, test_dataloader = build_loader(args, None, None, dataset_test, collate_fn=collate_fn_test(model.encoder.prepare_input))
    logger.info(f"train_dataloader length:{len(train_dataloader)}, valid dataloader length: {len(valid_dataloader)}")
    
    if args.use_ddp == False:
        device_ids = list(range(torch.cuda.device_count()))
        logger.info(f"use DP, device_ids={device_ids}")
        model = torch.nn.DataParallel(model, device_ids = device_ids)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
        
    else:
        logger.info("use DDP")
        model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    num_step_per_epoch = len(train_dataloader) // args.train_accum_iter
    total_steps = args.train_epochs * num_step_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps)
    ## chekpoint 
    
    ###
    # torch.backends.cudnn.benchmark = True
    logger.info("Start training")
    store = Store(args.store_names)
    start_time = time.time()
    for epoch in range(args.train_start_epochs, args.train_epochs):
        if args.use_ddp == True:
        #In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
            train_dataloader.sampler.set_epoch(epoch)
    
        train_one_epoch(args, model, None, train_dataloader, valid_dataloader, optimizer, epoch, lr_scheduler, scaler, device)
        
        if (epoch+1) % args.save_freq == 0:
            if args.use_ddp == True:
                if dist.get_rank() == 0:
                    save_checkpoint(args, epoch, model.module, optimizer, lr_scheduler, scaler, logger, store)
            else:
                save_checkpoint(args, epoch, model.module, optimizer, lr_scheduler, scaler, logger, store)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    os.makedirs(f"{args.output}/best_model", exist_ok=True)
    logger.info(f"BEST Model: {best_val_score}")
    torch.save(best_model.module.state_dict(), f"{args.output}/best_model/best_model.cpt")
    with open(f"{args.output}/best_model/best_val_score.txt", "w") as f:
        f.writelines(str(best_val_score))
        
    logger.info("TEST...")
    torch.cuda.empty_cache()
    test(args, test_dataloader, best_model.cuda())
            

### train code ####
def train_one_epoch(args, model, criterion, dataloader_train, dataloader_val, optimizer, epoch, lr_scheduler, scaler, device):
    model.train()
    global best_val_score
    global best_model
    global store
    
    optimizer.zero_grad()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    
    num_steps = len(dataloader_train)
    num_steped = (len(dataloader_train)) * epoch

    
    start = time.time()
    end = time.time()
    middle = time.time()
    for idx, (encoder_inputs, decoder_inputs, labels) in enumerate(dataloader_train):
        encoder_inputs = {k: v.cuda(non_blocking=True) for k, v in encoder_inputs.items()}
        decoder_inputs = {k: v.cuda(non_blocking=True)  for k, v in decoder_inputs.items()}
        labels = labels.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_amp):
            outputs = model(encoder_inputs, decoder_inputs["input_ids"], labels, attention_mask=decoder_inputs["attention_mask"])
            loss = outputs.loss
            # logger.info(f"LOSS: {loss}")
        loss = loss / args.train_accum_iter
        scaler.scale(loss).backward()
        
        
        if (idx + 1) % args.train_accum_iter == 0 or (idx + 1) == num_steps:
            if args.use_amp and args.train_clip_grad:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.train_clip_grad)
            else:
                if args.check_grad_norm:
                    ### check the grad_norm (because I only want to find the entire model’s gradient without cliping the gradients.)
                    # scaler.unscale_(optimizer)
                    grad_norm = get_grad_norm(model.parameters())
    
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            lr_scheduler.step()
        
        if args.use_ddp == True:
            torch.cuda.synchronize()
        
        ##save loss
        loss_meter.update(loss.item()*args.train_accum_iter, 1)
        if (args.check_grad_norm or args.use_amp) and (idx + 1) % args.train_accum_iter == 0 or (idx + 1) == num_steps:
            norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        #validation(change cord)
        if (((idx+1) % math.floor(num_steps*args.valid_ratio) == 0) and idx != 0) or ((idx+1) == num_steps):
            # torch.cuda.empty_cache()
            # gc.collect()
            valid_loss = validate(args, dataloader_val, model, loss_fn=None)
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            middle_time = time.time() - middle
            middle = time.time()
            
            if valid_loss < best_val_score["score"]:
                best_val_score["score"] = valid_loss
                best_val_score["epoch"] = epoch
                best_val_score["steps"] = idx + num_steped
                best_model = copy.deepcopy(model).to("cpu")
            
            store.update({
                "iter": idx + num_steped,
                "train_loss": loss_meter.avg,
                "val_loss": valid_loss,
            })
            logger.info(
                f'Train: (epoch/total epoch):[{epoch}/{args.train_epochs-1}](steps/total steps):[{idx+1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time(batch_time(avg_time)) {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'train_loss(batch_loss(avg_loss)) {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'valid_loss {valid_loss: 4f}\t'
                f'best model {best_val_score}\t'
                f'grad_norm(batch_norm(abg_norm)) {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'middel training time: {datetime.timedelta(seconds=int(middle_time))}'
                )

            loss_meter.reset()
            
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

          

def validate(args, data_loader, model, loss_fn):
    global decoder_tokenizer
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        for idx, (encoder_inputs, decoder_inputs, labels) in enumerate(data_loader):
            encoder_inputs = {k: v.cuda(non_blocking=True) for k, v in encoder_inputs.items()}
            decoder_inputs = {k: v.cuda(non_blocking=True)  for k, v in decoder_inputs.items()}
            labels = labels.cuda(non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_amp):
                outputs = model(encoder_inputs, decoder_inputs["input_ids"], labels, attention_mask=decoder_inputs["attention_mask"])
                loss = outputs.loss
            
            loss_meter.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()
    
    logger.info(f"Validation: Time: {batch_time.avg: 3f}")

    return  loss_meter.avg

def generation(args, data_loader, model):
    model.eval()
    with torch.no_grad():
        for idx, (encoder_inputs, decoder_inputs, _) in enumerate(data_loader):
            generate_text = model.module.inference(inputs=encoder_inputs)["predictions"]
            html = model.module.decoder.tokenizer.batch_decode(decoder_inputs["input_ids"])
            tmp = []
            for h in html:
                h = h.replace(model.module.decoder.tokenizer.eos_token, "").replace(model.module.decoder.tokenizer.pad_token, "")
                tmp.append(h)
            html = tmp
            break
    return  generate_text, html

def test(args, data_loader, model):
    pred = {}
    true = {}
    image_paths = []
    model.eval()
    with torch.no_grad():
        for idx, (encoder_inputs, labels, image_paths) in enumerate(data_loader):
            generate_text = model.module.inference(inputs=encoder_inputs)["predictions"]
            html = model.module.decoder.tokenizer.batch_decode(labels)
            tmp = []
            for h in html:
                h = h.replace(model.module.decoder.tokenizer.eos_token, "").replace(model.module.decoder.tokenizer.pad_token, "")
                tmp.append(h)
            html = tmp
            for i, path in enumerate(image_paths):
                pred[path] = generate_text[i]
                true[path] = html[i]
    with open(f"{args.output}/best_model/pred_test.json", "w") as f:
        json.dump(pred, f)
    with open(f"{args.output}/best_model/gt_test.json", "w") as f:
        json.dump(true, f)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ##高層化
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--use_ddp", action='store_true')
    parser.add_argument("--local_rank", type=int)
    #data
    parser.add_argument("--input_file_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--input_val_file", type=str)
    parser.add_argument("--input_test_file", type=str)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--preprocessing_path", type=str)
    parser.add_argument("--datasize", type=int)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--encoder_max_length", type=int)
    parser.add_argument("--decoder_max_length", type=int, required=True)
    
    #train
    parser.add_argument("--ratio_train", type=float,default=0.9)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--train_accum_iter", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=int, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--train_start_epochs", type=int, default=0)
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)
    parser.add_argument("--check_grad_norm", action="store_true")
    parser.add_argument("--valid_ratio", type=float,default=0.2)
    parser.add_argument("--train_clip_grad", type=float,default=1)
    #model
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--num_decoder_layer", type=int, default=6)
    parser.add_argument("--num_encoder_layer", type=int, default=12)
    parser.add_argument("--encoder_name_or_path", type=str)
    parser.add_argument("--decoder_name_or_path", type=str)
    parser.add_argument('--add_vocab_list', nargs='+',  help='adding new vocablary list') 
    ##other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument('--store_names', nargs='+', required=True, help='a list of string variables') 
    parser.add_argument("--save_freq", type=int, default=1)
    
    args = parser.parse_args()
    ## os setting
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
   
    if args.local_rank is not None:
        ### DDP: data.distribuiton.pararell 
        args.use_ddp = True
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = -1
            world_size = -1
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
        os.makedirs(args.output, exist_ok=True)
        logger = create_logger(output_dir=args.output, dist_rank=dist.get_rank(), name=f"{args.model_id}")
    else:
        ### DP: data pararell 
        os.makedirs(args.output, exist_ok=True)
        logger = create_logger(output_dir=args.output, dist_rank=0, name=f"{args.model_id}")
    # if dist.get_rank() == 0:
    #     with open(path, "w") as f:
    #         f.write(args.dump())
    logger.info(json.dumps(vars(args)))
    
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #fintabnet
    # add_vocab_list = ["<table>", "</table>", "<td>", "</td>", "<thead>", "</thead>", "<tr>", "<td", "</tr>", 
    #                 "rowspan=", "colspan=", "<sup>", "</sup>", "<b>", "</b>", "<i>", "</i>", "<sub>", "</sub>"]

    #pubtabnet
    add_vocab_list = ['<table>', '</table>', '<td>', '</td>', '<thead>', '</thead>', '<tr>', '<td', '</tr>', 'rowspan=', 'colspan=', '<b>', '<strike>', '</i>', '</b>', '</strike>', '</underline>', '</sup>', '<sub>', '<i>', '</sub>', '<underline>', '<overline>', '<sup>', '</overline>']

    main(args)
    
    
 
