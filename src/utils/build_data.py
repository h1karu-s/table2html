import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import numpy as np


def build_loader(config, dataset_train, dataset_val=None, dataset_test=None, collate_fn=None):
    data_loader_train = None
    data_loader_val = None
    data_loader_test = None
    
    if config.use_ddp == True:
        print("DDP dataloader")
        num_tasks = dist.get_world_size()
        rank = dist.get_rank()
        
        if dataset_train is not None:     
            sampler_train = torch.utils.data.DistributedSampler(
              dataset_train, num_replicas=num_tasks, rank=rank, shuffle=True,
            )
            data_loader_train = torch.utils.data.DataLoader(
              dataset_train,
              sampler=sampler_train,
              num_workers=config.num_workers,
              batch_size=config.train_batch_size,
              collate_fn=collate_fn,
              pin_memory=True,
              drop_last=True,
            )
        
        if dataset_val is not None:
            sampler_val = torch.utils.data.DistributedSampler(
              dataset_val, shuffle=False,
            )
            data_loader_val = torch.utils.data.DataLoader(
              dataset_val,
              sampler=sampler_val,
              num_workers=config.num_workers,
              batch_size=config.val_batch_size,
              shuffle=False,
              collate_fn=collate_fn,
              pin_memory=True,
            )

        if dataset_test is not None:
            data_loader_test = torch.utils.data.DataLoader(
              dataset_test,
              num_workers=config.num_workers,
              batch_size=config.val_batch_size,
              shuffle=False,
              collate_fn=collate_fn,
              pin_memory=True,
            )
    else:
        print("defult datalaoder")
        if dataset_train is not None:
            data_loader_train = torch.utils.data.DataLoader(
              dataset_train,
              num_workers=config.num_workers,
              batch_size=config.train_batch_size,
              collate_fn=collate_fn,
              pin_memory=True,
              drop_last=True,
            )
        
        if dataset_val is not None:
            data_loader_val = torch.utils.data.DataLoader(
              dataset_val,
              num_workers=config.num_workers,
              batch_size=config.val_batch_size,
              pin_memory=True,
              collate_fn=collate_fn,
            )
            
        if dataset_test is not None:
            data_loader_test = torch.utils.data.DataLoader(
              dataset_test,
              num_workers=config.num_workers,
              batch_size=config.val_batch_size,
              pin_memory=True,
              collate_fn=collate_fn,
            )

    return data_loader_train, data_loader_val, data_loader_test
  

class CreateDataset(Dataset):
    def __init__(self, data, vocab, max_len=512):
        # print(max_len)
        self.data = data
        self.max_len = max_len
        self.cls = vocab.index("<s>")
        self.sep = vocab.index("</s>")
        self.pad = vocab.index("<pad>")
        # self.cls = "<s>"
        # self.sep = "</s>"
        # self.pad = "<pad>"
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        token_len = len(self.data[idx]["input_ids"])
        input_ids = self.data[idx]["input_ids"]
        bbox = self.data[idx]["bbox"]
        ###layoutLMv3 inputs: 0-1000
        bbox = (bbox * 1000).astype(np.int64)
        bbox = bbox.tolist()
        # print("token_len: " ,token_len)
        if token_len + 2 == self.max_len:
            # print(token_len)
            input_ids = [self.cls] + input_ids + [self.sep]
            bbox = [[0, 0, 0, 0]] + bbox + [[0, 0, 0, 0]]
            attention_mask = [1] * self.max_len
        else:
            input_ids = [self.cls] + input_ids + [self.sep] + [self.pad] * (self.max_len - token_len - 2)
            bbox = [[0, 0, 0, 0]] + bbox + [[0, 0, 0, 0]] + [[0, 0, 0, 0]] * (self.max_len - token_len - 2)
            attention_mask = [1] * (token_len + 2) + [0] * (self.max_len - token_len - 2)
        
        return_dict = {}
        for key in self.data[idx]:
            if key == "input_ids":
                return_dict[key] = input_ids
            elif key == "bbox":
                return_dict[key] = bbox
            else:
                return_dict[key] = self.data[idx][key]
        return_dict["attention_mask"] = attention_mask
        
        return return_dict