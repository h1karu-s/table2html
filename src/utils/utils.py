import numpy as np
import os
import random
import torch
from torch._six import inf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import gc
import os
import pickle


def random_mask(seq_len):
    mask_len = int(seq_len*0.3)
    mask_bool = np.array([False]*seq_len)
    indexes = list(range(0, seq_len))
    random.shuffle(indexes)
    random.shuffle(indexes)
    mask_indexes = indexes[:mask_len]
    mask_indexes.sort()
    mask_bool[mask_indexes] = True
    return mask_bool, mask_indexes

def split_into_max_length(seq_ids, bboxes, max_length=512):
    max_length = max_length -2 
    subset_ids_l = []
    subset_bbox_l = []
    i = 0
    for i in range(0, len(seq_ids), max_length):
        if i == 0:
            continue
        subset_ids =  seq_ids[i-max_length:i]
        subset_bbox = bboxes[i-max_length: i]
        subset_ids_l.append(subset_ids)
        subset_bbox_l.append(subset_bbox)
    subset_ids = seq_ids[i:] 
    subset_bbox = bboxes[i:]
    subset_ids_l.append(subset_ids)
    subset_bbox_l.append(subset_bbox)
    return subset_ids_l, subset_bbox_l

###
def iou_np(a, b):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])
    
    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    # shape=(N,)のnumpy配列。Nは矩形の数
    b_area = (b[:,2] - b[:,0] + 1) \
             * (b[:,3] - b[:,1] + 1)
    
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h
    
    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (b_area)
    mask_indexes = iou > 0.5
    cand_words = np.where(mask_indexes == True)[0].tolist()
    return cand_words


# def patch_bbox(width, height, size=16):
#     if width % size != 0 or height % size != 0:
#         raise ValueError("invalid width or height for patch size!")
#     bbox = [(x, y, x+size, y+size) for x in range(0, width, size) for y in range(0, height, size)]
#     return bbox

def patch_bbox(width, height, size=16):
    if width % size != 0 or height % size != 0:
        raise ValueError("invalid width or height for patch size!")
    bbox = [(x, y, x+size, y+size) for y in range(0, height, size) for x in range(0, width, size)]
    return bbox


def plot_graph(config, epoch, store):
    iter = store.get("iter")
    train_losses = store.get("train_loss")
    val_losses = store.get("val_loss")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.plot(iter, train_losses)
    plt.plot(iter, val_losses)
    plt.legend(["train_loss", "val_loss"])
    fig.savefig(f"{config.output}/epoch_{epoch}/loss.png")
               
def plot_graph_acc(config, epoch, iter_list, acces, name):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel("steps")
    plt.ylabel("acc")
    plt.plot(iter_list, acces)
    plt.legend([name])
    fig.savefig(f"{config.output}/epoch_{epoch}/{name}.png")
 
        
def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, scaler, logger, store):
    os.makedirs(f"{config.output}/epoch_{epoch}", exist_ok=True)
    plot_graph(config, epoch, store)
    if "mlm_acc" in store.names:
        plot_graph_acc(config, epoch, store.get("iter"), store.get("mlm_acc"), "mlm_acc")
    elif "pwfi_acc" in store.names:
        plot_graph_acc(config, epoch, store.get("iter"), store.get("pwfi_acc"), "pwfi_acc")
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'epoch': epoch,
                  'config': config,
                  'val_losses': store.get("val_loss"),
                  'train_losses': store.get("train_loss"),
                  }
    torch.save(save_state, f"{config.output}/epoch_{epoch}/checkpoint.cpt")
    logger.info(f"{config.output}/epoch_{epoch}/checkpint.cpt saved !!!")

def save_checkpoint_only_score(config, epoch, logger, store):
    os.makedirs(f"{config.output}/epoch_{epoch}", exist_ok=True)
    plot_graph(config, epoch, store)
    save_state = {
                  'epoch': epoch,
                  'config': config,
                  'val_losses': store.get("val_loss"),
                  'train_losses': store.get("train_loss"),
                  'score': store.get("score"),
                  }
    torch.save(save_state, f"{config.output}/epoch_{epoch}/checkpoint.cpt")
    logger.info(f"{config.output}/epoch_{epoch}/checkpint.cpt saved !!!")



class Store(object):
    def __init__(self, names=None):
        self.names = names        
        self.reset()

    def reset(self):
        store = {}
        for name in self.names:
            store[name] = []
        self.store = store
    
    def update(self, val):
        for name in self.names:
            self.store[name].append(val[name])
    
    def get(self, name):
        return self.store[name]



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=None, fmt=':f'):
        self.names = None
        if type(name) is list:
            self.names = name
        else:
            self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.store = {}
        if self.names is not None:
            for name in self.names:
              self.store[name] = {"list": [], "val": 0, "avg": 0, "sum": 0, "count": 0}
        else:
            self.__val = 0
            self.__avg = 0
            self.__sum = 0
            self.__count = 0

    def update(self, val, n=1):
        if self.names is not None:
            for name in self.names:
              self.store[name]["val"] = val[name]
              self.store[name]["sum"] += val[name]*n
              self.store[name]["count"] += n
              self.store[name]["avg"] = self.store[name]["sum"] / self.store[name]["count"]
        else:          
            self.__val = val
            self.__sum += val * n
            self.__count += n
            self.__avg = self.__sum / self.__count

    @property
    def avg(self):
        if self.names is not None:
            store = {}
            for name in self.names:
                store[name] = self.store[name]["avg"]
            return store
        else:
            return self.__avg
        
    @property
    def val(self):
        if self.names is not None:
            store = {}
            for name in self.names:
                store[name] = self.store[name]["val"]
            return store
        else:
            return self.__val

    @property
    def count(self):
        if self.names is not None:
            store = {}
            for name in self.names:
                store[name] = self.store[name]["count"]
            return store
        else:
            return self.__count

    @property
    def sum(self):
        if self.names is not None:
            store = {}
            for name in self.names:
                store[name] = self.store[name]["sum"]
            return store
        else:
            return self.__sum
        

###util
####### dataset 
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
            attetnion_mask = [1] * self.max_len
        else:
            input_ids = [self.cls] + input_ids + [self.sep] + [self.pad] * (self.max_len - token_len - 2)
            bbox = [[0, 0, 0, 0]] + bbox + [[0, 0, 0, 0]] + [[0, 0, 0, 0]] * (self.max_len - token_len - 2)
            attetnion_mask = [1] * (token_len + 2) + [0] * (self.max_len - token_len - 2)
            
            
        return  {
            "input_ids": input_ids,
            "bbox": bbox,
            "pixel_values": self.data[idx]["pixel_values"],
            "attention_mask": attetnion_mask, 
            "mlm_labels": self.data[idx]["mlm_labels"],
            "mask_indexes": self.data[idx]["mask_indexes"],
        }



##IOU PWFI
def iou_for_pwfi(a, b):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])
    
    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    # shape=(N,)のnumpy配列。Nは矩形の数
    b_area = (b[:,2] - b[:,0] + 1) \
             * (b[:,3] - b[:,1] + 1)
    
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h
    
    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (b_area)
    mask_indexes = iou > 0.4
    cand_words = np.where(mask_indexes == True)[0].tolist()
    return cand_words

##IOU PIFW
def iou_for_pifw(a, b):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])
    
    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    # shape=(N,)のnumpy配列。Nは矩形の数
    b_area = (b[:,2] - b[:,0] + 1) \
             * (b[:,3] - b[:,1] + 1)
    
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h
    
    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (a_area)
    return iou


def check_mask(config, name2anno, save_path, vocab, font='NotoMono-Regular.ttf'):
    if isinstance(name2anno, dict):
        key = random.choice(list(name2anno.keys()))
        doc_anno = name2anno[key]
        key = random.choice(list(doc_anno.keys()))
        anno = doc_anno[key]
    elif isinstance(name2anno, list):
        index = random.randint(0, len(name2anno)-1)
        anno = name2anno[index]
    image_path = f"{config.input_file_dir}/{anno['save_path']}"
    image = Image.open(image_path)
    patch_bboxes = patch_bbox(896, 896, 64)
    img = image.resize((896, 896))
    #テスト
    draw = ImageDraw.Draw(img)
    for b in patch_bboxes:
      draw.rectangle(b, outline=(225, 0, 0))
      
    num = 0
    num = 0
    draw = ImageDraw.Draw(img)
    bboxes = anno["split_bboxes"][num]*896
    #font
    font = ImageFont.truetype(font, 10)
    ###patch
    t = 0
    for i in range(len(anno["patch_indexes"][num])):
        if anno["patch_indexes"][num][i] == True:
            draw.rectangle(patch_bboxes[i], outline=(0, 255, 255))
            label_id = anno["pwfi_labels"][num][t]
            t += 1
            draw.text((patch_bboxes[i][0],patch_bboxes[i][1]), vocab[label_id], "red", font=font)
    t = 0
    for i in range(len(anno["mask_indexes"][num])):
        if anno["mask_indexes"][num][i] == True:
            draw.rectangle(bboxes[i], outline=(0, 255, 0))
            label_id = anno["mlm_labels"][num][t]
            t += 1
            draw.text((bboxes[i][0], bboxes[i][1]-10), vocab[label_id], "green", font=font)
    img.save(save_path)


def save_big_data(data, save_path, num_split=4):
    if not isinstance(data, list):
        print("error")
    os.makedirs(save_path, exist_ok=True)
    data_len = len(data)
    print(data_len / num_split)
    chunk_len = int(data_len / num_split)
    for i, s in enumerate(range(0, data_len, chunk_len)):
        if i == num_split:
          chunk = data[s:]
          save_name = f"{s}-{data_len}.pkl"
        else:
          chunk = data[s: s+chunk_len]
          save_name = f"{s}-{s+chunk_len}.pkl"
        with open(f"{save_path}/{save_name}", "wb") as f:
            pickle.dump(chunk, f)
        print(f"saved ... {save_path}/{save_name}")
    print("saved all data")

def load_file(file_path, logger=None):
    if os.path.isfile(file_path) == True:
        if logger is not None:
            logger.info(f"load... {file_path}")
        else:
            print(f"load... {file_path}")
        with open(file_path, "rb") as f:
           data = pickle.load(f)
           
    elif os.path.isdir(file_path) == True:
        file_names = os.listdir(file_path)
        file_names.sort(key=len)
        data = []
        for name in file_names:
            if logger is not None:
                logger.info(f"load... {file_path}/{name}")
            else:
                print(f"load... {file_path}/{name}")
        
            with open(f"{file_path}/{name}", "rb") as f:
                chunk = pickle.load(f)
            data += chunk
    else:
        print("should be file_name or file_dir")
    
    return data

def count_prameters(model):
    million = 1000000
    state_dict = model.state_dict()
    cont = 0
    for x in state_dict:
        cont += state_dict[x].numel()
    return  cont / million