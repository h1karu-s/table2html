import os
import argparse
import random
import math
import json
import itertools
from PIL import Image
import pickle
import numpy as np
import torch
import sys
sys.path.append("./src/")
from utils.logger import create_logger
from utils.utils import *
from transformers import BartTokenizer, AutoProcessor, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor
from utils.mask_generator import MaskingGenerator

##RuntimeError: unable to open shared memory object </torch_3461140_3031248365_2027> in read-write mode: Too many open files (24)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

window_size = (14, 14)
num_masking_patches = 58
max_mask_patches_per_block = 2
min_mask_patches_per_block = 1
mask_generator = MaskingGenerator(
            window_size, num_masking_patches=num_masking_patches,
            max_num_patches=max_mask_patches_per_block,
            min_num_patches=min_mask_patches_per_block,
        )

# list_of_tokens = ["thead", "tbody", "</", " rowspan=", " colspan="]
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
    iou = intersect / (a_area)
    mask_indexes = iou > 0.5
    cand_words = np.where(mask_indexes == True)[0].tolist()
    return cand_words


def text_mask_generate(seq_len, mask_ratio):
    indexes = list(range(0, seq_len))
    num_mask = int(len(indexes) * mask_ratio)
    random.shuffle(indexes)
    mask_token_index = indexes[:num_mask]
    mask_token_index.sort()
    return mask_token_index

def calc_relation_maskwords(table_structure, mask_token_indexes, word_indexes):
    ###maks_token_indexのtableでの座標を得る
    i_xy = {}
    for subword_i in mask_token_indexes:
        i = word_indexes[subword_i]
        # ##指定したcellのpositionを取得
        # column = []
        # rows = []
        for x in range(len(table_structure)):
            for y in range(len(table_structure[0])):
                if table_structure[x][y][0] == i:
                    i_xy[i] = (x, y)

    mask_pair = []
    for pair in itertools.combinations(mask_token_indexes, 2):
      mask_pair.append(pair)

    row_cells = {}
    col_cells = {}
    for i, (x, y) in i_xy.items():
        row_cells[i] =  [table_structure[x][t][0] for t in range(0, len(table_structure[0])) if (table_structure[x][y][0] != table_structure[x][t][0] and table_structure[x][t][0] != -1)]
        col_cells[i] = [table_structure[t][y][0] for t in range(0, len(table_structure)) if (table_structure[x][y][0] !=  table_structure[t][y][0] and table_structure[t][y][0] != -1)]
    relations = []
    random.shuffle(mask_pair)
    #sample_cell: 0, col: 1, row: 2, othre: 3
    for pair in mask_pair[:30]:
      i_subword, j_subword = pair
      i = word_indexes[i_subword]
      j = word_indexes[j_subword]
      if i == j:
        rel = 0
      if j in col_cells[i]:
        rel = 1
      elif j in row_cells[i]:
        rel = 2
      else:
        rel = 3
      relations.append((i_subword, j_subword, rel))
      # relations.append((i, j, rel))
    return relations


def preprocessing(data, input_file_dir, max_length):
    tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
    pad_id = tokenizer.pad_token_id
    patch_bboxes = patch_bbox(224, 224, 16)
    new_data = []
    for d in data:
        sample = {}
        encoding = tokenizer(text=d["ocr_tokens"], boxes=d["ocr_bboxes"], max_length=max_length, truncation=True, word_labels=list(range(0, len(d["ocr_tokens"]))))

        ##tokenizerのwordとsubwordの対応付け
        prev = -100
        word_indexes = []
        for index in encoding["labels"]:
            if index == -100:
              index = prev
              word_indexes.append(index)
            else:
              word_indexes.append(index)
            prev = index
        word_indexes[-1] = -100
        # encoding["word_indexes"] = word_indexes
        encoding.pop("labels")

        #### maskを作成
        ###<s></s>のため, -2
        text_mask_indexes = text_mask_generate(len(encoding["input_ids"])-2, 0.2)
        mask_token_indexes = [i+1 for i in text_mask_indexes]
        sample["mask_token_index"] = mask_token_indexes
        ## tokenにmaskをする
        mask_id = tokenizer.mask_token_id
        is_header_label = []
        for i in mask_token_indexes:
          encoding["input_ids"][i] = mask_id
          # encoding["bbox"][i] = ["<mask>"]*4
          is_header_label.append(d["is_header"][word_indexes[i]])
        sample["is_header_label"] = is_header_label

        table_structure = d["table_structure"]
        relations = calc_relation_maskwords(table_structure, mask_token_indexes, word_indexes)
        sample["relations"] = relations
        
        
        #### image
        ##mim
        # bool_mi_pos = mask_generator()
        bool_mi_pos = torch.from_numpy(mask_generator()).flatten(0).unsqueeze(0)
        sample["bool_mi_pos"] = bool_mi_pos
        mim_labels = torch.tensor(d["visual_token"]).unsqueeze(0)[bool_mi_pos.to(torch.bool)]
        sample["mim_labels"] = mim_labels
        ###image text matched
        if random.uniform(0, 100) <= 20:
            i = random.randint(0, len(data)-1)
            encoding["pixel_values"] = input_file_dir + "/" + data[i]["image_path"]
            # "True"
            sample["replace_image"] = 1
        else:
          encoding["pixel_values"] = input_file_dir + "/" + d["image_path"]
        #   "False"
          sample["replace_image"] = 0
        
        
        ###image text alignment
        ITA_labels = []
        ##1: coverd, 0: not coverd
        for i in mask_token_indexes:
            text_bbox = (np.array(encoding["bbox"][i])*224).astype(np.int32)
            if iou_np(text_bbox, np.array(patch_bboxes)[bool_mi_pos.to(torch.bool)[0]]):
              ITA_labels.append(1)
            else:
              ITA_labels.append(0)
        sample["ITA_labels"] = ITA_labels
        
        new_data.append(sample)
        
        ### padding
        encoding["input_ids"] = encoding["input_ids"] + [pad_id]*(max_length-len(encoding["input_ids"]))
        encoding["bbox"] = encoding["bbox"] + [[0, 0, 0, 0]]*(max_length-len(encoding["bbox"]))
        encoding["attention_mask"] = encoding["attention_mask"] + [0]*(max_length-len(encoding["attention_mask"]))
        sample["encoding_inputs"] = encoding
    return new_data
        
        

### load data and preprocessing shape of [{"input_ids": [], "bbox": [], "label": [], ..}, {}, ...]
def preprocessing_data(args, model=None, logger=None):
    dataset_train = None
    dataset_val = None
    dataset_test = None
    #tokenizer
    ##load data
    if args.input_file:
        with open(f"{args.input_file_dir}/preprocessing_dir/{args.input_file}", "rb") as f:
            data = pickle.load(f)
        if args.datasize is not None:
            data = data[:args.datasize]
        new_data = preprocessing(data, input_file_dir=args.input_file_dir, max_length=args.max_length)
        dataset_train = new_data
    
    if args.input_val_file:
        with open(f"{args.input_file_dir}/preprocessing_dir/{args.input_val_file}", "rb") as f:
            data = pickle.load(f)
        if args.datasize is not None:
            data = data[:args.datasize]
        new_data = preprocessing(data, input_file_dir=args.input_file_dir, max_length=args.max_length)
        dataset_val = new_data
         
    if args.input_test_file:
        with open(f"{args.input_file_dir}/preprocessing_dir/{args.input_test_file}", "rb") as f:
            data = pickle.load(f)
        if args.datasize is not None:
            data = data[:args.datasize]
        new_data = preprocessing(data, input_file_dir=args.input_file_dir, max_length=args.max_length)
        dataset_test = new_data
    
    return dataset_train, dataset_val, dataset_test



class collate_fn:
    def __init__(self, config, logger=None):
        self.feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False, size=config.input_size)
        self.logger = logger

    def __call__(self, batch):
        input_dict = {}
        pixel_values = []
        for b in batch:
            image = Image.open(b["encoding_inputs"]["pixel_values"]).convert("RGB")
            pixel_values.append(self.feature_extractor(image)["pixel_values"][0])
        input_dict["pixel_values"] = torch.tensor(pixel_values)
        
        for i in ["input_ids", "bbox", "attention_mask"]:
            if i == "bbox":
                # input_dict[i] = torch.tensor([b["encoding_inputs"][i]*1000 for b in batch]).to(torch.long)
                input_dict[i] = (torch.tensor([b["encoding_inputs"][i] for b in batch])*1000).to(torch.long)
            else:
                input_dict[i] = torch.tensor([b["encoding_inputs"][i] for b in batch])
        
        input_dict["bool_mi_pos"] = torch.concat([b["bool_mi_pos"] for b in batch], 0)
        relations = [b["relations"] for b in batch]
        mim_labels = [b["mim_labels"] for b in batch]
        mask_indexes = [b["mask_token_index"] for b in batch]

        
        labels = {}
        labels["mim_labels"] = [b["mim_labels"] for b in batch]
        labels["is_header_labels"] = [torch.tensor(b["is_header_label"]).to(torch.long) for b in batch]
        labels["ita_labels"] = [b["ITA_labels"] for b in batch]
        labels["itm_labels"] = [[b["replace_image"]] for b in batch]
        
        return input_dict, relations, mask_indexes, labels


def main(args):   
    dataset = preprocessing_data(args, logger)
    
    logger.info(f"saving ... {args.output}/{args.save_name}")
    save_big_data(dataset, f"{args.output}/{args.save_name}")
    logger.info(f"saved ... {args.output}/{args.save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #data
    parser.add_argument("--input_file_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--datasize", type=int)
    args = parser.parse_args()
    ## os setting
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

   
    os.makedirs(args.output, exist_ok=True)
    logger = create_logger(output_dir=args.output, dist_rank=args.save_name, name=f"{args.save_name}")

    logger.info(json.dumps(vars(args)))

    main(args)
    