import os
import argparse
import math
import json
from PIL import Image
import pickle
import sys
sys.path.append("./src/")
from utils.logger import create_logger
from utils.utils import *


# list_of_tokens = ["thead", "tbody", "</", " rowspan=", " colspan="]
label2Id = {
  "col": 0,
  "row": 1,
  "no_rel": 2,
}


### load data and preprocessing shape of [{"input_ids": [], "bbox": [], "label": [], ..}, {}, ...]
def preprocessing_data(args, logger=None):
    dataset_train = None
    dataset_val = None
    dataset_test = None
    ##load data
    if args.input_file:
        with open(f"{args.input_file_dir}/preprocessing_dir/{args.input_file}", "rb") as f:
            data = pickle.load(f)
        if args.datasize is not None:
            data = data[:args.datasize]
        new_data = []
        for d in data:
            ##bbox >> 224pix
            d["encoding_inputs"] = args.input_file_dir + "/" + d["image_path"]
            new_rel = []
            for rel in d["knn_adja_rel"]:
                new_rel.append((rel[0], rel[1], label2Id[rel[-1]]))
            d["knn_adja_rel"] = new_rel
            new_data.append(d)
        dataset_train = new_data
        
    if args.input_val_file is not None:
        with open(f"{args.input_file_dir}/preprocessing_dir/{args.input_test_file}", "rb") as f:
            data = pickle.load(f)
        if args.datasize is not None:
            data = data[:args.datasize]
        new_data = []
        for d in data:
            d["encoding_inputs"] = args.input_file_dir + "/" + d["image_path"]
            new_rel = []
            for rel in d["knn_adja_rel"]:
                new_rel.append((rel[0], rel[1], label2Id[rel[-1]]))
            d["knn_adja_rel"] = new_rel
            new_data.append(d)
        dataset_val = new_data
    
    if args.input_test_file is not None:
        with open(f"{args.input_file_dir}/preprocessing_dir/{args.input_test_file}", "rb") as f:
            data = pickle.load(f)
        if args.datasize is not None:
            data = data[:args.datasize]
        new_data = []
        for d in data:
            ##bbox >> 224pix
            d["encoding_inputs"] = args.input_file_dir + "/" + d["image_path"]
            new_rel = []
            for rel in d["knn_adja_rel"]:
                new_rel.append((rel[0], rel[1], label2Id[rel[-1]]))
            d["knn_adja_rel"] = new_rel
            new_data.append(d)
        dataset_test = new_data
        
    return dataset_train, dataset_val, dataset_test


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
    