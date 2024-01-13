import os
import argparse
import math
import json
from PIL import Image
import pickle
import torch
import sys
sys.path.append("./src/")
from utils.logger import create_logger
from utils.utils import *
from transformers import BartTokenizer, AutoProcessor


# list_of_tokens = [
#   "<table>",
#   "</table>",
#   "<thead>",
#   "</thead>",
#   "<tbody>",
#   "</tbody>",
#   "<tr>",
#   "</tr>",
#   "<td>",
#   "</td>",
#   "<b>",
#   "</b>",
# ]

list_of_tokens = ["thead", "tbody", "</", " rowspan=", " colspan="]


### load data and preprocessing shape of [{"input_ids": [], "bbox": [], "label": [], ..}, {}, ...]
def preprocessing_data(args, logger=None, decoder_tokenizer=None):
    dataset_train = None
    dataset_test = None
    #tokenizer
    if decoder_tokenizer is None:
        logger.info(f"preprocessing_data: loading... decoder_tokenizdr. add_vocab is {list_of_tokens}")
        decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        decoder_tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
    encoder_processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    ##load data
    with open(f"{args.input_file_dir}/preprocessing_dir/{args.input_file}", "rb") as f:
        data = pickle.load(f)
    if args.datasize is not None:
        data = data[:args.datasize]
    for d in data:
        image = Image.open(args.input_file_dir + "/" + d["image_path"])
        encoder_encoding = encoder_processor(image, d["input"]["tokens"], boxes=d["input"]["bboxes"], max_length=args.max_length, padding="max_length", return_tensors ="pt")
        decoder_encoding = decoder_tokenizer(d["html"], return_tensors="pt", max_length=args.max_length, padding="max_length")
        ##bbox >> 224pix
        encoder_encoding["bbox"] = (encoder_encoding["bbox"]*1000).to(torch.int32)
        d["encoding_inputs"] = encoder_encoding
        d["encoding_html"] = decoder_encoding
    
    return data


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
    