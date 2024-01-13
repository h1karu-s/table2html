import os
import random
import json
import re
from PIL import Image

import pickle

from transformers import XLMRobertaTokenizer, LayoutLMv3Tokenizer

random.seed(42)

file_dir = "/data/pubtabnet" 
output_dir = "/data/pubtabnet/preprocessing_dir/"
full_file_name = "complete_html_my_vocab_v1.pkl"
# max_length = 200
# file_name="simple_html_mbart_layoutlmv3_200"

special_tag = ['<b>', '<strike>', '</i>', '</b>', '</strike>', '</underline>', '</sup>', '<sub>', '<i>', '</sub>', '<underline>', '<overline>', '<sup>', '</overline>']


add_vocab_list = ['<table>', '</table>', '<td>', '</td>', '<thead>', '</thead>', '<tr>', '<td', '</tr>', 'rowspan=', 'colspan=', '<b>', '<strike>', '</i>', '</b>', '</strike>', '</underline>', '</sup>', '<sub>', '<i>', '</sub>', '<underline>', '<overline>', '<sup>', '</overline>']

###crate full html (label)
def format_html(img):
    ''' Formats HTML code from tokenized annotation of img
    '''
    html_code = img['html']['structure']['tokens'].copy()
    #rowspan="2"などの"を除去する(lengthを少なくするため)
    html_code = [tag.replace('\"', '')if '\"' in tag else tag for tag in html_code]
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], img['html']['cells'][::-1]):
        # print(i)
        if cell['tokens']:
            ## tag(<b>など以外の文字をエスケープする)
            # cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
            cell = ''.join(cell["tokens"])
            html_code.insert(i + 1, cell)
    html_code = ''.join(html_code)
    html_code = f'<table>{html_code}</table>'
    return html_code


def create_inputs(img):
    inputs = {"tokens": [], "bboxes": []}
    image_path = f"{img['split']}/{img['filename']}"
    tmp = Image.open(f"{file_dir}{image_path}")
    image = tmp.copy()
    tmp.close()
    width, height = image.size
    cells = img["html"]["cells"]
    for cell in cells:
        if cell["tokens"] and "bbox" in cell.keys():
            tokens = [token for token in cell["tokens"] if token not in special_tag]
            tokens = "".join(tokens)
            bbox = cell["bbox"]
            x0 = bbox[0] / width
            y0 = bbox[1] / height
            x1 = bbox[2] / width
            y1 = bbox[3] / height
            bbox = [x0, y0, x1, y1]
            inputs["tokens"].append(tokens)
            inputs["bboxes"].append(bbox)
    return inputs, (width, height), image_path


with open(f"{file_dir}PubTabNet_2.0.0.jsonl") as f:
    jsonl_data = [json.loads(l) for l in f.readlines()]

#### htmlデータを作成
full_data = []
for i, img in enumerate(jsonl_data):
    html = format_html(img)
    inputs = create_inputs(img)
    full_data.append(
        {
            "filename": img["filename"],
            "split": img["split"],
            "ocr_tokens": inputs[0]["tokens"],
            "ocr_bboxes": inputs[0]["bboxes"],
            "width": inputs[1][0],
            "height": inputs[1][1],
            "image_path": inputs[2],
            "html":  html,
        }
    )
    if i % 1000 == 0:
        print(i)

print("all datas length: ", len(full_data))

encoder_tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
decoder_tokenizer = XLMRobertaTokenizer.from_pretrained("facebook/mbart-large-50")
decoder_tokenizer.add_tokens(sorted(set(add_vocab_list)))
for d in full_data:
    encoder_length = len(encoder_tokenizer(d["ocr_tokens"], boxes=d["ocr_bboxes"])["input_ids"])
    decoder_length = len(decoder_tokenizer(d["html"])["input_ids"])
    d["encoder_length"] = encoder_length
    d["decoder_length"] = decoder_length

save_data = {
  "encoder_tokenizer": "microsoft/layoutlmv3-base",
  "decoder_tokenizer": "facebook/mbart-large-50",
  "data": full_data,
  }

with open(f"{output_dir}{full_file_name}", "wb") as f:
    pickle.dump(save_data, f)


print("save data!")