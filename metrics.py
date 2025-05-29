import torch.nn.functional as F
from PIL import Image
import json
from tqdm import tqdm
from utils.util import get_mean


# CLIP-T
def cal_CLIP_T(model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            prompt = entry['prompt']
            img_feats = model(img)
            text_feats = model(prompt)
            clip_score = F.cosine_similarity(img_feats, text_feats)
            metric_val_list.append(clip_score) 
    avg_val = get_mean(metric_val_list)

    return avg_val

# CLIP-I
def cal_CLIP_I(model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img1 = Image.open(entry['img_path'])
            img2 = Image.open(entry['img_path2'])
            img_feats = model(img1)
            text_feats = model(img2)
            clip_score = F.cosine_similarity(img_feats, text_feats)
            metric_val_list.append(clip_score) 
    avg_val = get_mean(metric_val_list)

    return avg_val

# DINO
def cal_DINOv2(model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img1 = Image.open(entry['img_path'])
            img2 = Image.open(entry['img_path2'])
            img_feats = model(img1)
            text_feats = model(img2)
            clip_score = F.cosine_similarity(img_feats, text_feats)
            metric_val_list.append(clip_score) 
    avg_val = get_mean(metric_val_list)

    return avg_val
 
# DreamSim
def cal_DreamSim(metric_model, jsonl_path, device="cuda"):
    metric_val_list = []
    preprocess, model_func = metric_model[0], metric_model[1]

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img1 = Image.open(entry['img_path'])
            img2 = Image.open(entry['img_path2'])
            img1 = preprocess(img1).to(device)
            img2 = preprocess(img2).to(device)
            distance = model_func(img1, img2)
            metric_val_list.append(distance)
    avg_val = get_mean(metric_val_list)

    return avg_val

# LPIPS
def cal_LPIPS(metric_model, jsonl_path):
    metric_val_list = []
    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img1 = Image.open(entry['img_path'])
            img2 = Image.open(entry['img_path2'])
            val = metric_model(img1, img2)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)
                
    return avg_val

# LAION_Aes
def cal_LAION_Aes(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)        
    avg_val = get_mean(metric_val_list)

    return avg_val

# Q-Align
def cal_Q_Align(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img, task_='aesthetic')
            metric_val_list.append(val)    
    avg_val = get_mean(metric_val_list)

    return avg_val

# Q-Align-IQ
def cal_Q_Align_IQ(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img, task_='quality')
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)

    return avg_val

# CLIP-IQA
def cal_CLIP_IQA(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)
    
    return avg_val

# TOPIQ-NR
def cal_TOPIQ_NR(metric_model, jsonl_path):
    metric_val_list = []
    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)

    return avg_val

# MANIQA
def cal_MANIQA(metric_model, jsonl_path):
    metric_val_list = []
    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)

    return avg_val

# HYPERIQA
def cal_HYPERIQA(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)

    return avg_val


# **************************************************************************


METRIC_CAL_FUNC = {
    "CLIP-I": cal_CLIP_I,
    "CLIP-T": cal_CLIP_T,
    "DINO": cal_DINOv2,
    "DreamSim": cal_DreamSim,
    "LPIPS":  cal_LPIPS,
    "LAION-Aes":  cal_LAION_Aes,
    "Q-Align": cal_Q_Align,
    "Q-Align-IQ": cal_Q_Align_IQ,
    "CLIP-IQA": cal_CLIP_IQA,
    "TOPIQ-NR": cal_TOPIQ_NR,
    "MANIQA": cal_MANIQA,
    "HYPERIQA": cal_HYPERIQA,
}