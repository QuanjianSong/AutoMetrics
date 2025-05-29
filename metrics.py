import torch.nn.functional as F
from PIL import Image
import json
from tqdm import tqdm
from utils import get_mean


# CLIP-T
def cal_CLIP_T(model, gen_img_dict, seg_img_dict={}, ori_img_dict={}, placeholder_token="<v>"):
    # get sample names

    metric_val_list = []

    # 打开并遍历JSONL文件
    with open(gen_img_dict, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            # 将每一行解析为字典
            entry = json.loads(line.strip())
            img = Image.open(entry['path'])
            prompt = entry['prompt']
            #
            # breakpoint()
            # print(entry)
            #
            img_feats = model(img)
            text_feats = model(prompt)
            clip_score = F.cosine_similarity(img_feats, text_feats)
            metric_val_list.append(clip_score) 
    avg_val = get_mean(metric_val_list)

    return avg_val


# CLIP-I
def cal_CLIP_I(model, gen_img_dict):
    # group images by the sample name
    metric_val_list = []

    # 打开并遍历JSONL文件
    with open(gen_img_dict, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            # 将每一行解析为字典
            # breakpoint()
            entry = json.loads(line.strip())
            img1 = Image.open(entry['path'])
            img2 = Image.open(entry['img_address'])
            #
            # breakpoint()
            # print(entry)
            #
            img_feats = model(img1)
            text_feats = model(img2)
            clip_score = F.cosine_similarity(img_feats, text_feats)
            metric_val_list.append(clip_score) 
    avg_val = get_mean(metric_val_list)

    return avg_val

# DINO
def cal_DINOv2(model, gen_img_dict):
    metric_val_list = []

    # 打开并遍历JSONL文件
    with open(gen_img_dict, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            # 将每一行解析为字典
            # breakpoint()
            entry = json.loads(line.strip())
            img1 = Image.open(entry['path'])
            img2 = Image.open(entry['img_address'])
            #
            # breakpoint()
            # print(entry)
            #
            img_feats = model(img1)
            text_feats = model(img2)
            clip_score = F.cosine_similarity(img_feats, text_feats)
            metric_val_list.append(clip_score) 
    avg_val = get_mean(metric_val_list)

    # group images by the sample name
    # grouped_seg_img_dict = group_images_by_sample_name(seg_img_dict, type="seg")
    # grouped_ori_img_dict = group_images_by_sample_name(ori_img_dict, type="ori")
    # assert grouped_seg_img_dict.keys() == grouped_ori_img_dict.keys()

    # metric_val_list = []
    # for name, img_list in grouped_seg_img_dict.items():
    #     # get features
    #     seg_img_feats = model(img_list)
    #     ori_img_feats = model(grouped_ori_img_dict[name])
    #     seg_img_feats = F.normalize(seg_img_feats, dim=1)
    #     ori_img_feats = F.normalize(ori_img_feats, dim=1)
    #     # calculate the metric in an image group of the same sample
    #     cos_mat = torch.mm(seg_img_feats, ori_img_feats.transpose(0, 1))
    #     # cos_mat = (cos_mat + 1) / 2
    #     metric_val_list.append(cos_mat.mean())
    # avg_val = get_mean(metric_val_list)
    return avg_val


    
# DreamSim
def cal_DreamSim(metric_model, gen_img_dict, device="cuda"):
    preprocess, model_func = metric_model[0], metric_model[1]


    # calculate the metric in an image group of the same sample
    metric_val_list = []

    with open(gen_img_dict, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            # 将每一行解析为字典
            # breakpoint()
            entry = json.loads(line.strip())
            img1 = Image.open(entry['path'])
            img2 = Image.open(entry['img_address'])
            img1 = preprocess(img1).to(device)
            img2 = preprocess(img2).to(device)
            distance = model_func(img1, img2)
            metric_val_list.append(distance)
    avg_val = get_mean(metric_val_list)


    # for name, img_list in grouped_seg_img_dict.items():
    #     vals = []
    #     for seg_img in img_list:
    #         seg_img = preprocess(seg_img).to(device)
    #         for ori_img in grouped_ori_img_dict[name]:
    #             ori_img = preprocess(ori_img).to(device)
    #             distance = model_func(seg_img, ori_img)
    #             vals.append(distance)
    #     metric_val_list.append(get_mean(vals))
    # avg_val = get_mean(metric_val_list)

    return avg_val


# LPIPS
def cal_LPIPS(metric_model, gen_img_dict, seg_img_dict, ori_img_dict):
    # group images by the prompt
    grouped_gen_img_dict = {}
    for name, img in gen_img_dict.items():
        prefix = name.rsplit('-', 1)[0]
        grouped_gen_img_dict.setdefault(prefix, []).append(img)
    
    # calculate the metric in an image group of the same prompt
    metric_val_list = []
    for img_list in grouped_gen_img_dict.values():
        vals = []
        for i in range(len(img_list)-1):
            img1 = img_list[i]
            for j in range(i+1, len(img_list)):
                img2 = img_list[j]
                # img2 = img2.resize((img1.size[0], img1.size[1]))
                val = metric_model(img1, img2)
                vals.append(val)
        metric_val_list.append(get_mean(vals))
    avg_val = get_mean(metric_val_list)
                
    return avg_val


# LAION_Aes
def cal_LAION_Aes(metric_model, gen_img_dict):
    metric_val_list = []
    with open(gen_img_dict, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            # 将每一行解析为字典
            # breakpoint()
            entry = json.loads(line.strip())
            img = Image.open(entry['absolute_path'])
            val = metric_model(img)
            metric_val_list.append(val)        
    avg_val = get_mean(metric_val_list)
    return avg_val



# Q-Align
def cal_Q_Align(metric_model, gen_img_dict):
    metric_val_list = []

    with open(gen_img_dict, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            # 将每一行解析为字典
            # breakpoint()
            entry = json.loads(line.strip())
            img = Image.open(entry['absolute_path'])
            val = metric_model(img, task_='aesthetic')
            metric_val_list.append(val)

    # for img in gen_img_dict.values():
    #     val = metric_model(img, task_='aesthetic')
    #     metric_val_list.append(val)
    
    
    avg_val = get_mean(metric_val_list)
    return avg_val



# Q-Align-IQ
def cal_Q_Align_IQ(metric_model, gen_img_dict, seg_img_dict, ori_img_dict):
    metric_val_list = []
    for img in gen_img_dict.values():
        val = metric_model(img, task_='quality')
        metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)
    return avg_val


# CLIP-IQA
def cal_CLIP_IQA(metric_model, gen_img_dict, seg_img_dict, ori_img_dict):
    metric_val_list = []
    for img in gen_img_dict.values():
        val = metric_model(img)
        metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)
    return avg_val


# TOPIQ-NR
def cal_TOPIQ_NR(metric_model, gen_img_dict, seg_img_dict, ori_img_dict):
    metric_val_list = []
    for img in gen_img_dict.values():
        val = metric_model(img)
        metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)
    return avg_val


# MANIQA
def cal_MANIQA(metric_model, gen_img_dict, seg_img_dict, ori_img_dict):
    metric_val_list = []
    for img in gen_img_dict.values():
        val = metric_model(img)
        metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)
    return avg_val


# HYPERIQA
def cal_HYPERIQA(metric_model, gen_img_dict, seg_img_dict, ori_img_dict):
    metric_val_list = []
    for img in gen_img_dict.values():
        val = metric_model(img)
        metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)
    return avg_val


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