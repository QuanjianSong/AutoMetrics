import os
import pyiqa
from dreamsim import dreamsim
from models.DINO import DINOv2Model
from models.CLIP import CLIPModel


def init_metric_model(metric_list, device):
    metric_model_dict = {}

    for metric_name in metric_list:
        if metric_name == "CLIP-I":
            model = metric_model_dict["CLIP-T"] \
                if "CLIP-T" in metric_model_dict else \
                CLIPModel(device=device)
        elif metric_name == "CLIP-T":
            model = metric_model_dict["CLIP-I"] \
                if "CLIP-I" in metric_model_dict else \
                CLIPModel(device=device)
        elif metric_name == "DINO":
            model = DINOv2Model(device=device)
        elif metric_name == "LPIPS":  # pyiqa
            model = pyiqa.create_metric('lpips', device=device)
        elif metric_name == "DreamSim":
            model_func, preprocess = dreamsim(pretrained=True, 
                                              cache_dir=os.path.expanduser("~/.cache/dreamsim"),
                                              device=device)
            model = [preprocess, model_func]
        elif metric_name == "LAION-Aes":  # pyiqa
            model = pyiqa.create_metric('laion_aes', device=device)
        elif metric_name == "Q-Align":  # pyiqa
            model = metric_model_dict["Q-Align"] \
                if "Q-Align" in metric_model_dict else \
                pyiqa.create_metric('qalign', device=device)
        elif metric_name == "Q-Align-IQ":  # pyiqa
            model = metric_model_dict["Q-Align-IQ"] \
                if "Q-Align-IQ" in metric_model_dict else \
                pyiqa.create_metric('qalign', device=device)
        elif metric_name == "CLIP-IQA":  # pyiqa
            model = pyiqa.create_metric('clipiqa', device=device)
        elif metric_name == "TOPIQ-NR":  # pyiqa
            model = pyiqa.create_metric('topiq_nr', device=device)
        elif metric_name == "MANIQA":  # pyiqa
            model = pyiqa.create_metric('maniqa', device=device)
        elif metric_name == "HYPERIQA":  # pyiqa
            model = pyiqa.create_metric('hyperiqa', device=device)
        else:
            raise ValueError("Wrong metric name!")
        
        metric_model_dict[metric_name] = model

    print("Metirc models were all loaded!")

    return metric_model_dict
