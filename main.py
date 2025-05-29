from models.get_models import init_metric_model
from metrics import cal_CLIP_I, cal_CLIP_T, cal_DINOv2, cal_DreamSim, cal_LPIPS, cal_LAION_Aes, \
                    cal_Q_Align, cal_Q_Align_IQ, cal_CLIP_IQA, cal_TOPIQ_NR, cal_MANIQA, cal_HYPERIQA

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


if __name__ == '__main__':
    # choose your metrics to load
    metric_list = [
        "CLIP-T",
        "DreamSim",
        "LAION-Aes",
    ]
    model = init_metric_model(metric_list, "cuda")

    # Specify the corresponding jsonl file to compute the related metrics.
    jsonl_path = 'examples/img_txt.jsonl' # for prompt-img pair
    avg_val = cal_DreamSim(model['DreamSim'], jsonl_path)
    print(f"avg:{avg_val}")
    # ------------------------------------------------------------------
    jsonl_path = 'examples/img_img.jsonl' # for img-img pair
    avg_val = cal_DINOv2(model['DINO'], jsonl_path)
    print(f"avg:{avg_val}")
    # ------------------------------------------------------------------
    jsonl_path = 'examples/img.jsonl' # for img
    avg_val = cal_LAION_Aes(model['LAION-Aes'], jsonl_path)
    print(f"avg:{avg_val}")
