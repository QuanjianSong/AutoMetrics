from models.get_models import init_metric_model
from metrics import cal_Q_Align, cal_DINOv2, cal_Q_Align, cal_LAION_Aes


if __name__ == '__main__':
    metric_list = ["DINO",
                   'LAION-Aes',
                   'Q-Align',
                   ]
    model = init_metric_model(metric_list, "cuda")

    gen_img_dict = 'XXXX.jsonl'
    avg_val = cal_DINOv2(model['DINO'], gen_img_dict)
    print(f"avg:{avg_val}")
    # ------------------------------------------------------------------
    gen_img_dict = 'XXXX.jsonl'
    avg_val = cal_LAION_Aes(model['LAION-Aes'], gen_img_dict)
    print(f"avg:{avg_val}")
    # ------------------------------------------------------------------
    gen_img_dict = 'XXXX.jsonl'
    avg_val = cal_Q_Align(model['Q-Align'], gen_img_dict)
    print(f"avg:{avg_val}")