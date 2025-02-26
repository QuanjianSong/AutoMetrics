from models.get_models import init_metric_model
from metrics import cal_Q_Align


if __name__ == '__main__':
    metric_list = ['CLIP-I', "CLIP-T", "DINO", "DreamSim",
                #    'LAION-Aes',
                   'Q-Align',
                   ]
    model = init_metric_model(metric_list, "cuda")

    gen_img_dict = '/home/sqj/code/SceneDecorator/output_with_img.jsonl'
    # avg_val = cal_CLIP_T(model['CLIP-T'], gen_img_dict)
    # avg_val = cal_CLIP_I(model['CLIP-I'], gen_img_dict)
    # print(f"avg:{avg_val}")
    # avg_val = cal_DINOv2(model['DINO'], gen_img_dict)
    # print(f"avg:{avg_val}")
    # avg_val = cal_DreamSim(model['DreamSim'], gen_img_dict)
    # print(f"avg:{avg_val}")
    # avg_val = cal_LAION_Aes(model['LAION-Aes'], gen_img_dict)
    # print(f"avg:{avg_val}")
    avg_val = cal_Q_Align(model['Q-Align'], gen_img_dict)
    print(f"avg:{avg_val}")