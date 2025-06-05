<div align="center">
<h1>
AutoMetrics: A Pipeline for Metrics in AIGC
<br>
[Official Code of PyTorch]
</h1>

<div>
    <a href='https://github.com/QuanjianSong' target='_blank' style='text-decoration:none'>Quanjian Song</a>
</div>

---

</div>

## 🎉 News
<pre style="white-space: pre-wrap;">
• 🔥 The code of AutoMetrics has been released. It extends the capabilities of <a href="https://github.com/QuanjianSong/T2I-Metrics">T2I-Metrics</a>.
</pre>
  
## 🎬 Overview
<div align="justify">
This repository focuses on AIGC evaluation metrics and extends T2I-Metrics by adding metrics such as the DINO score, DreaSim score, aesthetic evaluations, and more.
</div>


## 🔧 Environment
```
# Git clone the repo
git clone https://github.com/QuanjianSong/AutoMetrics.git

# Installation with the requirement.txt
conda create -n AutoMetrics python=3.8
conda activate AutoMetrics
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116  # Choose a version that suits your GPU
pip install scipy
pip install git+https://github.com/openai/CLIP.git

# Or installation with the environment.yaml
conda env create -f environment.yaml
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116  # Choose a version that suits your GPU
pip install scipy
pip install git+https://github.com/openai/CLIP.git
```



## 🤗 Checkpoint

You need to download the inception_v3_google.pth, pt_inception.pth, and ViT-B-32.pt weights files and place them in the checkpoints folder. We have integrated them into the following links for your convenience.

[Baidu cloud disk link, extraction code: fpfp](https://pan.baidu.com/s/1nGPq5y2OfCumMQkY6ROKGA?)

## 📖 Dataset
Before starting the evaluation, you need to prepare the corresponding jsonl files in advance. Different evaluation metrics require reading different types of jsonl files. These generally fall into three categories: image-prompt pairs, image-image pairs, and single images. Each line in the jsonl file should include the appropriate file paths. We provide example files in the `./examples` directory to help you construct your own.


## 4. 🚀 Start
We provide simple examples in `main.py` to quickly get started with metric evaluation.
```
python main.py
```

Different evaluation metrics require reading different jsonl files, and we have provided corresponding examples in the previous step.

## 5. 🎓 Bibtex
[T2I-Metrics](https://github.com/QuanjianSong/T2I-Metrics)
