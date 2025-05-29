# AutoMetrics -- This is a Pytorch-integrated pipeline codebase for Metrics in AIGC

## News
<pre style="white-space: pre-wrap;">
• 🔥 The code of AutoMetrics has been released. It is based on <a href="https://github.com/QuanjianSong/T2I-Metrics">T2I-Metrics</a>. We plan to consolidate the two repositories in the near future, appreciate your continued interest!
</pre>
  
## 0. Project Introduction
This repository focuses on AIGC evaluation metrics and extends T2I-Metrics by adding metrics such as the DINO score, DreaSim score, aesthetic evaluations, and more.

## 1. Environment Configuration

#### 1.1 Installation with the requirement.txt

```
pip install -r requirements.txt
```

#### 1.2 Installation with environment.yaml

```
conda env create -f environment.yaml
```

#### 1.3 Installation with the pip command

- Install PyTorch:

```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116  # Choose a version that suits your GPU
```

- Install Scipy

```
pip install scipy
```

- Install CLIP:

```
pip install git+https://github.com/openai/CLIP.git
```

## 2. Model Weights Download

You need to download the inception_v3_google.pth, pt_inception.pth, and ViT-B-32.pt weights files and place them in the checkpoints folder. We have integrated them into the following links for your convenience.

[Baidu cloud disk link, extraction code: fpfp](https://pan.baidu.com/s/1nGPq5y2OfCumMQkY6ROKGA?)

## 3. Data Preparation

Before starting the evaluation, you need to prepare the corresponding jsonl files in advance. Different evaluation metrics require reading different types of jsonl files. These generally fall into three categories: image-prompt pairs, image-image pairs, and single images. Each line in the jsonl file should include the appropriate file paths. We provide example files in the `./examples` directory to help you construct your own.

## 4. Quick Start

We provide simple examples in `main.py` to quickly get started with metric evaluation.

```
python main.py
```

Different evaluation metrics require reading different jsonl files, and we have provided corresponding examples in the previous step.

## 5. Reference Source

[T2I-Metrics](https://github.com/QuanjianSong/T2I-Metrics)
