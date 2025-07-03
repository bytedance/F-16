# Improving LLM Video Understanding with 16 Frames Per Second

🚀🚀 Welcome to the repo of **F-16**!

F-16 is a powerful video large language model (LLM) that **perceives high-frame-rate videos**, which is developed by the Department of Electronic Engineering at Tsinghua University and ByteDance. 

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2503.13956'><img src='https://img.shields.io/badge/F_16_paper-PDF-green'></a><a href='[https://arxiv.org/abs/2503.13956](https://huggingface.co/tsinghua-ee/F-16)'><img src='https://img.shields.io/badge/F_16-checkpoint-yellow'></a>
</div>

## 🔥 News

- **2025-07-03**: We release the final checkpoint of F-16. 
- **2025-06-18**: We release the code of F-16.

## ⚡️ Future Plans

- ~~Release the code.~~
- ~~Release final F-16.~~

## 🌈 How to Use

### How to train a model

1. Prepare the dataset following `scripts/example_sft.json`.
2. Download LLaVA-OneVision Model from [huggingface](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov).
3. Modify the parameters in `scripts/train_sft.sh`.
4. Run `bash scripts/train_sft.sh`.

### How to evaluate a checkpoint

1. Prepare the dataset following `scripts/example_sft.json`.
2. Modify the parameters in `scripts/eval.sh`.
3. Run `bash scripts/eval.sh`.

## 👀 Team

**Team Tsinghua**: Yixuan Li, Changli Tang, Jimin Zhuang, Yudong Yang, Guangzhi Sun, Chao Zhang

**Team ByteDance**: Wei Li, Zejun Ma

## ✨ Citation
If you find F-16 useful, please cite the paper:

```
@inproceedings{li2025improving,
  title={Improving LLM Video Understanding with 16 Frames Per Second},
  author={Li, Yixuan and Tang, Changli and Zhuang, Jimin and Yang, Yudong and Sun, Guangzhi and Li, Wei and Ma, Zejun and Zhang, Chao},
  booktitle={Proc. ICML},
  year={2025}, 
  address={Vancouver}
}
```
