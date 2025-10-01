<div align="center">
<img align="left" width="90" height="90" src="https://github.com/gy65896/Neptune-X/blob/main/img_file/logo_Neptune-X.png" alt="">
 
## [NeurIPS 2025] Neptune-X: Active X-to-Maritime Generation for Universal Maritime Object Detection

[![ArXiv](https://img.shields.io/badge/NeptuneX-ArXiv-red.svg)](https://arxiv.org/abs/2509.20745)
[![Paper](https://img.shields.io/badge/NeptuneX-Paper-yellow.svg)]()
[![Web](https://img.shields.io/badge/NeptuneX-Web-blue.svg)](https://gy65896.github.io/projects/NeurIPS2025_Neptune-X/index.html)
[![Poster](https://img.shields.io/badge/NeptuneX-Poster-green.svg)]()
[![Video](https://img.shields.io/badge/NeptuneX-Video-orange.svg)]()
 
</div>

<div align=center>
<img src="https://github.com/gy65896/Neptune-X/blob/main/img_file/abstract.jpg" width="720">
</div>

---
>**Neptune-X: Active X-to-Maritime Generation for Universal Maritime Object Detection**<br>  [Yu Guo](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=5qAe9ZMAAAAJ), [Shengfeng He](http://www.shengfenghe.com/), [Yuxu Lu](https://scholar.google.com.hk/citations?user=XXge2_0AAAAJ&hl=zh-CN), [Haonan An](https://scholar.google.com/citations?user=YlRk19IAAAAJ&hl=zh-CN&authuser=1), [Yihang Tao](https://scholar.google.com/citations?user=YopoapwAAAAJ&hl=zh-CN), [Huilin Zhu](https://scholar.google.com.hk/citations?hl=zh-CN&user=fluPrxcAAAAJ), [Jingxian Liu](https://scholar.google.com/citations?user=wIpZ_PoAAAAJ&hl=zh-CN), [Yuguang Fang](https://scholar.google.com/citations?user=cs45mqMAAAAJ&hl=zh-CN) <br>
>Annual Conference on Neural Information Processing Systems<br>
>**Spotlight Presentation, Acceptance Rate: 3.2% (688/21575)**

> **Abstract:** *Maritime object detection is essential for navigation safety, surveillance, and autonomous operations, yet constrained by two key challenges: the scarcity of annotated maritime data and poor generalization across various maritime attributes (e.g., object category, viewpoint, location, and imaging environment). To address these challenges, we propose Neptune-X, a data-centric generative-selection framework that enhances training effectiveness by leveraging synthetic data generation with task-aware sample selection. From the generation perspective, we develop X-to-Maritime, a multi-modality-conditioned generative model that synthesizes diverse and realistic maritime scenes. A key component is the Bidirectional Object-Water Attention module, which captures boundary interactions between objects and their aquatic surroundings to improve visual fidelity. To further improve downstream tasking performance, we propose Attribute-correlated Active Sampling, which dynamically selects synthetic samples based on their task relevance. To support robust benchmarking, we construct the Maritime Generation Dataset, the first dataset tailored for generative maritime learning, encompassing a wide range of semantic conditions. Extensive experiments demonstrate that our approach sets a new benchmark in maritime scene synthesis, significantly improving detection accuracy, particularly in challenging and previously underrepresented settings.*
---

## News 🚀
* **2025.09.27**: Code is released.
* **2025.09.19**: Neptune-X is accepted by [NeurIPS2025](https://neurips.cc/).

## Generation Architecture

</div>
<div align=center>
<img src="https://github.com/gy65896/Neptune-X/blob/main/img_file/generation.jpg" width="1080">
</div>

## Selection Strategy

</div>
<div align=center>
<img src="https://github.com/gy65896/Neptune-X/blob/main/img_file/selection.jpg" width="1080">
</div>

## MGD Dataset Construction

</div>
<div align=center>
<img src="https://github.com/gy65896/Neptune-X/blob/main/img_file/labelling.jpg" width="1080">
</div>

## Quick Start
 
### Install

- python 3.8.20
- cuda 11.7

```
# git clone this repository
git clone https://github.com/gy65896/Neptune-X.git
cd Neptune-X

# create new anaconda env
conda env create -f environment.yml
```

### Pretrained Models

Please download our pre-trained models and put them in  `./ckpts`.

```
git clone https://huggingface.co/gy65896/Neptune-X
```

### Inference



## Citation

```
@inproceedings{guo2025neptune-x,
  title={Neptune-X: Active X-to-Maritime Generation for Universal Maritime Object Detection},
  author={Guo, Yu and He, Shengfeng and Lu, Yuxu and An, Haonan and Tao, Yihang and Zhu, Huilin and Liu, Jingxian and Fang, Yuguang},
  booktitle={Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
