# MR-DCoT: Prototype-Anchored Generalized Manifold Regression for Unknown-Domain Object Detection
## Overview

Single-Domain Generalized Object Detection (Single-DGOD) aims to train an object detector using only a **single-source domain**, while maintaining strong robustness across multiple unseen target domains.

In this work, we propose **Manifold Regression with Visual-Text Dual Chain-of-Thought (MR-DCoT)**, a new framework that reformulates unknown-domain object detection as a **prototype-anchored manifold regression** problem. Instead of merely expanding the training distribution through finite augmentation or static textual prompts, MR-DCoT learns a stable rectification mechanism that guides off-manifold features back toward the source semantic manifold.

Specifically, MR-DCoT contains two key stages. First, a **Visual-Text Dual Chain-of-Thought** module generates structured off-manifold hard examples by coupling VLM-guided global semantic evolution with diffusion-guided local structural perturbations. Second, a **Class-Specific Prototype Anchoring** mechanism learns a robust manifold regression objective, which pulls deviant features toward class-conditional prototype neighborhoods while preserving instance-level semantic consistency.

<p align="center">
  <img src="Figure/f1.jpg" width="90%">
</p>

<p align="center">
<b>Figure 1.</b> Overview of the proposed <b>Manifold Regression with Visual-Text Dual Chain-of-Thought (MR-DCoT)</b> framework.
MR-DCoT establishes a closed loop of <b>Simulate-to-Deviate</b> and <b>Regress-to-Rectify</b>: the Visual-Text Dual-CoT module synthesizes structured off-manifold samples, while prototype-anchored manifold regression rectifies deviant features back toward the semantic manifold for robust unknown-domain object detection.
</p>

### Datasets
Set the environment variable DETECTRON2_DATASETS to the parent folder of the datasets

```
    path-to-parent-dir/
        /diverseWeather
            /daytime_clear
            /daytime_foggy
            ...
        /comic
        /watercolor
        /VOC2007
        /VOC2012 
```



## Training

We train our models using four NVIDIA RTX 4090 GPUs.

For Single-DGOD on Diverse Weather:

```bash
python train_cot_mrdcot.py --config-file configs/diverse_weather_mrdcot.yaml
```

## Evaluation

Evaluate the trained model on unseen target domains:

```bash
python train_cot_mrdcot.py --config-file configs/diverse_weather_mrdcot.yaml --eval-only MODEL.WEIGHTS all_outs/...
```

