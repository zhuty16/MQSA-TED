# Multi-Query Self-Attention with Transition-Aware Embedding Distillation (MQSA-TED)

This is our Tensorflow implementation for the paper:

>Tianyu Zhu, Yansong Shi, Yuan Zhang, Yihong Wu, Fengran Mo, and Jian-Yun Nie. "Collaboration and Transition: Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation." WSDM 2024.

## Introduction
Graph-based Embedding Smoothing (GES) is a general framework for improving sequential recommendation methods with sequential and semantic item graphs.

![](https://github.com/zhuty16/MQSA-TED/blob/master/framework.jpg)

## Citation

## Environment Requirement
The code has been tested running under Python 3.6. The required packages are as follows:
* tensorflow == 2.8.0+
* numpy == 1.23.0+
* scipy == 1.8.0+
* pandas == 1.5.0+

## Dataset
* Download: [Google Drive](https://drive.google.com/drive/folders/1ny_jqRE_NwK3SbnxF4W3Ql_SiItKLKlC?usp=sharing)

## Example to Run the Codes
```
python main.py --dataset beauty --lr 1e-3 --l2_reg 1e-4 --max_len 50 --dropout_rate 0.5 --L 3 --alpha 0.5 --lambda_kd 0.1 --tau 0.1
```

