# SLUDA
UDA person re-ID use soft clustering label

![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch 1.1](https://img.shields.io/badge/pytorch-1.1-yellow.svg)


```shell
git clone https://github.com/knifefield/SLUDA.git
cd SLUDA
pip install -r requestments.txt
```

## Prepare Datasets

```shell
cd examples && mkdir data
```
Download the raw datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [MSMT17](https://arxiv.org/abs/1711.08565),
and then unzip them under the directory like
```
SLUDA/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```

#### Stage I: Pre-training on the source domain

```shell
sh scripts/pretrain.sh dukemtmc market1501 myresnet 1
sh scripts/pretrain.sh dukemtmc market1501 myresnet 2
```

#### Stage II: End-to-end training with MMT-500 
We utilized DBSCAN clustering algorithm in the paper.

```shell
sh scripts/train_mmt_dbscan.sh dukemtmc market1501 myresnet
```
