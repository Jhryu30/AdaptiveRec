#  AdaptiveRec & Fisher-weighted Merge of Sequential RecSys

This repository contains two main codes; **AdaptiveRec** & **Fisher-weighted Merge of Sequential Recommendation System**. Both papers are accepted in 'The Many Facets of Preference-based Learning' the Workshop at the ICML 2023.  

- [AdaptiveRec: Adaptively Construct Pairs for Contrastive Learning in Sequential Recommendation](https://arxiv.org/abs/2307.05469).

- [Fisher-Weighted Merge of Contrastive Learning Models in Sequential Recommendation](https://arxiv.org/abs/2307.05476).

# Dataset Preparation

Download datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.

```
$ tree
.
├── Amazon_Beauty
│   ├── Amazon_Beauty.inter
│   └── Amazon_Beauty.item
├── Amazon_Clothing_Shoes_and_Jewelry
│   ├── Amazon_Clothing_Shoes_and_Jewelry.inter
│   └── Amazon_Clothing_Shoes_and_Jewelry.item
├── Amazon_Sports_and_Outdoors
│   ├── Amazon_Sports_and_Outdoors.inter
│   └── Amazon_Sports_and_Outdoors.item
├── ml-1m
│   ├── ml-1m.inter
│   ├── ml-1m.item
│   ├── ml-1m.user
│   └── README.md
└── yelp
    ├── README.md
    ├── yelp.inter
    ├── yelp.item
    └── yelp.user

```


# Reproduce AdaptiveRec

If you want to reproduce AdaptiveRec and the previous works such as DuoRec and CL4SRec, please run `reproduce.sh`.

For more information of previous work, 
- [Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation](https://arxiv.org/abs/2110.05730) (accepted in WSDM 2022)
- [Contrastive Learning for Sequential Recommendation](https://arxiv.org/abs/2010.14395) (accepted in WWW 2022)

# Reproduce Fisher-weighted Merge of Sequential RecSys

In order to ensemble models, obtain learned parameters through reproducing previous models such as DuoRec, CL4SRec, and AdaptiveRec.
Run `run_model_soup.py`.


# Wandb

If you want to use wandb, writedown your wandb accounts at seq.yaml file.
```
entity: your_entity
project: ~
name: ~
```

# Credit
This repo is based on [RecBole](https://github.com/RUCAIBox/RecBole), [DuoRec](https://github.com/RuihongQiu/DuoRec) and [ModelSoup](https://github.com/mlfoundations/model-soups).
