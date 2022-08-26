# PyTorch implementation of NPA (Neural News Recommendation with Personalized Attention)

NPA paper: https://arxiv.org/pdf/1907.05559.pdf

Since the [original implementation](https://github.com/wuch15/KDD-NPA) of NPA uses Keras, and the [implementation](https://github.com/microsoft/recommenders/blob/main/recommenders/models/newsrec/models/npa.py) of Microsoft recommenders was also Keras. According to the results of my own survey, no one used pytorch to implement NPA at present.
So I want to share my implementation.

## Dataset
- [MIND](https://msnews.github.io/): the description about MIND dataset can refer to [this document](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)

## Environment
- Python == 3.7.13
- PyTorch == 1.8.1
- Tensorflow == 2.6.1  # Only used for "recommenders", we will use "recommenders" package to download data
- recommenders==1.1.1

## How to run the code
1. Try a toy example, train_npa.py will download "demo" dataset of MIND and train a simple model which only trained 1 epoch:
```commandline
# cpu version
$> python train_npa.py --mind_type demo --epochs 1

# gpu version
$> python train_npa.py --mind_type demo --epochs 1 --device 0

------------ You will see the message below------------------
Model use gpu 0
Total Parameters: 14869600
Number of Users: 50000, Number of vocab: 31027
Evaluating before training......
evaluating: 8874it [02:21, 62.51it/s]
Evaluated result before training: {'group_auc': 0.4971, 'mean_mrr': 0.2266, 'ndcg@5': 0.2243, 'ndcg@10': 0.2941}
Start Training ......
training: 1086it [00:57, 18.84it/s]
evaluating: 8874it [02:22, 62.47it/s]
at epoch 1
train info: negative log likelihood loss:1.5020435792745586
eval info: group_auc:0.5825, mean_mrr:0.2573, ndcg@10:0.3446, ndcg@5:0.2791
at epoch 1 , train time: 57.7s eval time: 147.9s
Finish Training
```


2. Train a formal version model, the hyperparameters are the same as [NPA paper](https://arxiv.org/pdf/1907.05559.pdf):
```commandline
# cpu version
$> python train_npa.py --mind_type large --epochs 15 --title_size 30

# gpu version
$> python train_npa.py --mind_type large --epochs 15 --title_size 30 --device 0
```