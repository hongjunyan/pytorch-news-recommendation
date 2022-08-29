# PyTorch implementation of NPA (Neural News Recommendation with Personalized Attention)

NPA paper: https://arxiv.org/pdf/1907.05559.pdf

Since the [original implementation](https://github.com/wuch15/KDD-NPA) of NPA used Keras, and the [implementation](https://github.com/microsoft/recommenders/blob/main/recommenders/models/newsrec/models/npa.py) of Microsoft recommenders was also Keras. So I want to share my PyTorch implementation.

## Dataset
- [MIND](https://msnews.github.io/): the description about MIND dataset can refer to [this document](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)

## Environment
- Python == 3.7.13
- PyTorch == 1.8.1
- Tensorflow == 2.6.1  # Only used for "recommenders", we will use "recommenders" package to download data
- recommenders==1.1.1

You can easily install dependencies by the following command 
```commandline
$> pip install -r requirements.txt
```

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
----------- Training Result t-------------------
Model use gpu 0
Total Parameters: 93975900
Number of Users: 711222, Number of vocab: 74307
Evaluating before training......
evaluating: 55022it [52:21, 17.51it/s]
Evaluated result before training: {'group_auc': 0.5275, 'mean_mrr': 0.2289, 'ndcg@5': 0.2387, 'ndcg@10': 0.3035}
Start Training ......
training: 13218it [49:19,  4.47it/s]
evaluating: 55022it [53:28, 17.15it/s]
at epoch 1
train info: negative log likelihood loss:1.2682446995703567
eval info: group_auc:0.6533, mean_mrr:0.3108, ndcg@10:0.4061, ndcg@5:0.339
at epoch 1 , train time: 2959.6s eval time: 3462.5s
training: 13218it [47:54,  4.60it/s]
evaluating: 55022it [53:29, 17.14it/s]
at epoch 2
train info: negative log likelihood loss:1.1675646897929035
eval info: group_auc:0.6571, mean_mrr:0.3134, ndcg@10:0.4097, ndcg@5:0.3431
at epoch 2 , train time: 2874.8s eval time: 3464.6s
training: 13218it [47:52,  4.60it/s]
evaluating: 55022it [53:30, 17.14it/s]
at epoch 3
train info: negative log likelihood loss:0.9504662779342895
eval info: group_auc:0.6485, mean_mrr:0.304, ndcg@10:0.3992, ndcg@5:0.333
at epoch 3 , train time: 2873.0s eval time: 3469.3s
training: 13218it [47:52,  4.60it/s]
evaluating: 55022it [53:28, 17.15it/s]
at epoch 4
train info: negative log likelihood loss:0.753869247551061
eval info: group_auc:0.635, mean_mrr:0.2919, ndcg@10:0.3854, ndcg@5:0.3178
at epoch 4 , train time: 2872.6s eval time: 3463.3s
training: 13218it [47:56,  4.60it/s]
evaluating: 55022it [53:24, 17.17it/s]
at epoch 5
train info: negative log likelihood loss:0.622324419219567
eval info: group_auc:0.6276, mean_mrr:0.2898, ndcg@10:0.3822, ndcg@5:0.3154
at epoch 5 , train time: 2876.0s eval time: 3461.6s
training: 13218it [47:49,  4.61it/s]
evaluating: 55022it [53:30, 17.14it/s]
at epoch 6
train info: negative log likelihood loss:0.5324889668079598
eval info: group_auc:0.6217, mean_mrr:0.2874, ndcg@10:0.3784, ndcg@5:0.312
at epoch 6 , train time: 2869.1s eval time: 3464.2s
training: 13218it [47:53,  4.60it/s]
evaluating: 55022it [53:30, 17.14it/s]
at epoch 7
train info: negative log likelihood loss:0.4700697038752767
eval info: group_auc:0.6143, mean_mrr:0.2831, ndcg@10:0.373, ndcg@5:0.3064
at epoch 7 , train time: 2873.4s eval time: 3463.5s
training: 13218it [47:50,  4.60it/s]
evaluating: 55022it [53:32, 17.13it/s]
at epoch 8
train info: negative log likelihood loss:0.42472870327066703
eval info: group_auc:0.616, mean_mrr:0.2852, ndcg@10:0.3755, ndcg@5:0.3094
at epoch 8 , train time: 2870.5s eval time: 3469.4s
training: 13218it [47:49,  4.61it/s]
evaluating: 55022it [53:30, 17.14it/s]
at epoch 9
train info: negative log likelihood loss:0.3897468598489322
eval info: group_auc:0.6167, mean_mrr:0.2853, ndcg@10:0.3755, ndcg@5:0.3101
at epoch 9 , train time: 2869.9s eval time: 3463.7s
training: 13218it [47:52,  4.60it/s]
evaluating: 55022it [53:28, 17.15it/s]
at epoch 10
train info: negative log likelihood loss:0.3636658953930881
eval info: group_auc:0.6106, mean_mrr:0.2834, ndcg@10:0.3723, ndcg@5:0.3065
at epoch 10 , train time: 2872.2s eval time: 3466.3s
Finish Training
```