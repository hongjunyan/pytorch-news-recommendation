# PyTorch implementation for news recommendation algorithms

For practice, I tried to implement the news recommendation algorithms in the [recommenders] with PyTorch


## Implemented Algorithms
- [NPA] (Wu et al., 2019, SIGKDD) [1]
- [NRMS] (Wu et al., 2019, IJCNLP) [2]
- [Fastformer] (Wu et al., 2021, arxiv)  TODO


## Dataset
Take it easy, do not download the dataset first, We will automatically download the dataset in `train_model.py`
- [MIND]: the description about MIND dataset can refer to [this document](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)


## Environment
- OS: windows 10 / Ubuntu 18.04
- Python == 3.7.13
- PyTorch == 1.8.2

Please follow the commands bellow:
1. Install the long-term supported PyTorch version:
```commandline
# cpu version
$> pip install torch==1.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
# gpu version
$> pip install torch==1.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102
```
2. Install the remaining dependencies
```commandline
$> pip install -r requirements.txt
```

## How to run the code
1. Try a toy example, train_model.py will download "demo" dataset of MIND and train a simple model which only trained 1 epoch:
```commandline
# show parameter hints
$> python train_model.py --help

# cpu version
$> python train_model.py --mind_type demo --model_type nrms --epochs 1

# gpu version
$> python train_model.py --mind_type demo --model_type nrms --epochs 1 --device 0
```


2. Train a formal model with MIND-large dataset, and the performance in the next section:
```commandline
# cpu version
$> python train_model.py --mind_type large --model_type nrms --epochs 10 --title_size 30 --batch_size 128

# gpu version
$> python train_model.py --mind_type large --model_type nrms --epochs 10 --title_size 30 --batch_size 128 --device 0
```
## Results
| Algorithms \ Metrics | AUC        | MRR        | NDCG@5     | NDCG@10    |
|----------------------|------------|------------|------------|------------|
| NPA                  | 0.6655     | 0.3171     | 0.4143     | 0.3504     |
| NRMS                 | **0.6822** | **0.3307** | **0.3655** | **0.4311** |

## Related projects
- [recommenders]: best practices for building recommendation systems

## Reference papers
- [1] Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. 2019b. Npa: Neural news recommendation with personalized attention. In KDD, pages 2576–2584. ACM.
- [2] Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, and Xing Xie. 2019c. Neural news recommendation with multi-head selfattention. In EMNLP-IJCNLP, pages 6390–6395.

[recommenders]: https://github.com/microsoft/recommenders/tree/b704c420ee20b67a9d756ddbfdf5c9afd04b576b
[NPA]: https://arxiv.org/pdf/1907.05559.pdf
[NRMS]: https://aclanthology.org/D19-1671.pdf
[MIND]: https://msnews.github.io/