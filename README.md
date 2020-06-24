# RayS
"RayS: A Ray Searching Method for Hard-label Adversarial Attack"\
*Jinghui Chen*, *Quanquan Gu*\
[https://arxiv.org/abs/2006.12792](https://arxiv.org/abs/2006.12792)

This repository contains our pytorch implementation of RayS: A Ray Searching Method for Hard-label Adversarial Attack in the paper [RayS: A Ray Searching Method for Hard-label Adversarial Attack](https://arxiv.org/abs/2006.12792) (accepted by KDD 2020). 

## Prerequisites: 
* PyTorch
* Numpy
* CUDA

## Usage Examples:
* Run attacks on naturally trained model (Inception):
```bash
  -  python3 attack_natural.py --dataset inception --alg rays --norm linf --targeted 0 --num 50 --epsilon 0.05
```
* Run attacks on naturally trained model (Cifar):
```bash
  - python3 attack_natural.py --dataset cifar --alg rays --norm linf --targeted 0 --num 50 --epsilon 0.031
```
* Run attacks on robust model:
```bash
  -  python3 attack_robust.py --dataset rob_cifar_trades --alg rays --norm linf --targeted 0 --num 50 --epsilon 0.031
```
 