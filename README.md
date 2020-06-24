# RayS: A Ray Searching Method for Hard-label Adversarial Attack
"RayS: A Ray Searching Method for Hard-label Adversarial Attack"\
*Jinghui Chen*, *Quanquan Gu*\
[https://arxiv.org/abs/2006.12792](https://arxiv.org/abs/2006.12792)

This repository contains our pytorch implementation of RayS: A Ray Searching Method for Hard-label Adversarial Attack in the paper [RayS: A Ray Searching Method for Hard-label Adversarial Attack](https://arxiv.org/abs/2006.12792) (accepted by KDD 2020). 

# What is RayS
RayS is a hard-label adversarial attack which only requires model's hard label output (prediction label). 

**It is gradient-free, hyper-parameter free and is also independent to adversarial losses such as CrossEntropy or C&W.**

Therefore, RayS can be used as a good sanity check for possible "falsely robust" models (models that may overfit to certain types of gradient-based attacks and adversarial losses).

**RayS also proposed a new model robustness metric: ADBD (average decision boundary distance), which reflects examples' average distance to their closest decision boundary, which is independent to the perturbation strength `epsilon`.**

# Model Robustness Evaluation

We have tested widely used CIFAR-10 benchmark with the maximum L_inf norm pertubation strength  `epsilon=0.031` (8/255)

**Note**: Ranking is the based the ADBD metric. * denotes models using extra data for training.

|#    |paper       |natural          |robust  (report) |robust  (RayS) |ADBD|
|:---:|:---:|---:|---:|---:|---:|
|**1**| [RST <br>(Carmon et al., 2019)](https://arxiv.org/abs/1905.13736)*|  89.7| 62.5| 64.6| 0.046|
|**2**| [TRADES <br>(Zhang et al., 2019b)](https://arxiv.org/abs/1901.08573)| 85.4| 56.4| 57.3| 0.040| 
|**3**| [Adversarial Training <br>(Madry et al., 2018)](https://arxiv.org/abs/1706.06083)| 87.1| 47.0| -| 0.038| 
|**4**| [Feature Scattering<br> (Zhang & Wang, 2019)](http://papers.nips.cc/paper/8459-defense-against-adversarial-attacks-using-feature-scattering-based-adversarial-training)|  91.3| 60.6| 44.5| 0.030|
|**5**| [Adversarial Interpolation Training<br> (Zhang & Xu, 2020)](https://openreview.net/forum?id=Syejj0NYvr&noteId=Syejj0NYvr) | 91.0| 68.7| 46.9| 0.031|
|**6**| [Sensible Adversarial Training <br>(Kim & Wang, 2020)](https://openreview.net/forum?id=rJlf_RVKwr)| 91.9| 57.2| 43.9| 0.029| 
 
Please contact us if you want to add your model into the leaderboard.

## How to use RayS to evaluate your model robostness:

### Prerequisites: 
* Python
* Numpy
* CUDA

### PyTorch models
Import RayS attack by 

```python
from general_torch_model import GeneralTorchModel
torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)

from RayS import RayS
attack = RayS(torch_model, epsilon=args.epsilon)
```

where:
+ `torch_model` is the PyTorch model under GeneralTorchModel warpper; For models using transformed images (exceed the range of [0,1]), simply set `im_mean=[0.5, 0.5, 0.5]` and `im_std=[0.5, 0.5, 0.5]` for instance,
+ `epsilon` is the maximum adversarial perturbation strength.

To actally run RayS attack, use

```python
x_adv, queries, adbd, succ = attack(data, label, query_limit)
```

it returns:
+ `x_adv`: the adversarial examples found by RayS,
+ `queries`: the number of queries used for finding the adversarial examples,
+ `adbd`: the average decision boundary distance for each example,
+ `succ`: indicate whether each example being successfully attacked.


* Sample usage on attacking robust model:
```bash
  -  python3 attack_robust.py --dataset rob_cifar_trades --query 10000 --batch 1000  --epsilon 0.031
```

### TensorFlow models
To evaluate TensorFlow models with RayS attack:

```python
from general_tf_model import GeneralTFModel 
tf_model = GeneralTFModel(model.logits, model.x_input, sess, n_class=10, im_mean=None, im_std=None)

from RayS import RayS
attack = RayS(tf_model, epsilon=args.epsilon)
```

where:
+ `model.logits`: logits tensor return by the Tensorflow model,
+ `model.x_input`: placeholder for model input (NHWC format),
+ `sess`: TF session .

The rest part is the same us evaluating PyTorch models.

## Reproduce experiments in the paper:
* Run attacks on naturally trained model (Inception):
```bash
  -  python3 attack_natural.py --dataset inception --epsilon 0.05
```
* Run attacks on naturally trained model (Resnet):
```bash
  -  python3 attack_natural.py --dataset resnet --epsilon 0.05
```
* Run attacks on naturally trained model (Cifar):
```bash
  - python3 attack_natural.py --dataset cifar --epsilon 0.031
```
* Run attacks on naturally trained model (MNIST):
```bash
  - python3 attack_natural.py --dataset mnist --epsilon 0.3
```

