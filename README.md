This repo is for [GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence Analyses](https://arxiv.org/abs/2112.01218), Boost Software License 1.0

```bibtex
@inproceedings{10.1145/3524842.3528456,
author = {Ma, Wei and Zhao, Mengjie and Soremekun, Ezekiel and Hu, Qiang and Zhang, Jie M. and Papadakis, Mike and Cordy, Maxime and Xie, Xiaofei and Traon, Yves Le},
title = {GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence Analyses},
year = {2022},
isbn = {9781450393034},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3524842.3528456},
doi = {10.1145/3524842.3528456},
abstract = {Code embedding is a keystone in the application of machine learning on several Software Engineering (SE) tasks. To effectively support a plethora of SE tasks, the embedding needs to capture program syntax and semantics in a way that is generic. To this end, we propose the first self-supervised pre-training approach (called GraphCode2Vec) which produces task-agnostic embedding of lexical and program dependence features. GraphCode2Vec achieves this via a synergistic combination of code analysis and Graph Neural Networks. GraphCode2Vec is generic, it allows pre-training, and it is applicable to several SE downstream tasks. We evaluate the effectiveness of GraphCode2Vec on four (4) tasks (method name prediction, solution classification, mutation testing and overfitted patch classification), and compare it with four (4) similarly generic code embedding baselines (Code2Seq, Code2Vec, CodeBERT, Graph-CodeBERT) and seven (7) task-specific, learning-based methods. In particular, GraphCode2Vec is more effective than both generic and task-specific learning-based baselines. It is also complementary and comparable to GraphCodeBERT (a larger and more complex model). We also demonstrate through a probing and ablation study that GraphCode2Vec learns lexical and program dependence features and that self-supervised pre-training improves effectiveness.},
booktitle = {Proceedings of the 19th International Conference on Mining Software Repositories},
pages = {524–536},
numpages = {13},
keywords = {code embedding, code representation, code analysis},
location = {Pittsburgh, Pennsylvania},
series = {MSR '22}
}
```

- Extract Graph tool
  - We implement the tool based on Soot. Soot Version is in `pom.xml`.
  - compile `mvn clean package`
  - run command see the example, `extractGraphs.sh`
- Requirements/Environment
  - recommend that you use `conda`
  - see  requirements.txt
- Download Dataset from [Zenodo](https://doi.org/10.5281/zenodo.6394383)
  - put all downloaded files to the `source` folder
  - make sure you have enough disk space, around 100G, `bash untar.sh`
  
- Pretrained Model
  1. https://drive.google.com/file/d/1PGF6e56CQ4XAfZMEU2Jl3w1TW4Q-59MG/view?usp=sharing
  2. put it into the `source` folder and decompress

- Run experiments
  1. `cd source/graphscripts_experiments/`
  2. run bash job files
  3. reminding, it contains all trial experiment jobs. We will clean it later and keep only experiments on the paper.

- We apply our approach to predict the flaky tests, which improves 1% performance compared with [Flakify: A Black-Box, Language Model-based Predictor for Flaky Tests](https://arxiv.org/abs/2112.12331#:~:text=23%20Dec%202021%5D-,Flakify%3A%20A%20Black%2DBox%2C%20Language%20Model%2D,based%20Predictor%20for%20Flaky%20Tests&text=Software%20testing%20assures%20that%20code,version%20of%20the%20source%20code.). Please see the repository, https://github.com/Marvinmw/Flaky .
  1. We compile all the code.
  2. We filter that test cases do not have .class files or failed to extract graphs, which results in us losing some flaky tests.
  3. To compare, we rerun Flakify and make sure we use the same dataset.
  4. You can find all info at https://github.com/Marvinmw/Flaky .
  
- paper link, [GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence Analyses](https://arxiv.org/abs/2112.01218)


