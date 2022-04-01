This repo is for [GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence Analyses](https://arxiv.org/abs/2112.01218), Boost Software License 1.0

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


