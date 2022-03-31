#<p align="center">
#  <img src="GraphExtractor/graphcode2vec.svg" width="400" height="300"/>
#</p>
#

### We are updating..
#### News - We apply our approch on Flaky Test Prediction, which gets improvement 1% performance compared with [Flakify: A Black-Box, Language Model-based Predictor for Flaky Tests](https://arxiv.org/abs/2112.12331#:~:text=23%20Dec%202021%5D-,Flakify%3A%20A%20Black%2DBox%2C%20Language%20Model%2D,based%20Predictor%20for%20Flaky%20Tests&text=Software%20testing%20assures%20that%20code,version%20of%20the%20source%20code.). You can find the details below.
#### News - We realse the tool based on Soot to extract the graphs from Java bytecode.
#### Reminding that GraphCode2Vec works with Java Bytecode so that all the handled source code should be compiled.
- Extract Graph tool
  - We implement the tool based on Soot. Soot Version is in `pom.xml`.
  - You can freely modify it.
  - You can use it to exctrat graph for your data.
  - compile `mvn clean package`
  - run command see the example, `extractGraphs.sh`
- Requriments/Environment
  - recommend that you use `conda`
  - see  requirements.txt
- Download Dataset from [Zenodo](https://doi.org/10.5281/zenodo.6394383)
  - put all downloaded files to `source` folder
  - make sure you have enough disk space, around 100G, `bash untar.sh`
  
- Pretrained Model
  1. https://drive.google.com/file/d/1PGF6e56CQ4XAfZMEU2Jl3w1TW4Q-59MG/view?usp=sharing
  2. put it into `source` folder and decompress

- Run experiments
  1. `cd source/graphscripts_experiments/`
  2. run bash job files
  3. reminding, it contains all trial experiment jobs. We will clean it later and keep only experiments in the paper.

- We apply our approch to predict the falky tests, which gets improvement 1% performance compared with Flakify: A Black-Box, Language Model-based Predictor for Flaky Tests. Please see the repository, https://github.com/Marvinmw/Flaky .
  1. We compile all the code.
  2. We filter that test cases do not have .class files or failed to extract graphs, which results in we lose some flaky tests.
  3. To compare, we rerun Flakify and make sure we use the same dataset.
  4. You can find all info in https://github.com/Marvinmw/Flaky .
  
- paper link, [GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence Analyses](https://arxiv.org/abs/2112.01218)
- License, Boost Software License 1.0

