### We are updating..
### News - We realse the tool based on Soot to extract the graphs from Java bytecode.
### All datasets are comming (uploading to Zenodo )

- Extract Graph tool
  - We implement the tool based on Soot. Soot Version is in `pom.xml`.
  - You can freely modify it.
  - You can use it to exctrat graph for your data.
  - compile `mvn clean package`
  - run command see the example, `extractGraphs.sh`
- Requriments/Environment
  - recommend that you use `conda`
  - see  requirements.txt
- Download Dataset from Zenodo
  - put all downloaded file to `source` folder
  - make sure you have enough disk space, around 100G, `bash untar.sh`
  
- Pretrained Model
  1. https://drive.google.com/file/d/1PGF6e56CQ4XAfZMEU2Jl3w1TW4Q-59MG/view?usp=sharing
  2. put it into `source` folder and decompress

- Run experiments
  1. `cd source/graphscripts_experiments/`
  2. run bash job files
  3. reminding, it contains all trial experiment jobs. We will clean it later and keep only experiments in the paper.

- paper link, [GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence Analyses](https://arxiv.org/abs/2112.01218)

