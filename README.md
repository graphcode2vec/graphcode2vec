### We are updating..
### News - We realse the tool based on Soot to extract the graphs from Java bytecode.
### All datasets are comming (uploading to Zenodo )

- Extract Graph tool
  - compile `mvn clean package`
  - run command see the example, `extractGraphs.sh`
- Requriments
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
  3. reminding: it contains all trial experiment jobs. We will clean it later and keep only experiments in the paper.

- paper link, [GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence Analyses](https://arxiv.org/abs/2112.01218)
- cite `@article{DBLP:journals/corr/abs-2112-01218,
  author    = {Wei Ma and
               Mengjie Zhao and
               Ezekiel Soremekun and
               Qiang Hu and
               Jie Zhang and
               Mike Papadakis and
               Maxime Cordy and
               Xiaofei Xie and
               Yves Le Traon},
  title     = {GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence
               Analyses},
  journal   = {CoRR},
  volume    = {abs/2112.01218},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.01218},
  eprinttype = {arXiv},
  eprint    = {2112.01218},
  timestamp = {Tue, 07 Dec 2021 12:15:54 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-01218.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}`

