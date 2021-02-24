# DICE
This is the official implementation of our WWW'21 paper:  

Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Depeng Jin, Yong Li, **Disentangling User Interest and Conformity for Recommendation with Causal Embedding**, In Proceedings of the Web Conference 2021.

***
First unzip the datasets and start the visdom server:
```
visdom -port 33336
```

Then simply run the following command to reproduce the experiments on corresponding dataset and model:
```
python app.py --flagfile ./config/xxx.cfg
```
