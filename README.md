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

If you use our codes and datasets in your research, please cite:
```
@inproceedings{zheng2021disentangling,
  title={Disentangling User Interest and Conformity for Recommendation with Causal Embedding},
  author={Zheng, Yu and Gao, Chen and Li, Xiang and He, Xiangnan and Li, Yong and Jin, Depeng},
  booktitle={Proceedings of the Web Conference 2021},
  pages={2980--2991},
  year={2021}
}
```
