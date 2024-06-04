# DICE
This is the official implementation of our WWW'21 paper:  

Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Depeng Jin, Yong Li, **Disentangling User Interest and Conformity for Recommendation with Causal Embedding**, In Proceedings of the Web Conference 2021.

## Model training
First unzip the datasets and start the visdom server:
```
visdom -port 33336
```

Then simply run the following command to reproduce the experiments on corresponding dataset and model:
```
python app.py --flagfile ./config/xxx.cfg
```

## Embedding visualization
The visualization codes to reproduce Figure 5(b) and Figure 7 can be found in the `viz` folder.

First, reduce the dimension of the embedding vectors to 2D using t-SNE (remember to change the path to the model checkpoint in `viz.py`):
```python
python viz.py
```

Then, visualize the 2D embedding vectors using MATLAB:
```matlab
embedding_viz.m
```

## Dataset processing
The dataset process codes are in this [repo](https://github.com/DavyMorgan/dps).
Please check this [issue](https://github.com/tsinghua-fib-lab/DICE/issues/1#issuecomment-903234948) for more details.

## Citation
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
