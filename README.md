# GraphDKL

This is a repo hosting the source code of our proposed method GraphDKL for causal effect estimation on graph data. If you find our code helpful, citing our paper is appreciated. 

IEEE ICDM'23 [To Predict or to Reject: Causal Effect Estimation with Uncertainty on Networked Data](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10415750).
```bibtex
@inproceedings{wen2023predict,
  title={To Predict or to Reject: Causal Effect Estimation with Uncertainty on Networked Data},
  author={Wen, Hechuan and Chen, Tong and Chai, Li Kheng and Sadiq, Shazia and Zheng, Kai and Yin, Hongzhi},
  booktitle={2023 IEEE International Conference on Data Mining (ICDM)},
  pages={1415--1420},
  year={2023},
  organization={IEEE}
}
```

## Package

```.sh
$ conda env create -f environment.yml
```

## Example

### Training
Training on the BlogCatalog dataset with imbalance k=0.5 with spectral normalizaton enable.

```.sh
main.sh
```

### Evaluation
Conduct rejection and save the estimation error for the rest at different rejection rate.

```.sh
evaluations.sh
```
