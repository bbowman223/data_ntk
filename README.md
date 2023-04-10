This repository contains the code for the paper ["Characterizing the spectrum of the NTK via a power series expansion"](https://arxiv.org/abs/2211.07844) (ICLR 2023).  If you use this code in a paper, please cite this paper using the bibtex reference below
```
@inproceedings{murray2023characterizing,
    title={Characterizing the spectrum of the {NTK} via a power series expansion},
    author={Michael Murray and Hui Jin and Benjamin Bowman and Guido Montufar},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=Tvms8xrZHyR}
}
```


This project is organized into two directories.
* **spec/**
    * Computes the NTK spectrum using Pytorch
* **powerseries/**
    * Computes the NTK power series 

The directories **spec/** and **powerseries/** both have their own **README** files to explain
the project organization

The dependencies for the project are specified in the **environment.yml** file.
If you use [Anaconda](https://www.anaconda.com/) you can construct the environment from this file.
