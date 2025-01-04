# Training CAMP (Cartesian Atomic Moment Potential)

This repo contains example training scripts for the [CAMP](https://github.com/wengroup/camp) model, the trained models, and scripts to run molecular dynamics simulations using the trained models.

## Installation
To use the scripts in this repo, you will first need to install the CAMP model. Please follow the instructions in the main [CAMP](https://github.com/wengroup/camp) repo.


## Folder and files
- [dataset](./dataset): the bilayer graphene and water datasets. For the LiPS and md17 datasets, please refer to the paper (below) for the download link.
- [models](./models): training scripts and the trained models.
- [md](./md): scripts to run molecular dynamics (MD) simulations in ASE using the trained models.


## Citation

Wen, M., Huang, W. F., Dai, J., & Adhikari, S. (2024). Cartesian Atomic Moment Machine Learning Interatomic Potentials. arXiv preprint arXiv:2411.12096.

```latex
@article{wen2024cartesian,
  title   = {Cartesian Atomic Moment Machine Learning Interatomic Potentials},
  author  = {Wen, Mingjian and Huang, Wei-Fan and Dai, Jin and Adhikari, Santosh},
  journal = {arXiv preprint arXiv:2411.12096},
  doi     = {10.48550/ARXIV.2411.12096},
  year    = {2024}
}
```
