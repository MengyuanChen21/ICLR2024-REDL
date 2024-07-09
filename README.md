## [ICLR 2024 Spotlight] R-EDL: Relaxing Nonessential Settings of Evidential Deep Learning

> **Authors**: Mengyuan Chen, Junyu Gao, Changsheng Xu.

> **Affiliations**: Institute of Automation, Chinese Academy of Sciences

### Correction:
There is a mistake below Eq.(9). The uncertainty mass expression in R-EDL should be $u_X=\lambda C/S_X$, where the $\lambda$ was missing.

### Dependencies:
Here we list our used requirements and dependencies.
 - GPU: GeForce RTX 3090
 - Python: 3.8.5
 - PyTorch: 1.12.0
 - Numpy: 1.21.2
 - Pandas: 1.1.3
 - Scipy: 1.3.1
 - Scikit-learn: 1.0.1
 - Wandb: 0.12.6
 - Tqdm: 4.62.3

### Data preparation:
The required datasets of the classical setting (MNIST/FMNIST/KMNIST/CIFAR-10/CIFAR-100/SVHN) will be automatically downloaded if your server has an Internet connection.

The required datasets of the few-shot setting (mini-ImageNet/CUB) can be downloaded from [Google Drive](https://drive.google.com/file/d/1yJC4xABAPkidHYEddZ6ncUgKaRIxNy6L/view?usp=sharing). Please unzip the file and place its contents ("features.md5" and "WideResNet28_10_S2M2_R") directly into the "code_fsl/features/" directory.

### Pre-trained models:
The pre-trained models of R-EDL can be downloaded from [Google Disk](https://drive.google.com/file/d/1e1qZBAJQlsHBbl3tjfIuouMqHd9UpbkK/view?usp=sharing).
They need to be unzipped and put in the directory './code_classical/saved_models/'.

### Quick start for experiments of classical setting:
To test pre-trained models, run:
   ```
   python main.py --configid "1_mnist/mnist-redl-test" --suffix test
   python main.py --configid "2_cifar10/cifar10-redl-test" --suffix test
   ```
   
To train from scratch, run:
   ```
   python main.py --configid "1_mnist/mnist-redl-train" --suffix test
   python main.py --configid "2_cifar10/cifar10-redl-train" --suffix test
   ```

### Quick start for experiments of few-shot setting:
Given that this setting involves conducting experiments across 10,000 few-shot episodes, providing pre-trained models for testing becomes nearly impossible.

To train from scratch, run:
  ```
  python main.py --configid "1_mini/5w1s-redl" --suffix test
  python main.py --configid "1_mini/5w5s-redl" --suffix test
  python main.py --configid "1_mini/5w20s-redl" --suffix test
  python main.py --configid "1_mini/10w1s-redl" --suffix test
  python main.py --configid "1_mini/10w5s-redl" --suffix test
  python main.py --configid "1_mini/10w20s-redl" --suffix test
  ```

## Citation
If you find the code useful in your research, please cite:
```
@inproceedings{chen2023r,
  title={R-EDL: Relaxing Nonessential Settings of Evidential Deep Learning},
  author={Chen, Mengyuan and Gao, Junyu and Xu, Changsheng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
  
### Acknowledgement
This project is built upon the repository of [IEDL](https://github.com/danruod/IEDL), [Posterior Network](https://github.com/sharpenb/Posterior-Network), and [Firth Bias Reduction in Few-shot Distribution Calibration](https://github.com/ehsansaleh/code_dcf). We would like to thank their authors for their excellent work. If you want to use and redistribe our code, please follow [this license](./LICENSE) as well.

### Contact
Feel free to contact me (Mengyuan Chen: [chenmengyuan2021@ia.ac.cn](mailto:chenmengyuan2021@ia.ac.cn)) if anything is unclear or you are interested in potential collaboration.
