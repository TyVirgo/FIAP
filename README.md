# FIAP [154K parameter]

This is an official implementation of the paper “An Efficient Frequency-Aware Information Asymmetric Processing Network for Super-Lightweight Image Super-Resolution”

## Dependencies

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.5.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.

# Codes 
- Our codes version based on [mmsr](https://github.com/open-mmlab/mmsr). 
- This codes provide the testing and training code.


  
## How to Test
1. Download the five test datasets (Set5, Set14, B100, Urban100, Manga109) from [Google Drive](https://drive.google.com/drive/folders/1lsoyAjsUEyp7gm1t6vZI9j7jr9YzKzcF?usp=sharing) 

2. Three versions of pretrained models have be placed in `./pretrained/` folder. 

3. The test commands are placed in the './src/demo.sh' file.Run test. 
```
cd codes
python test.py -opt option/test/test_PANx4.yml
```
More testing commonds can be found in `./codes/run_scripts.sh` file.
5. The output results will be sorted in `./results`. (We have been put our testing log file in `./results`) We also provide our testing results on five benchmark datasets on [Google Drive](https://drive.google.com/drive/folders/1F6unBkp6L1oJb_gOgSHYM5ZZbyLImDPH?usp=sharing).

## How to Train

1. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://github.com/LimBee/NTIRE2017) from [Google Drive](https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA)

2. Generate Training patches. Modified the path of your training datasets in `./codes/data_scripts/extract_subimages.py` file.

3. Run Training.

```
python train.py -opt options/train/train_PANx4.yml
```
4. More training commond can be found in `./codes/run_scripts.sh` file.

## Testing the Parameters, Mult-Adds and Running Time

1. Testing the parameters and Mult-Adds.
```
python test_summary.py
```

2. Testing the Running Time.

```
python test_running_time.py
```

## Related Work on AIM2020
Enhanced Quadratic Video Interpolation (winning solution of AIM2020 VTSR Challenge)
[paper](https://arxiv.org/pdf/2009.04642.pdf) | [code](https://github.com/lyh-18/EQVI)

## Contact
Email: hubylidayuan@gmail.com

If you find our work is useful, please kindly cite it.
```
@inproceedings{zhao2020efficient,
  title={Efficient image super-resolution using pixel attention},
  author={Zhao, Hengyuan and Kong, Xiangtao and He, Jingwen and Qiao, Yu and Dong, Chao},
  booktitle={European Conference on Computer Vision},
  pages={56--72},
  year={2020},
  organization={Springer}
}
```

