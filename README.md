# FIAP [154K parameter]

This is an official implementation of the paper “An Efficient Frequency-Aware Information Asymmetric Processing Network for Super-Lightweight Image Super-Resolution”

## Dependencies

- The cuda version used in the project is 12.2
- Python = 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
```
conda create -n your_env_name python=3.8
conda activate your_env_name
```
- Other packages required by the project are in the file ‘requirements.txt’
```
pip install -r requirements.txt
```

# Codes 
- This codes provide the testing, training, and SR image visualization code.
  
## How to Test
1. Download the five test datasets (Set5, Set14, B100, Urban100, Manga109) from [Google Drive](https://drive.google.com/drive/folders/1lsoyAjsUEyp7gm1t6vZI9j7jr9YzKzcF?usp=sharing) 

2. Three versions of pretrained models (FIAP-S, FIAP, and FIAP-L) have be placed in `./pretrained/` folder. 

3. The testing commands are placed in the './src/demo.sh' file. 
Close comments in 'demo.sh' and run 'demo.sh' to execute the corresponding command of testing. Such as:
```
python main.py --model FIAP_6BLOCK --save ./test/FIAP-S_Div2k_tiny_x2 --scale 2 --n_feats 32 --pre_train /Your_Path/FIAP/pretrained/FIAP-S_Div2k/fiaps_x2.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
```
If you need to change the values of other properties and see what they mean, check out './src/option.py' file.

More testing commonds can be found in `./src/demo.sh` file and the output results will be sorted in `./experiment/test/`.

## How to Train

1. 下载 [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) from [Google Drive](https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA)

2. Modify the training dataset path attributes '--dir_data' and '--data_train' in the `./src/option.py` file.

3. The training commands are placed in the './src/demo.sh' file.
Close comments in 'demo.sh' and run 'demo.sh' to execute the corresponding command of training. Such as:
```
python main.py --model FIAP_6BLOCK --save ./train/FIAP-S_Div2k_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
```
4. More training commond can be found in `./src/demo.sh` file, and the training results will be sorted in `./experiment/train/`.

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

