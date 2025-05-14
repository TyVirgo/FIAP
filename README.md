# HIGAN [154K parameter]

This is an official implementation of the paper “A high-frequency information guiding attention network for super-lightweight image super-resolution”.
We will continue to improve the code later.

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
# Model framework 
1. The network model implementation code (6 blocks and 10 blocks) is located in directory `./model`
2. Network model diagram：
![image](https://github.com/user-attachments/assets/08431b98-8649-412c-884c-9c405bdd96c1)

# Codes 
- This codes provide the pretrained model.
- Network training and testing code will be uploaded in the future.
  
## How to Test（Upload test files in the future）
1. Download the five test datasets (Set5, Set14, B100, Urban100, Manga109) from [Google Drive](https://drive.google.com/drive/folders/1lsoyAjsUEyp7gm1t6vZI9j7jr9YzKzcF?usp=sharing)

2. The pretrained model for ×2 task (HIGAN-S, HIGAN, and HIGAN-L) have be placed in `./pretrained/` folder. 

3. The testing commands are placed in the './src/demo.sh' file. 

4. The output results will be sorted in `./experiment/test/`.

## SR images visualization
1. We provided visualization of SR images for two versions of the model ( HIGAN and HIGAN-L) from [Google Drive](https://drive.google.com/drive/folders/1xiPOE22AExEcIe5-er3clOYFHCVCJo6F?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1vEOJaLGScgRGaIOeFI3q8w) (code：s8eg)
2. Visual comparison of reconstructions in the paper（HIGAN）:
![image](https://github.com/user-attachments/assets/2596a1ff-2813-4a63-a271-41dfade61593)


## Contact
Email: fjs1867@mnnu.edu.cn


If you find our work is useful, please kindly cite it.
```
Upload in the future
```

## License
This project is released under the Apache 2.0 license.


## Acknowledgements
This code is built on [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch). Thanks for the awesome work.

