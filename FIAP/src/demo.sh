"""Close comments and run 'demo.sh' to execute the corresponding command"""

"""If you need to change the values of other properties and see what they mean, check out 'option.py'.
    For example: train/test dataset path, number of Gpus, and optimization parameters """

###FIAP-S, FIAP, and FIAP-L training commands
#train_FIAP-S_6BLOCK_32C_64*64patch_size
#python main.py --model FIAP_6BLOCK --save FIAP-S_Div2k_tiny_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_6BLOCK --save FIAP-S_Div2k_tiny_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_6BLOCK --save FIAP-S_Div2k_tiny_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

#train_FIAP_10BLOCK_32C_64*64patch_size
#python main.py --model FIAP_10BLOCK --save FIAP_Div2K_tiny_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_10BLOCK --save FIAP_Div2K_tiny_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_10BLOCK --save FIAP_Div2K_tiny_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

#train_FIAP-L_10BLOCK_32C_64*64patch_size
#python main.py --model FIAP_10BLOCK --save FIAP-L_Div2K_x2 --scale 2 --lr 6e-4 --batch_size 48 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_10BLOCK --save FIAP-L_Div2K_x3 --scale 3 --lr 6e-4 --batch_size 48 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_10BLOCK --save FIAP-L_Div2K_x4 --scale 4 --lr 6e-4 --batch_size 48 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

###FIAP-S, FIAP, and FIAP-L testing commands
#test_FIAP-S_6BLOCK_32C_64*64patch_size
#python main.py --model FIAP_6BLOCK --save FIAP-S_Div2k_tiny_x2 --scale 2 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP-S_Div2k_tiny_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_6BLOCK --save FIAP-S_Div2k_tiny_x3 --scale 3 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP-S_Div2k_tiny_x3/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_6BLOCK --save FIAP-S_Div2k_tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP-S_Div2k_tiny_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

#test_FIAP_10BLOCK_32C_64*64patch_size
#python main.py --model FIAP_10BLOCK --save FIAP_Div2k_tiny_x2 --scale 2 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP_Div2k_tiny_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_10BLOCK --save FIAP_Div2k_tiny_x3 --scale 3 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP_Div2k_tiny_x3/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_10BLOCK --save FIAP_Div2k_tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP_Div2k_tiny_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

#test_FIAP-L_10BLOCK_32C_64*64patch_size
#python main.py --model FIAP_10BLOCK --save FIAP-L_Div2k_x2 --scale 2 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP-L_Div2k_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_10BLOCK --save FIAP-L_Div2k_x3 --scale 3 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP-L_Div2k_x3/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_10BLOCK --save FIAP-L_Div2k_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP-L_Div2k_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

###Visualization commands
#python main.py --model FIAP_10BLOCK --save ./A_visual/FIAP_Div2k_tiny_x2 --scale 2 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP_Div2k_tiny_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model FIAP_10BLOCK --save ./A_visual/FIAP_Div2k_tiny_x3 --scale 3 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP_Div2k_tiny_x3/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model FIAP_10BLOCK --save ./A_visual/FIAP_Div2k_tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/FIAP_Div2k_tiny_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt

