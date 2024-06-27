###FIAP-S, FIAP, and FIAP-L training commands
#train_FIAP-S_6BLOCK_32C_64*64patch_size
#python main.py --model FIAP_6BLOCK --save ./train/FIAP-S_Div2k_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_6BLOCK --save ./train/FIAP-S_Div2k_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_6BLOCK --save ./train/FIAP-S_Div2k_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

#train_FIAP_10BLOCK_32C_64*64patch_size
#python main.py --model FIAP_10BLOCK --save ./train/FIAP_Div2K_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_10BLOCK --save ./train/FIAP_Div2K_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_10BLOCK --save ./train/FIAP_Div2K_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

#train_FIAP-L_10BLOCK_32C_64*64patch_size
#python main.py --model FIAP_10BLOCK --save ./train/FIAP-L_Div2K_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_10BLOCK --save ./train/FIAP-L_Div2K_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model FIAP_10BLOCK --save ./train/FIAP-L_Div2K_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

###FIAP-S, FIAP, and FIAP-L testing commands
#test_FIAP-S_6BLOCK_32C_64*64patch_size
#python main.py --model FIAP_6BLOCK --save ./test/FIAP-S_Div2k_tiny_x2 --scale 2 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP-S_Div2k/fiaps_x2.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_6BLOCK --save ./test/FIAP-S_Div2k_tiny_x3 --scale 3 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP-S_Div2k/fiaps_x3.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_6BLOCK --save ./test/FIAP-S_Div2k_tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP-S_Div2k/fiaps_x4.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

#test_FIAP_10BLOCK_32C_64*64patch_size
#python main.py --model FIAP_10BLOCK --save ./test/FIAP_Div2k_tiny_x2 --scale 2 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP_Div2K/fiap_x2.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_10BLOCK --save ./test/FIAP_Div2k_tiny_x3 --scale 3 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP_Div2K/fiap_x3.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_10BLOCK --save ./test/FIAP_Div2k_tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP_Div2K/fiap_x4.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

#test_FIAP-L_10BLOCK_32C_64*64patch_size
#python main.py --model FIAP_10BLOCK --save ./test/FIAP-L_Div2k_x2 --scale 2 --n_feats 48 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP-L_Div2k/fiapl_x2.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_10BLOCK --save ./test/FIAP-L_Div2k_x3 --scale 3 --n_feats 48 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP-L_Div2k/fiapl_x3.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model FIAP_10BLOCK --save ./test/FIAP-L_Div2k_x4 --scale 4 --n_feats 48 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP-L_Div2k/fiapl_x4.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

###Visualization commands
#LR HR SR input
#python main.py --model FIAP_10BLOCK --save ./visual/FIAP_Div2k_tiny_x2 --scale 2 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP_Div2K/fiap_x2.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model FIAP_10BLOCK --save ./visual/FIAP_Div2k_tiny_x3 --scale 3 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP_Div2K/fiap_x3.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model FIAP_10BLOCK --save ./visual/FIAP_Div2k_tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProject/FIAP/pretrained/FIAP_Div2K/fiap_x4.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt

