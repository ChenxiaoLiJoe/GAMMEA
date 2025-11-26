# # w/o surface
# FBDB15K
bash run_meaformer.sh 1 FBDB15K norm 0.8 0 
bash run_meaformer.sh 1 FBDB15K norm 0.5 0 
bash run_meaformer.sh 1 FBDB15K norm 0.2 0 
# FBYG15K
bash run_meaformer.sh 1 FBYG15K norm 0.8 0 
bash run_meaformer.sh 1 FBYG15K norm 0.5 0 
bash run_meaformer.sh 1 FBYG15K norm 0.2 0 
# DBP15K
bash run_meaformer.sh 1 DBP15K zh_en 0.3 0 
bash run_meaformer.sh 1 DBP15K ja_en 0.3 0 
bash run_meaformer.sh 1 DBP15K fr_en 0.3 0
# # w/ surface
# DBP15K
bash run_meaformer.sh 1 DBP15K zh_en 0.3 1 
bash run_meaformer.sh 1 DBP15K ja_en 0.3 1 
bash run_meaformer.sh 1 DBP15K fr_en 0.3 1


# # w/o surface
# FBDB15K
bash run_meaformer_il.sh 1 FBDB15K norm 0.8 0 
bash run_meaformer_il.sh 1 FBDB15K norm 0.5 0 
bash run_meaformer_il.sh 1 FBDB15K norm 0.2 0 
# FBYG15K
bash run_meaformer_il.sh 1 FBYG15K norm 0.8 0 
bash run_meaformer_il.sh 1 FBYG15K norm 0.5 0 
bash run_meaformer_il.sh 1 FBYG15K norm 0.2 0 
# DBP15K
bash run_meaformer_il.sh 1 DBP15K zh_en 0.3 0 
bash run_meaformer_il.sh 1 DBP15K ja_en 0.3 0 
bash run_meaformer_il.sh 1 DBP15K fr_en 0.3 0
# # w/ surface
# DBP15K
bash run_meaformer_il.sh 1 DBP15K zh_en 0.3 1 
bash run_meaformer_il.sh 1 DBP15K ja_en 0.3 1 
bash run_meaformer_il.sh 1 DBP15K fr_en 0.3 1

# nohup bash run_meaformer.sh 7 FBYG15K norm 0.2 0 > ./output/bdd_wo_FBYG0.2_jz.log &

# -------------------------------------------------------

# nohup bash run_meaformer.sh 0 FBDB15K norm 0.8 0 > ./output/1_bdd_wo_FBDB0.8_img_tau35_jz.log &
# nohup bash run_meaformer.sh 1 FBDB15K norm 0.5 0 > ./output/2_bdd_wo_FBDB0.5_img_tau35_jz.log &
# nohup bash run_meaformer.sh 2 FBDB15K norm 0.2 0 > ./output/3_bdd_wo_FBDB0.2_img_tau35_jz.log &

# nohup bash run_meaformer.sh 3 FBYG15K norm 0.8 0 > ./output/4_bdd_wo_FBYG0.8_img_tau35_jz.log &
# nohup bash run_meaformer.sh 4 FBYG15K norm 0.5 0 > ./output/5_bdd_wo_FBYG0.5_img_tau35_jz.log &
# nohup bash run_meaformer.sh 5 FBYG15K norm 0.2 0 > ./output/6_bdd_wo_FBYG0.2_img_tau35_jz.log &

# nohup bash run_meaformer.sh 6 DBP15K zh_en 0.3 0 > ./output/7_bdd_wo_zh_en_img_tau35_jz.log &
# nohup bash run_meaformer.sh 7 DBP15K ja_en 0.3 0 > ./output/8_bdd_wo_ja_en_img_tau35_jz.log &
# nohup bash run_meaformer.sh 0 DBP15K fr_en 0.3 0 > ./output/9_bdd_wo_fr_en_img_tau35_jz.log &

# nohup bash run_meaformer.sh 1 DBP15K zh_en 0.3 1 > ./output/10_bdd_w_zh_en_img_tau35_jz.log &
# nohup bash run_meaformer.sh 2 DBP15K ja_en 0.3 1 > ./output/11_bdd_w_ja_en_img_tau35_jz.log &
# nohup bash run_meaformer.sh 3 DBP15K fr_en 0.3 1 > ./output/12_bdd_w_fr_en_img_tau35_jz.log &

# -------------------------------------------------------

# nohup bash run_meaformer_il.sh 0 FBDB15K norm 0.8 0 > ./output/13_dd_wo_FBDB0.8_img_tau35_jz.log &
# nohup bash run_meaformer_il.sh 1 FBDB15K norm 0.5 0 > ./output/14_dd_wo_FBDB0.5_img_tau35_jz.log &
# nohup bash run_meaformer_il.sh 2 FBDB15K norm 0.2 0 > ./output/15_dd_wo_FBDB0.2_img_tau35_jz.log &

# nohup bash run_meaformer_il.sh 3 FBYG15K norm 0.8 0 > ./output/16_dd_wo_FBYG0.8_img_tau35_jz.log &
# nohup bash run_meaformer_il.sh 0 FBYG15K norm 0.5 0 > ./output/17_dd_wo_FBYG0.5_img_tau35_jz.log &
# nohup bash run_meaformer_il.sh 1 FBYG15K norm 0.2 0 > ./output/18_dd_wo_FBYG0.2_img_tau35_jz.log &

# nohup bash run_meaformer_il.sh 2 DBP15K zh_en 0.3 0 > ./output/19_dd_wo_zh_en_img_tau35_jz.log &
# nohup bash run_meaformer_il.sh 3 DBP15K ja_en 0.3 0 > ./output/20_dd_wo_ja_en_img_tau35_jz.log &
# nohup bash run_meaformer_il.sh 0 DBP15K fr_en 0.3 0 > ./output/21_dd_wo_fr_en_img_tau35_jz.log &

# nohup bash run_meaformer_il.sh 1 DBP15K zh_en 0.3 1 > ./output/22_dd_w_zh_en_img_tau35_jz.log &
# nohup bash run_meaformer_il.sh 2 DBP15K ja_en 0.3 1 > ./output/23_dd_w_ja_en_img_tau35_jz.log &
# nohup bash run_meaformer_il.sh 3 DBP15K fr_en 0.3 1 > ./output/24_dd_w_fr_en_img_tau35_jz.log &










