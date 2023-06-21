CUDA_VISIBLE_DEVICES=0 python test.py  --hdim=16 --output_height=256 --m_plus=120 --weight_rec=0.05 --weight_kl=1.0  --weight_neg=0.5 --num_vae=0 --trainsize=800 --test_iter=100 --save_iter=100 --start_epoch=0  --batchSize=1 --nrow=8 --lr_e=0.0002 --lr_g=0.0002   --cuda  --nEpochs=0

