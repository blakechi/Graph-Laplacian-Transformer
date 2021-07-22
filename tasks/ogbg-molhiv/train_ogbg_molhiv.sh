nohup python main.py \
    --log_msg d_10_e_256_h_8_he_4_a_1e-2_add_relu \
    --log_dir /media/storage0/pwchi/Graph_Laplacian_Transformer \
    --dataset_dir /media/data/pwchi/Graph_Laplacian_Transformer \
    --dataset_name ogbg-molhiv \
    --num_workers 2 \
    --cuda_device 1 \
    --epoch 150 \
    --batch_size 32 \
    --lr 0.001 \
    --num_token_layer 10 \
    --num_cls_layer 2 \
    --dim 256 \
    --heads 8 \
    --head_expand_scale 4 \
    --alpha 0.01 \
    --ff_dropout 0.2 \
    --attention_dropout 0.1 \
    --path_dropout 0.1 \
    --min_lr 0.000001 \
    --grad_clip_value "None" \
    --weight_decay 0.00001 \
    --num_classes 1 \
    --use_bias \
    --use_edge_bias \
    --use_attn_expand_bias \
    > nohup_add_relu.out \
    &
