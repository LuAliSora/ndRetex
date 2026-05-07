# ndRetex
Normal-guided Diffusion for Re-texturing
# train
## uvRex
`python uvRex_main.py --input_dir input --model_dir weights --pretrained True --epoch_sum 9 --batch_size 2`
or
`python uvRex_main.py --input_dir input --model_dir weights --Init_Epoch 9 --epoch_sum 99 --batch_size 2`
## sd
`python sd_main.py --input_dir input --uvRex_model_dir weights --uvRex_Epoch 9 --tex_pretrained True --epoch_sum 9 --batch_size 2 --grad_acc_steps 2`
or
`python sd_main.py --input_dir input --uvRex_model_dir weights --uvRex_Epoch 9 --Init_Epoch 9 --epoch_sum 99 --batch_size 2 --grad_acc_steps 2`
# infer
## uvRex
`python uvRex_main.py --mode predict --input_dir input/test --model_dir weights  --Init_Epoch 9 --img 00006_00.jpg --texture 003.jpg`
## sd
`python sd_main.py --mode predict --input_dir input/test --uvRex_model_dir weights --uvRex_Epoch 9 --Init_Epoch 9 --img 00006_00.jpg --texture 003.jpg`
# others
check folder "nets" and download necessary pretrained models.