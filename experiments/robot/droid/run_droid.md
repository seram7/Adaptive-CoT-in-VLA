

```
# start the server  
conda activate polymetis-local
python ~/code/droid/scripts/server/run_server.py


conda activate openvla
cd /home/monkgogi/code/openvla
python experiments/robot/droid/run_droid_eval.py \
--pretrained_checkpoint logs/openvla-7b+custom_droid_rlds_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug
```