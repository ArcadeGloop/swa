

# cd 'C:\Users\alexy\OneDrive\Desktop\my_drop_box\second_degree\Deep Learning'

#python3 swa/train.py --dir=datadir --dataset=CIFAR100 --data_path=datadir  --model=PreResNet110 --epochs=10 --lr_init=0.1 --wd=3e-4 --swa --swa_start=3 --swa_lr=0.05 


import subprocess

subprocess.run(['python train.py --dir=datadir --dataset=CIFAR100 --data_path=datadir  --model=PreResNet110 --epochs=10 --lr_init=0.1 --wd=3e-4 --swa --swa_start=3 --swa_lr=0.05'],shell=True)

with open("train.py") as f:
    exec(f.read())

