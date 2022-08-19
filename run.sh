pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

python  train.py --model align --batch_size 2 --checkpoint_path checkpoint --lr 2e-4 --print_interval 100 --save_interval 100 
# python -m torch.distributed.launch train.py --model psp --batch_size 2 --checkpoint_path checkpoint-test --lr 2e-4 --print_interval 1 --save_interval 1 --test_interval 1
