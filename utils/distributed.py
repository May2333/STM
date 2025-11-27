import os
import torch
import torch.distributed as dist
import logging


logger = logging.getLogger(__name__)


def setup_for_distributed(is_master):
    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)

    if not is_master:
        logging.disable()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # job started by torch.distributed.launch
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # local rank on the current node / global rank
        local_rank = int(os.environ['SLURM_LOCALID'])
        global_rank = int(os.environ['SLURM_PROCID'])
        # number of processes / GPUs per node
        world_size = int(os.environ["SLURM_NNODES"]) * \
            int(os.environ["SLURM_TASKS_PER_NODE"][0])

        print(world_size)

        args.rank = global_rank
        args.gpu = local_rank
        args.world_size = world_size
    else:
        logger.info('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'

    if "tcp" in args.dist_url:  # in slurm, multiple program runs in a single node
        dist_port = int(args.dist_url.split(":")[-1])
        while is_port_in_use(dist_port):
            dist_port += 10
        args.dist_url = ":".join(args.dist_url.split(":")[:-1] + [str(dist_port)])

    logger.info('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url))
    if "SLURM_JOB_ID" in os.environ:
        logger.info(f"SLURM_JOB_ID {os.environ['SLURM_JOB_ID']}")
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# Copyright (c) Facebook, Inc. and its affiliates.
# copied from https://github.com/facebookresearch/vissl/blob/master/vissl/utils/distributed_gradients.py
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


# copied from megavlt
def gather_tensor_along_batch_with_backward(tensor, dim=0):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    tensor_list = GatherLayer.apply(tensor)
    tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


@torch.no_grad()
def gather_tensor_along_batch(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        tensor_list = []

        for _ in range(world_size):
            tensor_list.append(torch.zeros_like(tensor))

        dist.all_gather(tensor_list, tensor)
        tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


######  with different batch_size ~
def ddp_allgather(input):
    tmp_input = input.cuda()
    size = torch.tensor(tmp_input.shape[0]).cuda()
    size_list = [torch.zeros_like(size) for i in range(dist.get_world_size())]
    dist.all_gather(size_list, size)
    max_size = max(size_list).item()
    padding_size = max_size - size 
    if padding_size > 0 :
        padding_tensor = torch.zeros(padding_size,*tmp_input.shape[1:]).to(tmp_input)
        tmp_input = torch.cat((tmp_input, padding_tensor), dim = 0)
    tmp_list = [torch.zeros_like(tmp_input) for i in range(dist.get_world_size())]
    dist.all_gather(tmp_list, tmp_input)
    output = []
    for t, s in zip(tmp_list, size_list):
        output.append(t[:s])
    output = torch.cat(output,dim=0) 
    return output 

import pickle
import math
def all_gather_list(data):
    """Gathers arbitrary data from all nodes into a list."""
    enc = pickle.dumps(data)

    enc_size = len(enc)
    max_size = ddp_allgather(torch.tensor([enc_size]).cuda()).max().item()
    in_buffer, enc_byte = _encode(enc, max_size)

    out_buffer = ddp_allgather(in_buffer[:enc_byte+enc_size])

    results = []
    for _ in range(dist.get_world_size()):
        bytes_list, shift = _decode(out_buffer, enc_byte)
        out_buffer = out_buffer[shift:]
        result = pickle.loads(bytes_list)
        results.append(result)
    return results

def _encode(enc, max_size, use_max_size=False):
    enc_size = len(enc)
    enc_byte = max(math.floor(math.log(max_size, 256)+1), 1)
    if use_max_size:
        # this is used for broadcasting
        buffer_ = torch.cuda.ByteTensor(max_size+enc_byte)
    else:
        buffer_ = torch.cuda.ByteTensor(enc_size+enc_byte)
    remainder = enc_size
    for i in range(enc_byte):
        base = 256 ** (enc_byte-i-1)
        buffer_[i] = remainder // base
        remainder %= base
    buffer_[enc_byte:enc_byte+enc_size] = torch.ByteTensor(list(enc))
    return buffer_, enc_byte


def _decode(buffer_, enc_byte):
    size = sum(256 ** (enc_byte-i-1) * buffer_[i].item()
               for i in range(enc_byte))
    bytes_list = bytes(buffer_[enc_byte:enc_byte+size].tolist())
    shift = size + enc_byte
    return bytes_list, shift