import os
import math
import torch
import torch.distributed as dist
from torch.autograd.function import Function
    
class parallel_class(torch.nn.Module):
    def __init__(self,nx,ny,nz,npanel):
        super(parallel_class, self).__init__()

        print(
            'MASTER_ADDR',os.environ['MASTER_ADDR'],'\n',
            'MASTER_PORT',os.environ['MASTER_PORT'],'\n',
            'LOCAL_RANK',os.environ['LOCAL_RANK'],'\n',
            'RANK',os.environ['RANK'],'\n',
            'WORLD_SIZE',os.environ['WORLD_SIZE'],'\n',
            ''
        )

        rank = int( os.environ['LOCAL_RANK'] )
        # rank = torch.distributed.get_rank()
        nproc = int( os.environ['WORLD_SIZE'] )
        # nproc = torch.distributed.get_world_size()

        torch.cuda.set_device(0)
        device = torch.device('cuda',0)
        
        self.device = device
        self.rank = rank
        self.nproc = nproc

        if self.rank==0:
            print('is_mpi_available ',torch.distributed.is_mpi_available())
            print('is_gloo_available ',torch.distributed.is_gloo_available())
            print('is_nccl_available ',torch.distributed.is_nccl_available())
            print('')

        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=nproc)
    
    def final(self):
        print( 'Finish run, destory the distributed environment' )
        dist.destroy_process_group()
    
    def all_gether_panels(self,var):
        input_list = [torch.zeros_like(var) for k in range(self.nproc)]
        dist.all_gather(input_list, var, async_op=False)
        # inputs = torch.stack(input_list, dim=2)
        var = torch.cat(input_list, dim=2)
        return var
    
    def all_gether_new_dim(self,var):
        ndim = var.ndimension()
        input_list = [torch.zeros_like(var) for k in range(self.nproc)]
        dist.all_gather(input_list, var, async_op=False)
        var = torch.stack(input_list, dim=ndim)
        return var

    def round_robin(self,nx):
        # Attension! rank start from 0
        mod = nx%self.nproc
        min_points_per_proc = math.floor( nx / self.nproc )
        max_points_per_proc = min_points_per_proc + 1

        n_large_proc = mod
        n_small_proc = self.nproc - n_large_proc

        if self.rank+1 <= mod:
            nx_local = max_points_per_proc
            ids = self.rank * max_points_per_proc + 1
            ide = ids + nx_local - 1
        else:
            nx_local = min_points_per_proc
            ids = n_large_proc * max_points_per_proc + ( self.rank - n_large_proc ) * min_points_per_proc + 1
            ide = ids + nx_local - 1

        return ids, ide, nx_local

    def all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        """
        Differentiable counterpart of `dist.all_reduce`.
        """
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() == 1
        ):
            return input
        return _AllReduce.apply(input)
    
class _AllReduce(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad_output, async_op=False)
        return grad_output
