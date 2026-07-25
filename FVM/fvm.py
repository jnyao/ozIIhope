import os
import sys

if int(os.environ["NUMSPLIT"]) > 6:
    OZCUDNN_OUTCHNLS = 80
else:
    OZCUDNN_OUTCHNLS = 64
os.environ["OZCUDNN_OUTCHNLS"] = str(OZCUDNN_OUTCHNLS)

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.fx import symbolic_trace
from torch.profiler import profile, record_function, ProfilerActivity
import gp
import numpy
import cudnn

from torch.nn.parallel import DistributedDataParallel as DDP
from parallel import parallel_class
from preprocess import preprocess_class

from ncio import ncio_class, netcdf_read
from hinterp import interp_latlon_to_cube_class
from dycore import dycore_class
from diag import diag_class, calc_mass, calc_energy, calc_vor, \
                calc_total_mass, calc_total_energy, plot_cube_field, calc_gh_norm_error
import csv

if __name__ == "__main__":

    print('torch version',torch.__version__)
    print('cuda  version',torch.version.cuda)

    torch.autograd.set_detect_anomaly(False)
    # torch.set_default_dtype(torch.float32)
    # torch.set_printoptions(sci_mode=True,linewidth=800,precision=6)
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(sci_mode=True,linewidth=80,precision=15)

    D2R = math.pi / 180.
    R2D = 180. / math.pi
    nvar = 3
    
    case_num = 2 #case 2: Steady State Geostrophic Flow, case 6: Rossby-Haurwitz Wave, case 8: Perturbed Jet Flow
    run_days = 2 #12 for case_num=2, 90 for case_num=6, 6 for case_num=8
    run_hours = 0
    run_minutes = 0
    run_seconds = 0
    dt = 200  #600 for nx=ny=30 (330km), 400 for nx=ny=45 (220km), 200 for nx=ny=90 (110km), 100 for nx=ny=180 (55km)
    history_interval = 86400
    nx = 90
    ny = 90
    nz = 1
    sw = 5
    npanel = 6
    recon_scheme = 'TPP' # Choose from TPP or WENO
    output_full_field = False

    if dt>history_interval:
        print('Error: dt > history_interval')
        sys.exit(1)

    nt = round( ( run_days * 86400 + run_hours * 3600 + run_minutes * 60 + run_seconds ) / dt )
    rw = round( ( sw - 1 ) / 2 )
    iht = 0
    history_step = history_interval / dt
    nht = round( nt / history_step )

    parallel = parallel_class(nx,ny,nz,npanel)

    nbatch = 1
    nt_epoch = 1 # number of time slot in single epoch
    case = preprocess_class(case_num,nbatch,nt_epoch,nx,ny,nz,npanel,rw,parallel)
    mesh = case.mesh.to(parallel.device)
    q0 = case()
    q = q0.clone()

    dycore = dycore_class(mesh,parallel,case,recon_scheme,nvar,dt,q).to(parallel.device)

    file_name = 'output_case' + str(case.case_num) + '_order' + str(sw) + '_C' + str(nx) + '.nc'
    npy_name = 'npy_case' + str(case.case_num) + '_order' + str(sw) + '_C' + str(nx)
    ncio = ncio_class(file_name,mesh,case,dycore.recon,output_full_field,history_interval)
    # Integration from cell center point to cell average
    nvar, npanel, nx, ny = q.shape
    nx_halo, ny_halo = mesh.nx_halo, mesh.ny_halo
    nrx, nry = mesh.nrx, mesh.nry
    irs, ire, jrs, jre = mesh.irs, mesh.ire, mesh.jrs, mesh.jre
    q[...,irs:ire,jrs:jre] = torch.nn.functional.conv2d(q0.view(nvar*npanel,1,nx_halo,ny_halo), dycore.recon.conv_cell)\
                                                        .squeeze().view(nvar,npanel,nrx,nry)
    q = dycore.fill_ghost(q)
    q0 = q.clone()

    diag_field = diag_class()

    ght = q[0,:,mesh.ids:mesh.ide,mesh.jds:mesh.jde] / mesh.jab[:,mesh.ids:mesh.ide,mesh.jds:mesh.jde]
    c_max = torch.sqrt( torch.max( ght ) )
    dx_min = torch.sqrt( torch.min( mesh.jab[:,mesh.ids:mesh.ide,mesh.jds:mesh.jde] * mesh.jab_stretching ) * mesh.dx * mesh.dy )
    max_dt = dx_min / c_max
    print('max_dt = ',max_dt.item())

    mass0   = calc_total_mass(q,mesh)
    energy0 = calc_total_energy(q,mesh,case)
    vor0    = calc_vor(q,mesh,dycore.recon,dycore.fill_ghost)
    
    mass   = calc_total_mass(q,mesh)
    energy = calc_total_energy(q,mesh,case)
    vor    = calc_vor(q,mesh,dycore.recon,dycore.fill_ghost)
    diag_field.total_mass   = mass0
    diag_field.total_energy = energy0
    diag_field.vor          = vor0
    MCR = ( mass - mass0 ) / mass0
    ECR = ( energy - energy0 ) / energy0
    ncio.write_stat(q,diag_field,iht)
    print('iht ',iht, '/', nht,' MCR =',0., 'ECR =',0.,'L2 =',0.)

    torch.cuda.synchronize()
    for it in range(nt):
        gp.start("time_marching")
        q = dycore.temporal_operator(q)
        gp.stop("time_marching")
        # q = dycore.implicit_time_marching(q)
        # q = dycore.RKRW(q)
        # q = dycore.RKRW2(q)
        # make_dot(q).view()

        if( (it+1)%history_step==0 and (it+1)>=history_step ):
            iht += 1

            mass    = calc_total_mass(q,mesh)
            energy  = calc_total_energy(q,mesh,case)
            vor     = calc_vor(q,mesh,dycore.recon,dycore.fill_ghost)
            diag_field.total_mass   = mass
            diag_field.total_energy = energy
            diag_field.vor          = vor
            MCR = ( mass - mass0 ) / mass0
            ECR = ( energy - energy0 ) / energy0
            L2_error = calc_gh_norm_error(q,q0,mesh,case)
            ncio.write_stat(q,diag_field,iht)
            print('iht ',iht, '/', nht,' MCR =',MCR.item(), 'ECR =',ECR.item(),'L2 =',L2_error.item())


    torch.cuda.synchronize()
    
    print('Finish rank',parallel.rank)
    parallel.final()
    print(gp.timings)
