import os
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.autograd.function import Function
import netCDF4 as nc

class ncio_class(torch.nn.Module):
    def __init__(self,ncfile,mesh,case,recon,output_full_field,history_interval):
        super(ncio_class, self).__init__()
        print('Create',ncfile,'for output')
        pi = math.pi
        R2D = 180. / pi
        self.ncfile = ncfile
        self.mesh = mesh
        self.case = case
        self.recon = recon
        self.netcdf_format = 'NETCDF4'
        # self.netcdf_format = 'NETCDF4_CLASSIC'
        # self.netcdf_format = 'NETCDF3_64BIT_DATA'

        npanel = mesh.npanel
        if output_full_field:
            nx = mesh.nx_halo
            ny = mesh.ny_halo
            ids = mesh.ims
            ide = mesh.ime
            jds = mesh.jms
            jde = mesh.jme
        else:
            nx = mesh.nx
            ny = mesh.ny
            ids = mesh.ids
            ide = mesh.ide
            jds = mesh.jds
            jde = mesh.jde
        
        self.nx = nx
        self.ny = ny
        self.npanel = npanel
        self.case_num = self.case.case_num
        self.dx = self.mesh.dx
        self.dy = self.mesh.dy
        self.history_interval = history_interval
        self.ids = ids
        self.ide = ide
        self.jds = jds
        self.jde = jde

        dim_list     = [['time',None  ]]
        dim_list.append(['i'   ,nx    ])
        dim_list.append(['j'   ,ny    ])
        dim_list.append(['dom' ,npanel])
        ndim = len(dim_list[:])
        self.ndim = ndim
        self.dim_list = dim_list

        varname_idx     = 0
        varytpe_idx     = 1
        dimension_idx   = 2
        long_name_idx   = 3
        description_idx = 4
        units_idx       = 5
        output_idx      = 6
        
        self.varname_idx     = varname_idx    
        self.varytpe_idx     = varytpe_idx    
        self.dimension_idx   = dimension_idx  
        self.long_name_idx   = long_name_idx  
        self.description_idx = description_idx
        self.units_idx       = units_idx      
        self.output_idx      = output_idx     

                       # varname, vartype, dimension, long_name, description, units, output
        var_list     = [['time'         ,'f8',('time'              ),'time'               ,'time'                    ,'seconds since 0001-01-01 00:00:00.0',True]]
        var_list.append(['longitude'    ,'f8',(       'dom','j','i'),'Longitude'          ,'longitude of cell center','degree_east'                        ,False])
        var_list.append(['latitude'     ,'f8',(       'dom','j','i'),'Latitude'           ,'latitude of cell center' ,'degree_north'                       ,False])
        var_list.append(['jab'          ,'f8',(       'dom','j','i'),'Metric_Jacobian'    ,'Metric Jacobian'         ,'m2'                                 ,False])
        var_list.append(['ghs'          ,'f8',(       'dom','j','i'),'topo_geo_height'    ,'topography geo-height'   ,'m2/s2'                              ,False])
        var_list.append(['q1'           ,'f8',('time','dom','j','i'),'q1'                 ,'prognostic variable 1'   ,'-'                                  ,True])
        var_list.append(['q2'           ,'f8',('time','dom','j','i'),'q2'                 ,'prognostic variable 2'   ,'-'                                  ,True])
        var_list.append(['q3'           ,'f8',('time','dom','j','i'),'q3'                 ,'prognostic variable 3'   ,'-'                                  ,True])
        var_list.append(['ght'          ,'f8',('time','dom','j','i'),'geopotential_height','geopotential height'     ,'m2/s2'                              ,True])
        var_list.append(['uc'           ,'f8',('time','dom','j','i'),'uc'                 ,'u contravariant wind'    ,'-'                                  ,True])
        var_list.append(['vc'           ,'f8',('time','dom','j','i'),'vc'                 ,'v contravariant wind'    ,'-'                                  ,True])
        var_list.append(['us'           ,'f8',('time','dom','j','i'),'zonal_wind'         ,'zonal wind'              ,'m/s'                                ,True])
        var_list.append(['vs'           ,'f8',('time','dom','j','i'),'meridianal_wind'    ,'meridianal wind'         ,'m/s'                                ,True])
        var_list.append(['total_mass'   ,'f8',('time'              ),'total_mass'         ,'total mass'              ,'m'                                  ,True])
        var_list.append(['total_energy' ,'f8',('time'              ),'total_energy'       ,'total energy'            ,'J'                                  ,True])
        var_list.append(['vor'          ,'f8',('time','dom','j','i'),'relative_vorticity' ,'relative vorticity'      ,'s-1'                                ,True])

        nvar = len(var_list[:])
        self.nvar = nvar
        self.var_list = var_list

        Dataset = nc.Dataset(ncfile,mode='w',format=self.netcdf_format)

        for idim in range(ndim):
            dim = Dataset.createDimension(dimname=dim_list[idim][0],size=dim_list[idim][1])

        for ivar in range(nvar):
            var = Dataset.createVariable(varname     = var_list[ivar][varname_idx],
                                         datatype    = var_list[ivar][varytpe_idx],
                                         dimensions  = var_list[ivar][dimension_idx],
                                         compression = 'zlib',
                                         complevel   = 5)
            var.long_name   = var_list[ivar][long_name_idx]
            var.description = var_list[ivar][description_idx]
            var.units       = var_list[ivar][units_idx]
        
        ghs = self.case.ghs[...,ids:ide,jds:jde]
        Dataset['longitude'][:] = mesh.lon[:,ids:ide,jds:jde].permute(0,2,1).cpu().numpy() * R2D
        Dataset['latitude' ][:] = mesh.lat[:,ids:ide,jds:jde].permute(0,2,1).cpu().numpy() * R2D
        Dataset['ghs'      ][:] = ghs                        .permute(0,2,1).cpu().numpy()
        Dataset['jab'      ][:] = mesh.jab[:,ids:ide,jds:jde].permute(0,2,1).cpu().numpy()

        Dataset.HOPE_Version = 'PyTorch'
        Dataset.case_num = self.case_num
        Dataset.dx = self.dx * R2D
        Dataset.dy = self.dy * R2D
        
        if output_full_field:
            Dataset.ids = self.ids
            Dataset.ide = self.ide - 1
            Dataset.jds = self.jds
            Dataset.jde = self.jde - 1
            Dataset.ims = self.mesh.ims
            Dataset.ime = self.mesh.ime - 1
            Dataset.jms = self.mesh.jms
            Dataset.jme = self.mesh.jme - 1
        else:
            Dataset.ids = 1
            Dataset.ide = mesh.nx
            Dataset.jds = 1
            Dataset.jde = mesh.ny
            Dataset.ims = 1
            Dataset.ime = mesh.nx
            Dataset.jms = 1
            Dataset.jme = mesh.ny

        Dataset.ifs = 1
        Dataset.ife = 6

        Dataset.close()

    def write_stat(self,q0,diag_field,time_slot):
        ncfile   = self.ncfile
        var_list = self.var_list
        nvar_out = self.nvar
        mesh     = self.mesh
        pc       = self.mesh.pc
        nPOR     = self.mesh.nPOR
        np       = self.mesh.npanel_local
        nx       = self.mesh.nx
        ny       = self.mesh.ny
        nrx      = self.mesh.nrx
        nry      = self.mesh.nry
        nx_halo  = self.mesh.nx_halo
        ny_halo  = self.mesh.ny_halo
        ids      = self.mesh.ids
        ide      = self.mesh.ide
        jds      = self.mesh.jds
        jde      = self.mesh.jde
        irs      = self.mesh.irs
        ire      = self.mesh.ire
        jrs      = self.mesh.jrs
        jre      = self.mesh.jre
        rw       = self.mesh.rw
             
        varname_idx     = self.varname_idx    
        varytpe_idx     = self.varytpe_idx    
        dimension_idx   = self.dimension_idx  
        long_name_idx   = self.long_name_idx  
        description_idx = self.description_idx
        units_idx       = self.units_idx      
        output_idx      = self.output_idx     

        # diag variables
        nvar_prog, _, _, _ = q0.shape
        qrec = self.recon( q0.view(nvar_prog*np,1,nx_halo,ny_halo) ).view(nvar_prog,np,nPOR,nrx,nry)
        qc = qrec[...,pc,:,:]
        
        ght   = qc[0,...] / mesh.jab[:,irs:ire,jrs:jre]
        ghs   = self.case.ghs[...,irs:ire,jrs:jre]
        Jabgh = qc[0,...] - ghs * mesh.jab[:,irs:ire,jrs:jre]
        uc    = qc[1,...] / Jabgh
        vc    = qc[2,...] / Jabgh
        us, vs = mesh.contravProjPlane2Sphere( mesh.A[...,irs:ire,jrs:jre], uc, vc )

        combine_vars = torch.stack((ght,uc,vc,us,vs),dim=0)
        nvar_combine = combine_vars.shape[0]
        combine_vars = torch.nn.functional.conv2d(combine_vars.view(nvar_combine*np,1,nrx,nry), self.recon.conv_cell).squeeze() \
                                                        .view(nvar_combine,np,nx,ny)
        ght, uc, vc, us, vs = combine_vars[0,...], combine_vars[1,...], combine_vars[2,...], combine_vars[3,...], combine_vars[4,...]

        total_mass   = diag_field.total_mass
        total_energy = diag_field.total_energy
        vor          = diag_field.vor

        # Collect variables for output
        var_out = torch.zeros(nvar_out,np,ny,nx)
        var_out[0 ,...] = time_slot * self.history_interval
        var_out[1 ,...] = q0[0,:,ids:ide,jds:jde]
        var_out[2 ,...] = q0[1,:,ids:ide,jds:jde]
        var_out[3 ,...] = q0[2,:,ids:ide,jds:jde]
        var_out[4 ,...] = ght
        var_out[5 ,...] = uc
        var_out[6 ,...] = vc
        var_out[7 ,...] = us
        var_out[8 ,...] = vs
        var_out[9 ,...] = total_mass
        var_out[10,...] = total_energy
        var_out[11,...] = vor

        Dataset = nc.Dataset(ncfile,mode='a',format=self.netcdf_format)

        ifield = 0
        for ivar in range(nvar_out):
            if var_list[ivar][output_idx] == True:
                varname = var_list[ivar][varname_idx]
                dim_info = var_list[ivar][dimension_idx]
                if dim_info==('time'              ):
                    Dataset[varname][time_slot] = var_out[ifield,0,0,0].detach().cpu().numpy()
                    ifield += 1
                elif dim_info==('time','dom','j','i'):
                    Dataset[varname][time_slot,...] = var_out[ifield,...].permute(0,2,1).detach().cpu().numpy()
                    ifield += 1

        Dataset.close()

def netcdf_read(ncfile,varname,device):
    Dataset = nc.Dataset(ncfile)
    data = Dataset[varname][:]
    data = torch.tensor( data ).to(device)
    Dataset.close()
    return data
    
def netcdf_read_ghost_interp_matrix(ncfile,varname,dtype,device):
    crow_indices_dim_name = 'crow_indices_'+varname
    col_indices_dim_name = 'col_indices_'+varname
    nnz_dim_name = 'nnz_'+varname

    Dataset = nc.Dataset(ncfile,mode='r',format='NETCDF4')
    crow_indices = Dataset[crow_indices_dim_name][:]
    col_indices = Dataset[col_indices_dim_name][:]
    values = Dataset[varname][:]
    values = torch.tensor( values, dtype=dtype )
    mtx = torch.sparse_csr_tensor(crow_indices, col_indices, values, device=device)
    
    Dataset.close()
    return mtx

def netcdf_write_ghost_interp_matrix_init(ncfile,varname,open_mode='w'):
    crow_indices_dim_name = 'crow_indices_'+varname
    col_indices_dim_name = 'col_indices_'+varname
    nnz_dim_name = 'nnz_'+varname
    compress_method = 'zlib'
    compress_level = 5

    Dataset = nc.Dataset(ncfile,mode=open_mode,format='NETCDF4')
    dim = Dataset.createDimension(dimname=crow_indices_dim_name,size=None)
    dim = Dataset.createDimension(dimname=nnz_dim_name         ,size=None)

    var = Dataset.createVariable(varname     = crow_indices_dim_name,
                                 datatype    = 'i8',
                                 dimensions  = (crow_indices_dim_name),
                                 compression = compress_method,
                                 complevel   = compress_level)
    
    var = Dataset.createVariable(varname     = col_indices_dim_name,
                                 datatype    = 'i8',
                                 dimensions  = (nnz_dim_name),
                                 compression = compress_method,
                                 complevel   = compress_level)
    
    var = Dataset.createVariable(varname     = varname,
                                 datatype    = 'f8',
                                 dimensions  = (nnz_dim_name),
                                 compression = compress_method,
                                 complevel   = compress_level)
    
    Dataset[varname].completed_idx = 0
    Dataset[varname].completed = 0

    Dataset.close()

def netcdf_write_ghost_interp_matrix(ncfile,varname,mtx):
    nnz = mtx._nnz()
    row = mtx.crow_indices().cpu().numpy()
    col = mtx.col_indices().cpu().numpy()
    nrow_idx = row.shape[0]
    val = mtx.values().cpu().numpy()

    crow_indices_dim_name = 'crow_indices_'+varname
    col_indices_dim_name = 'col_indices_'+varname
    nnz_dim_name = 'nnz_'+varname
    compress_method = 'zlib'
    compress_level = 5

    Dataset = nc.Dataset(ncfile,mode='w',format='NETCDF4')

    dim = Dataset.createDimension(dimname=crow_indices_dim_name,size=nrow_idx)
    dim = Dataset.createDimension(dimname=nnz_dim_name         ,size=nnz     )

    var = Dataset.createVariable(varname     = crow_indices_dim_name,
                                 datatype    = 'i8',
                                 dimensions  = (crow_indices_dim_name),
                                 compression = compress_method,
                                 complevel   = compress_level)
    Dataset[crow_indices_dim_name][:] = row
    
    var = Dataset.createVariable(varname     = col_indices_dim_name,
                                 datatype    = 'i8',
                                 dimensions  = (nnz_dim_name),
                                 compression = compress_method,
                                 complevel   = compress_level)
    Dataset[col_indices_dim_name][:] = col
    
    var = Dataset.createVariable(varname     = varname,
                                 datatype    = 'f8',
                                 dimensions  = (nnz_dim_name),
                                 compression = compress_method,
                                 complevel   = compress_level)
    Dataset[varname][:] = val
    Dataset[varname].completed = 1

    Dataset.close()

def netcdf_write_ghost_interp_matrix_append(ncfile,varname,mtx):
    crow_indices_dim_name = 'crow_indices_'+varname
    col_indices_dim_name = 'col_indices_'+varname
    nnz_dim_name = 'nnz_'+varname

    nnz = mtx._nnz()
    row = mtx.crow_indices().cpu().numpy()
    col = mtx.col_indices().cpu().numpy()
    nrow = row.shape[0]
    val = mtx.values().cpu().numpy()

    Dataset = nc.Dataset(ncfile,mode='a',format='NETCDF4')
    nrow_idx_prev = max( [Dataset[crow_indices_dim_name][:].shape[0] - 1, 0] )
    nrow_idx = nrow_idx_prev + nrow
    nnz_prev = Dataset[col_indices_dim_name][:].shape[0]
    Dataset[crow_indices_dim_name][nrow_idx_prev:nrow_idx] = row + nnz_prev
    Dataset[col_indices_dim_name][nnz_prev:nnz_prev+nnz] = col
    Dataset[varname][nnz_prev:nnz_prev+nnz] = val
    Dataset[varname].completed_idx = nrow_idx - 1
    Dataset.close()

def netcdf_write_ghost_interp_matrix_final(ncfile,varname):
    Dataset = nc.Dataset(ncfile,mode='a',format='NETCDF4')
    Dataset[varname].completed = 1
    Dataset.close()

def netcdf_write_ghost_interp_matrix_check(ncfile,varname):
    Dataset = nc.Dataset(ncfile,mode='r',format='NETCDF4')
    completed_idx = Dataset[varname].completed_idx
    completed = Dataset[varname].completed

    Dataset.close()
    if completed==1:
        completed = True
    else:
        completed = False
    return completed_idx, completed

def netcdf_check_variable_exists(ncfile, varname):
    """
    Check if a variable exists in the nc file.

    :param ncfile: Path to the .nc file
    :param varname: Name of the variable to check
    :return: True if the variable exists, otherwise False
    """
    try:
        # Open the nc file
        with nc.Dataset(ncfile, 'r') as dataset:
            # Get all variable names in the file
            variables = dataset.variables.keys()
            
            # Check if the target variable is in the variable list
            if varname in variables:
                print(f"Variable '{varname}' exists in " + ncfile)
                return True
            else:
                print(f"Variable '{varname}' does not exist in " + ncfile)
                return False
        nc.Dataset.close()
        
    except Exception as e:
        print(f"Error reading {ncfile}: {e}")
        return False