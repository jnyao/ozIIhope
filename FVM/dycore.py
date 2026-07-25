import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
import gp
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import numpy
import math
from hinterp import interp_latlon_to_cube_class
from recon import recon_class
from diag import pause, plot_cube_field
from ncio import netcdf_read_ghost_interp_matrix 
from ncio import netcdf_write_ghost_interp_matrix
from ncio import netcdf_write_ghost_interp_matrix_init
from ncio import netcdf_write_ghost_interp_matrix_append
from ncio import netcdf_write_ghost_interp_matrix_final
from ncio import netcdf_write_ghost_interp_matrix_check
from ncio import netcdf_check_variable_exists
import netCDF4 as nc
import cudnn
OZAKI_OP_DIR = os.environ["OZAKI_OP_DIR"]
torch.ops.load_library(OZAKI_OP_DIR)
numsplit = int(os.environ["NUMSPLIT"])
bits_per_slice = int(os.environ["BITS_PER_SLICE"])
os.environ["BITS_PER_SLICE"] = str(bits_per_slice)
OZCUDNN_CHNLS = 16
OZCUDNN_OUTCHNLS = int(os.environ["OZCUDNN_OUTCHNLS"])
OZ_NORMAL = 0
OZ_BSHALF = 1
OZ_MODE = int(os.environ["OZ_MODE"])
prec_mode = int(os.environ["PREC_MODE"])

class dycore_class(torch.nn.Module):
    def __init__(self,mesh,parallel,case,recon_scheme,nvar,dt,q0,prec_mode=prec_mode):
        super(dycore_class, self).__init__()
        self.nvar = nvar
        self.dt = dt
        self.parallel = parallel
        self.recon_scheme = recon_scheme
        self.device = self.parallel.device

        self.case = case
        self.mesh = mesh
        self.sw = mesh.sw
        self.rw = mesh.rw

        nEOC = self.mesh.nEOC
        nPOC = self.mesh.nPOC
        nQOC = self.mesh.nQOC
        pc  = self.mesh.pc
        pls = self.mesh.pls
        ple = self.mesh.ple
        prs = self.mesh.prs
        pre = self.mesh.pre
        pbs = self.mesh.pbs
        pbe = self.mesh.pbe
        pts = self.mesh.pts
        pte = self.mesh.pte
        pqs = self.mesh.pqs
        pqe = self.mesh.pqe
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        irs = self.mesh.irs
        ire = self.mesh.ire
        jrs = self.mesh.jrs
        jre = self.mesh.jre
        
        np, nPOE, nx, ny = self.mesh.npanel, self.mesh.nPOE, self.mesh.nx, self.mesh.ny
        nrx, nry = self.mesh.nrx, self.mesh.nry
        nx_halo, ny_halo = self.mesh.nx_halo, self.mesh.ny_halo

        order = self.sw
        self.mtx_file = 'ghost_mtx_nc'+str(nx)+'_order'+str(order)+'.nc'

        self.gx = self.mesh.gx
        self.gw = self.mesh.gw

        self.recon = recon_class( self.mesh, self.recon_scheme, self.nvar, self.device )

        # Prepare static data for calc_src
        self.coef_M = self.mesh.coef_M[...,irs:ire,jrs:jre]

        self.iGQ_C = self.case.Coriolis * self.mesh.jab * self.mesh.jab_stretching * self.mesh.iG
        self.iGQ_C = self.iGQ_C[...,irs:ire,jrs:jre]

        self.jabghsQ = self.mesh.jab * self.case.ghs
        self.iGQ_S   = self.jabghsQ * self.mesh.iG
        self.iGQ_S   = self.iGQ_S[...,irs:ire,jrs:jre]
        self.jabghsQ = self.jabghsQ[...,irs:ire,jrs:jre]

        # Prepare static_data for riemann solver
        self.convert_coefL = torch.sqrt( self.mesh.iGL[0,0,...] )
        self.convert_coefB = torch.sqrt( self.mesh.iGB[1,1,...] )
        self.JiGL = self.mesh.jabL * self.mesh.iGL[0:2,0,...]
        self.JiGB = self.mesh.jabB * self.mesh.iGB[0:2,1,...]
        self.jabghsL = self.mesh.jabL * self.case.ghsL
        self.jabghsB = self.mesh.jabB * self.case.ghsB

        self.iter = 0

        if self.gw.dtype==torch.float64:
            self.bdy_tolerance = torch.tensor( 1.e-15, dtype=torch.float64, device=self.device )
        elif self.gw.dtype==torch.float32:
            self.bdy_tolerance = torch.tensor( 1.e-5, dtype=torch.float32, device=self.device )
            
        # Calculate ghost interp matrix
        operator = self.fill_ghost_out_domain
        q = torch.zeros(nvar,np,nx,ny,device=self.device)
        varname = 'gst_interp_mtx'
        nrow_per_block = 20
        self.gst_mtx = self.get_operator_jacobian(operator,q,varname,nrow_per_block)

        # Calculate unify panel bdy flux matrix
        operator = self.unify_panel_bdy_flux_by_q_edge
        q = torch.zeros(nvar,np,nPOE*nEOC,nrx,device=self.device)
        varname = 'unify_panel_bdy_flux_mtx'
        nrow_per_block = 500
        self.unify_panel_bdy_flux_matrix = self.get_operator_jacobian(operator,q,varname,nrow_per_block)
        
        self.unify_panel_bdy_batch_csr_matmul = torch.vmap( self.unify_panel_bdy_flux_matrix.matmul )

        self.qC = torch.zeros(nvar,np,1   ,nrx ,nry ,device=self.device)
        self.qL = torch.zeros(nvar,np,nPOE,nx+1,nry ,device=self.device)
        self.qR = torch.zeros(nvar,np,nPOE,nx+1,nry ,device=self.device)
        self.qB = torch.zeros(nvar,np,nPOE,nrx ,ny+1,device=self.device)
        self.qT = torch.zeros(nvar,np,nPOE,nrx ,ny+1,device=self.device)

        yworksize = 2**16*10
        self.d_x = torch.zeros(yworksize, dtype=torch.float32, device=self.device)
        self.d_y = torch.zeros(yworksize, dtype=torch.float32, device=self.device)
        self.handle = handle = cudnn.create_handle()
        self.prec_mode = prec_mode
        if self.prec_mode == 0:
            self.input_type64  = torch.get_default_dtype()
            self.output_type64 = torch.get_default_dtype()
            if self.output_type64 == torch.double:
                self.cudnn_type = cudnn.data_type.DOUBLE
            else:
                self.cudnn_type = cudnn.data_type.FLOAT
            self.convgraph64 = cudnn.pygraph(
                handle=handle,
                io_data_type=self.cudnn_type,
                intermediate_data_type=self.cudnn_type,
                compute_data_type=self.cudnn_type
            )
            size_img = [np, 1, nx_halo, ny_halo]
            size_knl = list(self.recon.DxyMtxC_conv.shape)
            size_out = [np,size_knl[0],nrx,nry]
            n = size_img[0]
            c = size_img[1]
            h = size_img[2]
            w = size_img[3]
            k = size_knl[0]
            r = size_knl[2]
            s = size_knl[3]
            self.tensor_float64 = torch.zeros(size_img, dtype=self.input_type64, device=self.device)
            self.vector_float64 = torch.zeros(size_knl, dtype=self.input_type64, device=self.device)
            self.y64    = torch.zeros(size_out, dtype=self.output_type64, device=self.device)
            self.y_dp64 = torch.zeros(size_out, dtype=self.output_type64, device=self.device)

            self.x_cudnn_tensor64 = self.convgraph64.tensor_like(self.tensor_float64)
            self.x_cudnn_tensor64.set_name("x").set_stride([c * h * w, 1, c * w, c])
            self.w_cudnn_tensor64 = self.convgraph64.tensor_like(self.vector_float64)
            self.w_cudnn_tensor64.set_name("w").set_stride([c * r * s, 1, c * s, c])
            self.y_cudnn_tensor64 = self.convgraph64.conv_fprop(
                name="convolution",
                image=self.x_cudnn_tensor64,
                weight=self.w_cudnn_tensor64,
                compute_data_type=self.cudnn_type,
                padding = [0,0], stride = [1,1], dilation=[1,1]
            )
            self.y_cudnn_tensor64.set_name("y").set_output(True).set_data_type(self.output_type64)
            self.convgraph64.validate()
            self.convgraph64.build_operation_graph()
            self.convgraph64.create_execution_plans([cudnn.heur_mode.A])
            self.convgraph64.check_support()
            self.convgraph64.build_plans()
            self.workspace64 = torch.empty(self.convgraph64.get_workspace_size(), device=self.device, dtype=torch.uint8)
        self.input_type  = torch.float16
        self.output_type = torch.float32
        self.convgraph = cudnn.pygraph(
            handle=handle,
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT
        )
        if OZ_MODE == OZ_BSHALF:
                size_img = [np // 2, OZCUDNN_CHNLS, nx_halo, ny_halo]
        elif OZ_MODE == OZ_NORMAL:
                size_img = [np, OZCUDNN_CHNLS, nx_halo, ny_halo]
        size_knl = list(self.recon.DxyMtxC_conv.shape)
        size_knl[0] = OZCUDNN_OUTCHNLS
        size_knl[1] = OZCUDNN_CHNLS
        size_out    = [np,                OZCUDNN_OUTCHNLS,nrx,nry]
        size_out_dp = [np,self.recon.DxyMtxC_conv.shape[0],nrx,nry]
        n = size_img[0]
        c = size_img[1]
        h = size_img[2]
        w = size_img[3]
        k = size_knl[0]
        r = size_knl[2]
        s = size_knl[3]
        size_scl = [1,  k, 1, 1]
        self.tensor_float32 = torch.zeros(size_img, dtype=self.input_type, device=self.device)
        self.vector_float32 = torch.zeros(size_knl, dtype=self.input_type, device=self.device)
        self.y    = torch.empty([n, k, h-(size_knl[2]-1), h-(size_knl[3]-1)], dtype=torch.float, device=self.device, memory_format=torch.channels_last)
        self.y.zero_()
        self.y_dp = torch.empty(size_out_dp, dtype=torch.double, device=self.device)
        self.y_dp.zero_()
        self.x_cudnn_tensor = self.convgraph.tensor_like(self.tensor_float32)
        self.x_cudnn_tensor.set_name("x").set_stride([c * h * w, 1, c * w, c])
        self.w_cudnn_tensor = self.convgraph.tensor_like(self.vector_float32)
        self.w_cudnn_tensor.set_name("w").set_stride([c * r * s, 1, c * s, c])
        self.y_cudnn_tensor = self.convgraph.conv_fprop(
            name="convolution",
            image=self.x_cudnn_tensor,
            weight=self.w_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            padding = [0,0], stride = [1,1], dilation=[1,1]
        )
        self.y_cudnn_tensor_casted = self.convgraph.identity(
            input=self.y_cudnn_tensor, compute_data_type=cudnn.data_type.FLOAT
        )
        self.y_cudnn_tensor_casted.set_name("y").set_output(True).set_data_type(self.output_type)

        self.convgraph.validate()
        self.convgraph.build_operation_graph()
        self.convgraph.create_execution_plans([cudnn.heur_mode.A])
        self.convgraph.check_support()
        self.convgraph.build_plans()
        self.workspace = torch.empty(self.convgraph.get_workspace_size(), device=self.device, dtype=torch.uint8)

        self.tensor_float32 = self.tensor_float32.to(memory_format=torch.channels_last)
        self.vector_float32 = self.vector_float32.to(memory_format=torch.channels_last)
        
        # ############################
        # # For Implicit scheme only #
        # ############################
        # dqLdqrec, \
        # dqRdqrec, \
        # dqBdqrec, \
        # dqTdqrec, \
        # dqQdqrec = self.calc_jacobian_qrec()

        # dqrecdq       = self.calc_jacobian_conv2d(self.recon.Rmtx  ,nvar=nvar)
        # ddphitdxdphit = self.calc_jacobian_conv2d(self.recon.DxMtxC,nvar=1)
        # ddphitdydphit = self.calc_jacobian_conv2d(self.recon.DyMtxC,nvar=1)
        
        # nrow = np*nx_halo*ny_halo
        # ncol = np*nvar*nx_halo*ny_halo
        # I = torch.arange(nrow,dtype=torch.long,device=self.device).view(np,nx_halo,ny_halo).flatten()
        # J = torch.arange(ncol,dtype=torch.long,device=self.device).view(nvar,np,nx_halo,ny_halo)
        # J = J[0,...].flatten()
        # nnz = I.shape[0]
        # V = torch.ones(nnz,device=self.device) / self.mesh.jabCell.flatten()
        # dphitdq = torch.sparse_coo_tensor( torch.stack((I,J),dim=0), V, (nrow,ncol), device=self.device )
        # ddphitdxdq = spspmm( ddphitdxdphit, dphitdq )
        # ddphitdydq = spspmm( ddphitdydphit, dphitdq )

        # dqLdq = spspmm(dqLdqrec, dqrecdq) # (np,nx+1,ny,nPOE,nvar)*(nvar*np*nPOC*nx*ny) * (nvar*np*nPOC*nx*ny)*(nvar*np*nx_halo*ny_halo)
        # dqRdq = spspmm(dqRdqrec, dqrecdq)
        # dqBdq = spspmm(dqBdqrec, dqrecdq)
        # dqTdq = spspmm(dqTdqrec, dqrecdq)
        # dqQdq = spspmm(dqQdqrec, dqrecdq)

        # dqdq = self.fill_ghost_jacobian() # Convert dqfull to dq

        # self.dqLdq = spspmm(dqLdq, dqdq).coalesce() # (np,nx+1,ny,nPOE,nvar)*(nvar*np*nx_halo*ny_halo) * (nvar*np*nx_halo*ny_halo)*(nvar*np*nx*ny)
        # self.dqRdq = spspmm(dqRdq, dqdq).coalesce()
        # self.dqBdq = spspmm(dqBdq, dqdq).coalesce()
        # self.dqTdq = spspmm(dqTdq, dqdq).coalesce()
        # self.dqQdq = spspmm(dqQdq, dqdq).coalesce()
        # self.ddphitdxdq = spspmm(ddphitdxdq, dqdq).coalesce()
        # self.ddphitdydq = spspmm(ddphitdydq, dqdq).coalesce()

        # # print('dqLdq', self.dqLdq.shape, self.dqLdq.values().min(), self.dqLdq.values().max())
        # # print('dqRdq', self.dqRdq.shape, self.dqRdq.values().min(), self.dqRdq.values().max())
        # # print('dqBdq', self.dqBdq.shape, self.dqBdq.values().min(), self.dqBdq.values().max())
        # # print('dqTdq', self.dqTdq.shape, self.dqTdq.values().min(), self.dqTdq.values().max())
        # # print('dqQdq', self.dqQdq.shape, self.dqQdq.values().min(), self.dqQdq.values().max())

        # self.dfxdq_idx_mtx, self.dfydq_idx_mtx = self.calc_f_derivative_idx_matrix()
        # self.dfxdq_idx_mtx = self.dfxdq_idx_mtx / self.mesh.dx
        # self.dfydq_idx_mtx = self.dfydq_idx_mtx / self.mesh.dy

        # # Calculate standard conversion matrix and inverse matrix
        # jab = self.mesh.jab[:,ids:ide,jds:jde]
        # sqrt_iG11 = torch.sqrt( self.mesh.iG[0,0,:,ids:ide,jds:jde] )
        # sqrt_iG22 = torch.sqrt( self.mesh.iG[1,1,:,ids:ide,jds:jde] )

        # h_avg = torch.sqrt( q0[0,...,ids:ide,jds:jde].sum() / jab.sum() ) # phase speed of gravity wave, total mass conservation <-> dmass/dt=0
        
        # nrow = nvar*np*nx*ny
        # ncol = nrow
        # I = torch.arange(nvar*np*nx*ny,device=self.device)
        # J = torch.arange(nvar*np*nx*ny,device=self.device)
        # V = torch.ones(nvar,np,nx,ny,device=self.device)
        # V[0,...] = V[0,...] / jab             / h_avg**2
        # V[1,...] = V[1,...] / jab / sqrt_iG11 / h_avg**3
        # V[2,...] = V[2,...] / jab / sqrt_iG22 / h_avg**3
        # V = V.flatten()
        # IJ = torch.stack((I,J),dim=0)
        # self.Cmtx = torch.sparse_coo_tensor(IJ,V,(nrow,ncol),device=self.device)
        # V = torch.ones(nvar,np,nx,ny,device=self.device)
        # V[0,...] = V[0,...] * jab             * h_avg**2
        # V[1,...] = V[1,...] * jab * sqrt_iG11 * h_avg**3
        # V[2,...] = V[2,...] * jab * sqrt_iG22 * h_avg**3
        # V = V.flatten()
        # self.iCmtx = torch.sparse_coo_tensor(IJ,V,(nrow,ncol),device=self.device)
        
    #@torch.compile
    def unify_panel_bdy_flux(self,qL,qR,qB,qT):
        # nvar, npanel, nPOE, nx, ny
        # ! Panel 1
        qL[...,0,:, 0, :] = qL[...,3,:,-1, :] #! Left
        qR[...,0,:,-1, :] = qR[...,1,:, 0, :] #! Right
        qB[...,0,:, :, 0] = qB[...,5,:, :,-1] #! below
        qT[...,0,:, :,-1] = qT[...,4,:, :, 0] #! over
        # ! Panel 2
        qL[...,1,:, 0, :] = qL[...,0,:,-1,:] #! Left
        qR[...,1,:,-1, :] = qR[...,2,:, 0,:] #! Right
        qB[...,1,:, :, 0] = qL[...,5,:,-1,:].flip(-2,-1) #! below
        qT[...,1,:, :,-1] = qL[...,4,:,-1,:] #! over
        # ! Panel 3
        qL[...,2,:, 0, :] = qL[...,1,:,-1, :] #! Left
        qR[...,2,:,-1, :] = qR[...,3,:, 0, :] #! Right
        qB[...,2,:, :, 0] = qT[...,5,:,: , 0].flip(-2,-1) #! below
        qT[...,2,:, :,-1] = qB[...,4,:,: ,-1].flip(-2,-1) #! over
        # ! Panel 4
        qL[...,3,:, 0, :] = qL[...,2,:,-1,:] #! Left
        qR[...,3,:,-1, :] = qR[...,0,:, 0,:] #! Right
        qB[...,3,:, :, 0] = qR[...,5,:, 0,:] #! below
        qT[...,3,:, :,-1] = qR[...,4,:, 0,:].flip(-2,-1) #! over
        # ! Panel 5
        qL[...,4,:, 0, :] = qB[...,3,:,:,-1].flip(-2,-1) #! Left
        qR[...,4,:,-1, :] = qB[...,1,:,:,-1] #! Right
        qB[...,4,:, :, 0] = qB[...,0,:,:,-1] #! below
        qT[...,4,:, :,-1] = qB[...,2,:,:,-1].flip(-2,-1) #! over
        # ! Panel 6
        qL[...,5,:, 0, :] = qT[...,3,:,:,0] #! Left
        qR[...,5,:,-1, :] = qT[...,1,:,:,0].flip(-2,-1) #! Right
        qB[...,5,:, :, 0] = qT[...,2,:,:,0].flip(-2,-1) #! below
        qT[...,5,:, :,-1] = qT[...,0,:,:,0] #! over

        qL[..., 0, :] = self.convert_field_across_panel(qL[..., 0, :], self.mesh.jabL_bdy_cvt, self.mesh.AL_bdy_cvt)
        qR[...,-1, :] = self.convert_field_across_panel(qR[...,-1, :], self.mesh.jabR_bdy_cvt, self.mesh.AR_bdy_cvt)
        qB[..., :, 0] = self.convert_field_across_panel(qB[..., :, 0], self.mesh.jabB_bdy_cvt, self.mesh.AB_bdy_cvt)
        qT[..., :,-1] = self.convert_field_across_panel(qT[..., :,-1], self.mesh.jabT_bdy_cvt, self.mesh.AT_bdy_cvt)

        return qL, qR, qB, qT
    
    def unify_panel_bdy_flux_by_q_edge(self,q_edge):
        
        np, nPOE, nx, ny = self.mesh.npanel, self.mesh.nPOE, self.mesh.nrx, self.mesh.nry

        pls = 0 * nPOE
        ple = 1 * nPOE
        prs = 1 * nPOE
        pre = 2 * nPOE
        pbs = 2 * nPOE
        pbe = 3 * nPOE
        pts = 3 * nPOE
        pte = 4 * nPOE

        nvar = self.nvar
        
        qL = torch.zeros(nvar,np,nPOE,2 ,ny,device=self.device)
        qR = torch.zeros(nvar,np,nPOE,2 ,ny,device=self.device)
        qB = torch.zeros(nvar,np,nPOE,nx,2 ,device=self.device)
        qT = torch.zeros(nvar,np,nPOE,nx,2 ,device=self.device)

        qL[...,-1,:] = q_edge[..., pls:ple, :]
        qR[..., 0,:] = q_edge[..., prs:pre, :]
        qB[...,:,-1] = q_edge[..., pbs:pbe, :]
        qT[...,:, 0] = q_edge[..., pts:pte, :]

        qL, qR, qB, qT = self.unify_panel_bdy_flux(qL,qR,qB,qT)

        q_edge_unified = torch.zeros_like(q_edge)
        q_edge_unified[..., pls:ple, :] = qL[..., 0,:]
        q_edge_unified[..., prs:pre, :] = qR[...,-1,:]
        q_edge_unified[..., pbs:pbe, :] = qB[...,:, 0]
        q_edge_unified[..., pts:pte, :] = qT[...,:,-1]

        return q_edge_unified
    
    def panel_bdy_flux_correction(self,qL,qR,qB,qT):
        nvar = self.nvar
        np, nPOE, nEOC, nx, ny = self.mesh.npanel, self.mesh.nPOE, self.mesh.nEOC, self.mesh.nrx, self.mesh.nry

        pls = 0 * nPOE
        ple = 1 * nPOE
        prs = 1 * nPOE
        pre = 2 * nPOE
        pbs = 2 * nPOE
        pbe = 3 * nPOE
        pts = 3 * nPOE
        pte = 4 * nPOE

        q_edge = torch.zeros(nvar,np,nPOE*nEOC,nx,device=self.device)

        q_edge[..., pls:ple, :] = qL[...,-1,:]
        q_edge[..., prs:pre, :] = qR[..., 0,:]
        q_edge[..., pbs:pbe, :] = qB[...,:,-1]
        q_edge[..., pts:pte, :] = qT[...,:, 0]

        q_panel_bdy = self.unify_panel_bdy_flux_matrix.matmul(q_edge.flatten()).view(nvar,np,nPOE*nEOC,nx)

        qL[..., 0,:] = q_panel_bdy[..., pls:ple, :]
        qR[...,-1,:] = q_panel_bdy[..., prs:pre, :]
        qB[...,:, 0] = q_panel_bdy[..., pbs:pbe, :]
        qT[...,:,-1] = q_panel_bdy[..., pts:pte, :]
        
        return qL, qR, qB, qT
    
    # #@torch.compile
    def fill_ghost(self,q):
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        np = self.mesh.npanel
        nvar = self.nvar
        n_gst_cell_per_panel = self.mesh.n_gst_cell_per_panel
        
        q[...,self.mesh.igs,self.mesh.jgs] = self.gst_mtx.matmul(q[...,ids:ide,jds:jde].flatten()).view(nvar,np,n_gst_cell_per_panel)

        return q
    
    # #@torch.compile
    # def fill_ghost(self,q):
    #     q_prev = q[ 0, :, self.mesh.igs,self.mesh.jgs ]#.clone()
    #     # q_prev = torch.tensor( 1.e5, device=self.device )
    #     # q_prev = torch.zeros_like( q[ 0, :, self.mesh.igs,self.mesh.jgs ], device=self.device )
    #     max_iter = 100
    #     iter = 0
    #     q_diff = torch.tensor( 10., device=self.device )
    #     while q_diff>self.bdy_tolerance and iter<max_iter:
    #         # Ghost points
    #         qrec = self.recon_ghost(q)

    #         # Replace source cell value
    #         q[...,self.mesh.igs,self.mesh.jgs] = qrec

    #         q_diff = torch.max( torch.abs( q_prev - qrec[0,...] ) / q_prev )

    #         q_prev = qrec[0,...]

    #         iter += 1
    #     # print('iter',iter)
    #     return q
    
    #@torch.compile
    def recon_ghost(self,q):
        qrec = torch.sum( q[...,self.mesh.gst_pr,self.mesh.gst_ir,self.mesh.gst_jr] * self.recon.Rmtx_gst, dim=-1 ) \
                        .view( self.nvar, self.mesh.n_gst_pts )

        qrec = self.convert_field_across_panel(qrec,self.mesh.jabG_cvt,self.mesh.AG_cvt)

        qolp = torch.sum( q[...,self.mesh.gst_pr2,self.mesh.gst_ir2,self.mesh.gst_jr2] * self.recon.Rmtx_olp, dim=-1 ) \
                        .view( self.nvar, 2, self.mesh.n_olp_pts )

        qolp = self.convert_field_across_panel(qolp,self.mesh.jabG2_cvt,self.mesh.AG2_cvt)
        qolp = torch.mean( qolp, dim=1 )

        # Replace recon result on overlap points
        qrec[...,self.mesh.olp_idx] = qolp

        # Gaussian quadrature in source cells
        qrec = qrec.view( self.nvar, self.mesh.npanel_local, self.mesh.n_out_dom_cell_per_panel, self.mesh.nQOC )
        qrec = torch.sum( qrec * self.mesh.gw2d, dim=-1 )
        return qrec
    
    def convert_field_across_panel(self,q,jab_cvt,A_cvt):
        q[1:3,...] = torch.sum( A_cvt * q[1:3,...], dim=1 )
        q = q * jab_cvt
        return q
        
    # torch.compiler.allow_in_graph(gp.start)
    # torch.compiler.allow_in_graph(gp.stop)

    def oz_conv(self, A, C):
        outbs = C.shape[0]
        knlsize = [C.shape[2], C.shape[3]]
        torch.ops.my_ops.custom_ozcudnn(A, C, self.y_dp, self.tensor_float32, self.d_y, torch.tensor([OZ_MODE],dtype=torch.int))
        d_yreshp = self.d_y[:numsplit*outbs*16*knlsize[0]*knlsize[1]].reshape([numsplit*outbs,16,knlsize[0],knlsize[1]])
        curr_ind = 0
        for i in range(numsplit):
            self.vector_float32[outbs*i:outbs*(i+1),curr_ind:curr_ind+(i+1),:,:] = d_yreshp[outbs*i:outbs*(i+1),curr_ind:curr_ind+(i+1),:,:]
        self.vector_float32[outbs*numsplit:2*outbs*numsplit, numsplit:2*numsplit,:,:] = self.vector_float32[:outbs*numsplit, 0:numsplit,:,:]
        variant_pack = {
            self.x_cudnn_tensor: self.tensor_float32,
            self.w_cudnn_tensor: self.vector_float32,
            self.y_cudnn_tensor_casted: self.y,
        }
        self.convgraph.execute(variant_pack, self.workspace)
        B = torch.ops.my_ops.custom_accumulate_ozcudnn(A, C, self.y_dp, self.d_x, self.d_y, self.y, torch.tensor([OZ_MODE],dtype=torch.int))
        Brshp = B.reshape([B.shape[0],B.shape[2],B.shape[3],outbs])
        B = Brshp.permute([0,3,1,2])
        return B

    #@torch.compile
    def spatial_operator(self,q):
        nvar    = self.nvar
        np      = self.mesh.npanel
        nx_halo = self.mesh.nx_halo
        ny_halo = self.mesh.ny_halo
        nx      = self.mesh.nx
        ny      = self.mesh.ny
        nrx     = self.mesh.nrx
        nry     = self.mesh.nry
        nPOE    = self.mesh.nPOE
        nPOR    = self.mesh.nPOR
        nQOC    = self.mesh.nQOC
        rw      = self.mesh.rw
        pc  = self.mesh.pc
        pls = self.mesh.pls
        ple = self.mesh.ple
        prs = self.mesh.prs
        pre = self.mesh.pre
        pbs = self.mesh.pbs
        pbe = self.mesh.pbe
        pts = self.mesh.pts
        pte = self.mesh.pte
        pqs = self.mesh.pqs
        pqe = self.mesh.pqe
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        irs = self.mesh.irs
        ire = self.mesh.ire
        jrs = self.mesh.jrs
        jre = self.mesh.jre
        
        # Reconstruction
        torch.cuda.synchronize()
        gp.start("recon_ght")
        phit = q[0,...] / self.mesh.jabCell

        if self.prec_mode == 0:
            variant_pack = {
                self.x_cudnn_tensor64: phit.view(np,1,nx_halo,ny_halo),
                self.w_cudnn_tensor64: self.recon.DxyMtxC_conv,
                self.y_cudnn_tensor64: self.y_dp64,
            }
            self.convgraph64.execute(variant_pack, self.workspace64)
            dphitdxy = self.y_dp64.reshape([np,self.y_dp64.shape[2],self.y_dp64.shape[3],self.recon.DxyMtxC_conv.shape[0]]).permute([0,3,1,2])
        elif self.prec_mode == 1:
            phit1 = phit.view(np,1,nx_halo,ny_halo)
            phit1 = phit1 / 2**16
            DxyMtxQ1 = self.recon.DxyMtxC_conv / 500
            dphitdxy = self.oz_conv(phit1,DxyMtxQ1)
            dphitdxy = dphitdxy * 500*2**16/  4 **(bits_per_slice-7)
        
        self.dphitdx = dphitdxy[:,0,:,:].view(np,nrx,nry)
        self.dphitdy = dphitdxy[:,1,:,:].view(np,nrx,nry)
        torch.cuda.synchronize()
        gp.stop("recon_ght")

        torch.cuda.synchronize()
        gp.start("recon_q")
        q = q.view(nvar*np,1,nx_halo,ny_halo)
        qrec = self.recon(q).view(nvar,np,nPOR,nrx,nry)
        torch.cuda.synchronize()
        gp.stop("recon_q")

        torch.cuda.synchronize()
        gp.start("fill_bdy_q")
        qC = self.qC
        qL = self.qL
        qR = self.qR
        qB = self.qB
        qT = self.qT
        
        qC                = qrec[...,pc     ,:,:]
        qL[...,1:  ,:   ] = qrec[...,prs:pre,rw:-rw,:] # nvar, npanel, npts, nrx, nry
        qR[...,0:-1,:   ] = qrec[...,pls:ple,rw:-rw,:]
        qB[...,:   ,1:  ] = qrec[...,pts:pte,:,rw:-rw]
        qT[...,:   ,0:-1] = qrec[...,pbs:pbe,:,rw:-rw]
        
        # qL, qR, qB, qT = self.unify_panel_bdy_flux(qL, qR, qB, qT)
        qL, qR, qB, qT = self.panel_bdy_flux_correction(qL, qR, qB, qT)
        torch.cuda.synchronize()
        gp.stop("fill_bdy_q")

        torch.cuda.synchronize()
        gp.start("riemann_solver")
        fluxL = self.riemann_solver(qL,qR,self.jabghsL,self.mesh.jabL,self.JiGL,self.convert_coefL,dir=0)
        fluxB = self.riemann_solver(qB,qT,self.jabghsB,self.mesh.jabB,self.JiGB,self.convert_coefB,dir=1).transpose(-2,-1)

        fluxL = torch.nn.functional.conv1d(fluxL.reshape(nvar*np*(nx+1),1,fluxL.shape[3]), self.recon.conv_edge) \
                .squeeze().view(nvar,np,nx+1,ny)
        fluxB = torch.nn.functional.conv1d(fluxB.reshape(nvar*np*(ny+1),1,fluxB.shape[3]), self.recon.conv_edge) \
                .squeeze().view(nvar,np,ny+1,nx).transpose(-2,-1)
        torch.cuda.synchronize()
        gp.stop("riemann_solver")

        torch.cuda.synchronize()
        gp.start("calc_src")
        src = self.calc_src(qC,self.dphitdx,self.dphitdy)
        src = torch.nn.functional.conv2d(src.view(nvar*np,1,nrx,nry), self.recon.conv_cell).squeeze().view(nvar,np,nx,ny)
        torch.cuda.synchronize()
        gp.stop("calc_src")

        torch.cuda.synchronize()
        gp.start("sum_tend")
        tend_x = ( fluxL[...,0:-1,:] - fluxL[...,1:,:] ) * self.mesh.inv_dx
        tend_y = ( fluxB[...,:,0:-1] - fluxB[...,:,1:] ) * self.mesh.inv_dy
        tend = tend_x + tend_y + src
        torch.cuda.synchronize()
        gp.stop("sum_tend")
        
        # # Storage for implicit time marching
        # self.qC = qC.clone()
        # self.qL = qL.clone()
        # self.qR = qR.clone()
        # self.qB = qB.clone()
        # self.qT = qT.clone()

        return tend
    
    #@torch.compile
    def calc_src(self,q,dphitdx,dphitdy):
        # nvar, np, nx, ny = q.shape
        nvar = self.nvar
        np   = self.mesh.npanel_local
        nx   = self.mesh.nrx
        ny   = self.mesh.nry

        jabghsQ = self.jabghsQ
        iGQ_C   = self.iGQ_C  
        iGQ_S   = self.iGQ_S  
        coef_M  = self.coef_M

        q2q2 = q[1,...] * q[1,...]
        q2q3 = q[1,...] * q[2,...]
        q3q3 = q[2,...] * q[2,...]
        
        psi = torch.zeros( 3, nvar, np, nx, ny, device=self.device )
        psi[0,1,...] = coef_M[0,0,...] * q2q2 + coef_M[0,1,...] * q2q3
        psi[0,2,...] = coef_M[1,0,...] * q2q3 + coef_M[1,1,...] * q3q3
        psi[0,1:3,...] = psi[0,1:3,...].clone() / ( q[0,...] - jabghsQ )

        psi[1,1:3,...] = iGQ_C[0:2,0,...] * q[2,...] \
                       - iGQ_C[0:2,1,...] * q[1,...]

        psi[2,1:3,...] = iGQ_S[0:2,0,...] * dphitdx \
                       + iGQ_S[0:2,1,...] * dphitdy 
        
        src = torch.sum( psi, dim=0 )
        return src
    
    # @torch.compile
    def riemann_solver(self,qL,qR,jabghs,jab,JiG,convert_coef,dir):
        # qL = qL.clone() # Avoid modifying the original data
        # qR = qR.clone() # Avoid modifying the original data

        ghtL   = qL[0,...] / jab
        ghtR   = qR[0,...] / jab
        jabghL = qL[0,...] - jabghs
        jabghR = qR[0,...] - jabghs

        # Convert wind to perpendicular to the cell edges
        uL = qL[dir+1,...] / ( jabghL * convert_coef )
        uR = qR[dir+1,...] / ( jabghR * convert_coef )
        # uL = qL[dir+1,...].clone() / ( jabghL * convert_coef )
        # uR = qR[dir+1,...].clone() / ( jabghR * convert_coef )

        # m, p = self.LMARS(uL,uR,ghtL,ghtR)
        m, p = self.AUSM(uL,uR,ghtL,ghtR)
        m = m * convert_coef
        # s = torch.sign(m)

        qL[0,...] = qL[0,...] - jabghs # Reset mass flux from phitu to phiu
        qR[0,...] = qR[0,...] - jabghs # Reset mass flux from phitu to phiu

        flux_pts = 0.5 * m * ( qL + qR - torch.sign(m) * ( qR - qL ) )
        flux_pts[1:3,...] = flux_pts[1:3,...] + JiG[0:2,...] * p
        
        # flux = flux_pts.squeeze()
        return flux_pts.squeeze()

    # @torch.compile
    def LMARS(self,uL,uR,phiL,phiR):
        # cL = torch.sqrt( phiL )
        # cR = torch.sqrt( phiR )
        # c = 0.5 * ( cL + cR )

        c = 0.5 * ( torch.sqrt( phiL ) + torch.sqrt( phiR ) )

        phi = 0.5 * ( phiL + phiR - c * ( uR - uL ) )
        m   = 0.5 * ( uL + uR - ( phiR - phiL ) / c )
        p   = 0.5 * phi * phi
        return m, p

    #@torch.compile
    def AUSM(self,uL,uR,phiL,phiR):
        Ku = 0.75
        Kp = 0.25
        sigma = 1.

        cL = torch.sqrt( phiL )
        cR = torch.sqrt( phiR )

        c = 0.5 * ( cL + cR )
        c2 = c**2
        
        phi = 0.5 * ( phiL + phiR )

        pL = 0.5 * phiL * phiL
        pR = 0.5 * phiR * phiR

        ML = uL / c
        MR = uR / c

        Mbar2 = ( uL**2 + uR**2 ) / ( 2 * c2 )

        def M2(M,s):
            return s * 0.25 * ( M + s )**2
        
        def M4(M,s):
            beta  = 0.125
            M_sign = torch.sign( M.abs() - 1. )
            M_supersonic = 0.5 * ( M + s * M.abs() )
            M_subsonic = M2(M,s) * ( 1. - s * 16. * beta * M2(M,-s) )
            return 0.5 * ( M_subsonic + M_supersonic - M_sign * ( M_subsonic - M_supersonic ) )
        
        zero = torch.zeros_like(Mbar2,device=self.device)
        compare = torch.max( torch.stack( [1.-sigma*Mbar2, zero], dim=-1 ), dim=-1 )[0]
        Mh = M4(ML,1) + M4(MR,-1) - Kp * compare * (pR - pL) / ( phi * c2 )
        m = c * Mh

        def P5(M,s):
            alpha = 0.1875
            M_sign = torch.sign( M.abs() - 1. )
            M_supersonic = 0.5 * ( 1. + s * torch.sign( M ) )
            M_subsonic = M2(M,s) * ( ( 2. * s - M ) - s * 16. * alpha * M * M2( M, -s ) )
            return 0.5 * ( M_subsonic + M_supersonic - M_sign * ( M_subsonic - M_supersonic ) )
        
        P5L = P5(ML,1)
        P5R = P5(MR,-1)
        p = P5L * pL + P5R * pR - Ku * P5L * P5R * 2 * phi * c * ( uR - uL )

        return m, p
    
    #@torch.compile
    def temporal_operator(self,q0):

        q = q0.clone()
        # q = torch.zeros_like(q0,device=self.device)
        # q[ ..., self.mesh.igs,self.mesh.jgs ] = q0[ ..., self.mesh.igs,self.mesh.jgs ]

        dq = self.spatial_operator(q0)
        q = self.update_state(q,q0,dq,self.dt/3.)

        dq = self.spatial_operator(q)
        q = self.update_state(q,q0,dq,self.dt/2.)

        dq = self.spatial_operator(q)
        q = self.update_state(q,q0,dq,self.dt)

        return q
    
    #@torch.compile
    def update_state(self,q,q0,dq,dt):
        torch.cuda.synchronize()
        gp.start("update_state")
        
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        
        q[...,ids:ide,jds:jde] = q0[...,ids:ide,jds:jde] + dt * dq
        q = self.fill_ghost(q)

        torch.cuda.synchronize()
        gp.stop("update_state")
        return q
    
    def forward(self,q0):
        q = self.spatial_operator(q0)
        return q
    
    def get_operator_jacobian(self,operator,q,varname,nrow_per_block=100):
        print('')
        print('generate jacobian of '+varname+' start')

        dtype = self.gw.dtype
        
        mtx_completed = 0
        completed_idx = 0
        file_exists = os.path.exists(self.mtx_file)
        if file_exists:
            var_exists = netcdf_check_variable_exists(self.mtx_file,varname)
            if var_exists:
                completed_idx, mtx_completed = netcdf_write_ghost_interp_matrix_check(self.mtx_file,varname)

        if not mtx_completed:
            if not file_exists:
                netcdf_write_ghost_interp_matrix_init(self.mtx_file,varname,'w')
            else:
                if not var_exists:
                    netcdf_write_ghost_interp_matrix_init(self.mtx_file,varname,'a')

            ncol = q.nelement()

            q_out = operator(q)
            q_out_shape = q_out.shape
            nrow = q_out.nelement()

            (_, get_vjp) = torch.func.vjp( operator, q )
            get_vjp = torch.vmap( get_vjp )

            nblock = math.ceil( (nrow-completed_idx) / nrow_per_block )
            print( 'nrow_per_block, nrow, nrow%nrow_per_block ',nrow_per_block, nrow, nrow%nrow_per_block )
            for iblock in range(nblock):
                rows = iblock * nrow_per_block + completed_idx
                rowe = min( [( iblock + 1 ) * nrow_per_block - 1 + completed_idx, nrow-1] )
                nrow_pres = rowe - rows + 1

                idx = torch.linspace(rows,rowe,nrow_pres,dtype=torch.int)
                qe = torch.zeros(nrow_pres,nrow,device=self.device)
                qe[idx-rows,idx] = 1
                qe = qe.view(nrow_pres,*q_out_shape)

                tmp = get_vjp(qe)[0]

                # Remove values too close to zero
                sign = torch.sign( tmp.abs() - self.bdy_tolerance )
                tmp = 0.5 * ( tmp + sign * tmp )

                tmp = tmp.view(nrow_pres,ncol).to_sparse_csr()
                netcdf_write_ghost_interp_matrix_append(self.mtx_file,varname,tmp)
                print(iblock,'/',nblock-1,'inc nnz',tmp._nnz())
                # print('rows,rowe,nrow_pres',rows,rowe,nrow_pres)
                del tmp, qe
            if iblock==nblock-1:
                netcdf_write_ghost_interp_matrix_final(self.mtx_file,varname)

        Jv = netcdf_read_ghost_interp_matrix(self.mtx_file,varname,dtype,self.device)
        print( varname+' matrix nnz     ',Jv._nnz() )
        print( varname+' matrix min     ',Jv.values().min() )
        print( varname+' matrix max     ',Jv.values().max() )
        print( varname+' matrix max(abs)',Jv.values().abs().max() )
        print( varname+' matrix min(abs)',Jv.values().abs().min() )
        print('generate jacobian of '+varname+' end')
        print('')

        return Jv
    
    def fill_ghost_out_domain(self,q0):
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        nx_halo = self.mesh.nx_halo # ime - ims
        ny_halo = self.mesh.ny_halo # jme - jms
        np = self.mesh.npanel
        nvar = self.nvar
        
        q = torch.zeros(nvar,np,nx_halo,ny_halo,device=self.device)
        q[...,ids:ide,jds:jde] = q0

        max_iter = 25
        iter = 0 
        while iter<max_iter:
            # Ghost points
            qrec = self.recon_ghost(q)
            # Replace source cell value
            q[...,self.mesh.igs,self.mesh.jgs] = qrec
            iter += 1
        return qrec
    
    #@torch.compile
    def CSR_combine(self,Jv,tmp):
        # CSR
        nnz_Jv = Jv._nnz()
        nnz_tmp = tmp._nnz()
        nnz = nnz_Jv + nnz_tmp

        row_Jv = Jv.crow_indices()
        row_tmp = tmp.crow_indices() + nnz_Jv

        nrow_idx_Jv = row_Jv.shape[0]
        nrow_idx_tmp = row_tmp.shape[0]
        nrow_idx = nrow_idx_Jv + nrow_idx_tmp - 1
        
        row = torch.zeros(nrow_idx,dtype=torch.int64,device=self.device)
        row[:nrow_idx_Jv-1] = row_Jv[:-1]
        row[-nrow_idx_tmp:] = row_tmp

        col = torch.zeros(nnz,dtype=torch.int64,device=self.device)
        col[:nnz_Jv] = Jv.col_indices()
        col[-nnz_tmp:] = tmp.col_indices()

        val = torch.zeros(nnz,device=self.device)
        val[:nnz_Jv] = Jv.values()
        val[-nnz_tmp:] = tmp.values()
        
        Jv = torch.sparse_csr_tensor(row, col, val, device=self.device)
        return Jv

    def implicit_time_marching(self,q0):
        max_iter = 10000
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        nx  = self.mesh.nx # ide - ids
        ny  = self.mesh.ny # jde - jds
        np  = self.mesh.npanel
        nvar = self.nvar

        q0 = self.fill_ghost(q0.clone())
        q = q0.clone()
        for iter in range(max_iter):
            # q_new, dq = self.linear_approximation(q,q0,self.dt)
            # res = torch.norm(dq)
            # q[...,ids:ide,jds:jde] = q_new
            # q = self.fill_ghost(q)

            q, dq = self.NONTM(q,q0,self.dt)
            res = torch.norm(dq)

            print('nonlinear iter, residual:',iter,res)
            if res<3.e-4:
                break
            
        # tend = self.spatial_operator(q)
        # q[...,ids:ide,jds:jde] = q0[...,ids:ide,jds:jde] + self.dt * tend

        # q0 = q0.clone()
        # q  = q0.clone()
        # nstage = 3
        # tend = torch.zeros(nstage,nvar,np,nx,ny,device=self.device)
        # a = torch.zeros(nstage,nstage,device=self.device)
        # b = torch.zeros(nstage,device=self.device)
        # alpha = 2 * math.cos(math.pi/18.)/math.sqrt(3.)
        # gm = (1+alpha)*0.5
        # a[0,0] = gm
        # a[1,0] = -0.5*alpha
        # a[1,1] = gm
        # a[2,0] = 1+alpha
        # a[2,1] = -(1+2*alpha)
        # a[2,2] = gm
        # b[0] = 1/(6*alpha**2); b[1] = 1-1/(3*alpha**2); b[2] = 1/(6*alpha**2)
        # a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # b = b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # for istage in range(nstage):
        #     for iter in range(max_iter):
        #         q = self.fill_ghost(q)
        #         dt = a[istage,istage,...].squeeze() * self.dt
        #         q, dq = self.NONTM(q,q0,dt)
        #         res = torch.norm(dq)
        #         print('istage, nonlinear iter, residual:',istage,iter,res)
        #         if res<3.e-4:
        #             break
        #     tend[istage,...] = self.spatial_operator(q)
        #     q[...,ids:ide,jds:jde] = q0[...,ids:ide,jds:jde] + self.dt * torch.sum(a[istage,:istage+1,...]*tend[:istage+1,...],dim=0)
        
        # q[...,ids:ide,jds:jde] = q0[...,ids:ide,jds:jde] + self.dt * torch.sum(b*tend,dim=0)

        return q
    
    # Solver accoring to Li X. et al., 
    # A NINTH-ORDER NEWTON-TYPE METHOD TO SOLVE SYSTEMS OF NONLINEAR EQUATIONS
    # IJRRAS 16 (2), August 2013
    def NONTM(self,q,q0,dt):
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde

        nvar, np, nx_halo, ny_halo = q0.shape
        nx, ny = self.mesh.nx, self.mesh.ny
    
        qn = q0.clone()
        x  = q.clone()
        q  = torch.zeros_like(q0,device=self.device)

        # Stage 1
        tend = self.spatial_operator(x)
        A, _ = self.generate_jacobian(dt)
        b = self.calculate_rhs_b(x,qn,tend,dt)
        dq = torch.sparse.spsolve( A, b ).view(nvar,np,nx,ny)
        x = torch.matmul( self.Cmtx, x[...,ids:ide,jds:jde].flatten() ).view(nvar,np,nx,ny)
        y = x + dq
        y = torch.matmul( self.iCmtx, y.flatten() ).view(nvar,np,nx,ny)
        q[...,ids:ide,jds:jde] = y
        y = self.fill_ghost(q)

        # Stage 2
        tend = self.spatial_operator(y)
        b = self.calculate_rhs_b(y,qn,tend,dt)
        dq = torch.sparse.spsolve( A, b ).view(nvar,np,nx,ny)
        y = torch.matmul( self.Cmtx, y[...,ids:ide,jds:jde].flatten() ).view(nvar,np,nx,ny)
        z = y + dq
        z = torch.matmul( self.iCmtx, z.flatten() ).view(nvar,np,nx,ny)
        q[...,ids:ide,jds:jde] = z
        z = self.fill_ghost(q)

        # Stage 3
        tend = self.spatial_operator(z)
        A, _ = self.generate_jacobian(dt)
        b = self.calculate_rhs_b(z,qn,tend,dt)
        dq = torch.sparse.spsolve( A, b ).view(nvar,np,nx,ny)
        z = torch.matmul( self.Cmtx, z[...,ids:ide,jds:jde].flatten() ).view(nvar,np,nx,ny)
        m = z + dq
        m = torch.matmul( self.iCmtx, m.flatten() ).view(nvar,np,nx,ny)
        q[...,ids:ide,jds:jde] = m
        m = self.fill_ghost(q)

        # Stage 4
        tend = self.spatial_operator(m)
        b = self.calculate_rhs_b(m,qn,tend,dt)
        dq = torch.sparse.spsolve( A, b ).view(nvar,np,nx,ny)
        m = torch.matmul( self.Cmtx, m[...,ids:ide,jds:jde].flatten() ).view(nvar,np,nx,ny)
        p = m + dq
        p = torch.matmul( self.iCmtx, p.flatten() ).view(nvar,np,nx,ny)
        q[...,ids:ide,jds:jde] = p
        q = self.fill_ghost(q)

        return q, dq
    
    def RKRW(self,q):
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        nx  = self.mesh.nx

        nvar, np, nx_halo, ny_halo = q.shape
        nx, ny = self.mesh.nx, self.mesh.ny
    
        qn = q.clone()
        q  = q.clone()

        dt = self.dt

        nstage = 3
        # ROS3P
        gm = 7.886751345948129e-01
        a = torch.zeros(nstage,nstage,device=self.device)
        a[1,0] = 1.267949192431123
        a[2,0] = 1.267949192431123
        c = torch.zeros(nstage,nstage,device=self.device)
        c[1,0] = -1.607695154586736
        c[2,0] = -3.464101615137755
        c[2,1] = -1.732050807568877
        m = torch.zeros(nstage,device=self.device)
        m[0] = 2
        m[1] = 5.773502691896258e-01
        m[2] = 4.226497308103742e-01

        # nstage = 4
        # # RODAS3 based on F. Bassi et al. 2015, Computers & Fluids 118, 305-320
        # gm = 0.5
        # a = torch.zeros(nstage,nstage,device=self.device)
        # a[1,0] = 0
        # a[2,0] = 2; a[2,1] = 0
        # a[3,0] = 2; a[3,1] = 0; a[3,2] = 1
        # c = torch.zeros(nstage,nstage,device=self.device)
        # c[1,0] = 4
        # c[2,0] = 1; c[2,1] =-1
        # c[3,0] = 1; c[3,1] =-1; c[3,2] =-8/3
        # m = torch.zeros(nstage,device=self.device)
        # m[0] = 2
        # m[1] = 0
        # m[2] = 1
        # m[3] = 1

        # nstage = 4
        # # ROS4 based on F. Bassi et al. 2015, Computers & Fluids 118, 305-320
        # gm = 0.5
        # a = torch.zeros(nstage,nstage,device=self.device)
        # a[1,0] = 2
        # a[2,0] = 48/25; a[2,1] = 6/25
        # a[3,0] = 48/25; a[3,1] = 6/25; a[3,2] = 0
        # c = torch.zeros(nstage,nstage,device=self.device)
        # c[1,0] =-8
        # c[2,0] =  372/25; c[2,1] = 12/5
        # c[3,0] =-112/125; c[3,1] =-54/125; c[3,2] =-2/5
        # m = torch.zeros(nstage,device=self.device)
        # m[0] = 19/9
        # m[1] = 0.5
        # m[2] = 25/108
        # m[3] = 125/108

        # # RODASP (RO4-6) based on F. Bassi et al. 2015, Computers & Fluids 118, 305-320
        # nstage = 6
        # gm = 0.25
        # a = torch.zeros(nstage,nstage,device=self.device)
        # a[1,0] = 3
        # a[2,0] = 1.831036793486759e+00; a[2,1] = 4.955183967433795e-01
        # a[3,0] = 2.304376582692669e+00; a[3,1] =-5.249275245743001e-02; a[3,2] =-1.176798761832782e+00
        # a[4,0] =-7.170454962423024e+00; a[4,1] =-4.741636671481785e+00; a[4,2] =-1.631002631330971e+01; a[4,3] =-1.062004044111401e+00
        # a[5,0] =-7.170454962423024e+00; a[5,1] =-4.741636671481785e+00; a[5,2] =-1.631002631330971e+01; a[5,3] =-1.062004044111401e+00; a[5,4] = 1
        # c = torch.zeros(nstage,nstage,device=self.device)
        # c[1,0] =-12
        # c[2,0] =-8.791795173947035e+00; c[2,1] =-2.207865586973518e+00
        # c[3,0] = 1.081793056857153e+01; c[3,1] = 6.780270611428266e+00; c[3,2] = 1.953485944642410e+01
        # c[4,0] = 3.419095006749676e+01; c[4,1] = 1.549671153725963e+01; c[4,2] = 5.474760875964130e+01; c[4,3] = 1.416005392148534e+01
        # c[5,0] = 3.462605830930532e+01; c[5,1] = 1.530084976114473e+01; c[5,2] = 5.699955578662667e+01; c[5,3] = 1.840807009793095e+01; c[5,4] =-5.714285714285717e+00
        # m = torch.zeros(nstage,device=self.device)
        # m[0] =-7.170454962423024e+00
        # m[1] =-4.741636671481785e+00
        # m[2] =-1.631002631330971e+01
        # m[3] =-1.062004044111401e+00
        # m[4] = 1
        # m[5] = 1

        # # ROW6A(RO6-6) based on F. Bassi et al. 2015, Computers & Fluids 118, 305-320
        # nstage = 6
        # gm = 3.341423670680504e-01
        # a = torch.zeros(nstage,nstage,device=self.device)
        # a[1,0] = 2
        # a[2,0] = 1.751493065942685e+00; a[2,1] =-1.454290536332865e-01
        # a[3,0] =-1.847093912231436e+00; a[3,1] =-2.513756792158473e+00; a[3,2] = 1.874707432337999e+00
        # a[4,0] = 1.059634783677141e+01; a[4,1] = 1.974951525952609e+00; a[4,2] =-1.905211286263863e+00; a[4,3] =-3.575118228830491e+00
        # a[5,0] = 2.417642067883312e+00; a[5,1] = 3.050984437044573e-01; a[5,2] =-2.346208879122501e-01; a[5,3] =-1.327038464607418e-01; a[5,4] = 3.912922779645768e-02
        # c = torch.zeros(nstage,nstage,device=self.device)
        # c[1,0] =-1.745029492512995e+01
        # c[2,0] =-1.202359936227844e+01; c[2,1] = 1.315910110742745e+00
        # c[3,0] = 2.311230597159272e+01; c[3,1] = 1.297893129565445e+01; c[3,2] =-8.445374594562038e+00
        # c[4,0] =-3.147228891330713e+00; c[4,1] =-1.761332622909965e+00; c[4,2] = 6.115295934038585e+00; c[4,3] = 1.499319950457112e+01
        # c[5,0] =-2.015840911262880e+01; c[5,1] =-1.603923799800133e+00; c[5,2] = 1.155870096920252e+00; c[5,3] = 6.304639815292044e-01; c[5,4] =-1.602510215637174e-01
        # m = torch.zeros(nstage,device=self.device)
        # m[0] = 3.399347452674165e+01
        # m[1] =-2.091829882847333e+01
        # m[2] =-1.375688477471081e+01
        # m[3] =-1.113925929930077e+01
        # m[4] = 2.873406527609468e+00
        # m[5] = 3.876609945620840e+01

        # id = torch.linalg.inv(d)
        # c = ( torch.eye(nstage,dtype=torch.float64,device=self.device) / gm - id ).to(torch.float)
        # a = torch.matmul( a, id ).to(torch.float)
        # m = torch.matmul( m, id ).to(torch.float)

        a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        c = c.unsqueeze(-1)
        m = m.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        torch.cuda.empty_cache()
        
        dq = torch.zeros(nstage,nvar,np,nx,ny,device=self.device)

        q_old = torch.matmul( self.Cmtx, qn[...,ids:ide,jds:jde].flatten() ).view(nvar,np,nx,ny)
        tend = self.spatial_operator(q)
        A, Jv = self.generate_jacobian(gm*dt)

        for istage in range(nstage):
            q_new = q_old + torch.sum( dq[:istage,...] * a[istage,:istage,...], dim=0 )
            q_new = torch.matmul( self.iCmtx, q_new.flatten() ).view(nvar,np,nx,ny)
            q[...,ids:ide,jds:jde] = q_new

            q = self.fill_ghost(q)
            tend = self.spatial_operator(q)
            # A, Jv = self.generate_jacobian(gm*dt)
            b = torch.matmul( self.Cmtx, tend.flatten() ) \
              + torch.sum( dq[:istage,...].flatten(1) * c[istage,:istage,...]/dt, dim=0 )

            # Solve the linear system
            dq[istage,...] = torch.sparse.spsolve( A, b ).view(nvar,np,nx,ny)

        q_new = q_old + torch.sum( dq * m, dim=0 )
        q_new = torch.matmul( self.iCmtx, q_new.flatten() ).view(nvar,np,nx,ny)
        q[...,ids:ide,jds:jde] = q_new
        return q
    
    def RKRW2(self,q):
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        nx  = self.mesh.nx

        nvar, np, nx_halo, ny_halo = q.shape
        nx, ny = self.mesh.nx, self.mesh.ny
    
        qn = q.clone()
        q  = q.clone()

        dt = self.dt

        # nstage = 3
        # gm = 7.8867513459481287e-01
        # a = torch.zeros(nstage,nstage,dtype=torch.float64,device=self.device)
        # a[1,0] = 2.3660254037844388e+00
        # a[2,0] = 0.0000000000000000e+00; a[2,1] = 1.0000000000000000e+00
        # d = torch.zeros(nstage,nstage,dtype=torch.float64,device=self.device)
        # I = torch.arange(nstage,device=self.device)
        # d[I,I] = gm
        # d[1,0] =-2.3660254037844388e+00
        # d[2,0] =-2.8468642516567449e-01; d[2,1] =-1.0813389786187642e+00
        # m = torch.zeros(nstage,dtype=torch.float64,device=self.device)
        # m[0] = 2.9266384402395124e-01
        # m[1] =-8.1338978618764143e-02
        # m[2] = 7.8867513459481287e-01

        # # RODASPR2
        # nstage = 6
        # gm = 3.125e-01
        # a = torch.zeros(nstage,nstage,device=self.device)
        # a[1,0] = 9.3750000000000000e-01
        # a[2,0] =-4.7145892646261345e-02; a[2,1] = 5.4531286650471122e-01
        # a[3,0] = 4.6915543899742240e-01; a[3,1] = 4.4490537602383673e-01; a[3,2] =-2.2498239334061121e-01
        # a[4,0] = 1.0950372887345903e+00; a[4,1] = 6.3223023457294381e-01; a[4,2] =-8.9232966090485821e-01; a[4,3] = 1.6506213759732410e-01
        # a[5,0] =-1.7746585073632790e-01; a[5,1] =-5.8241418952602364e-01; a[5,2] = 6.8180612588238165e-01; a[5,3] = 7.6557391437996980e-01; a[5,4] = 3.1250000000000000e-01
        # d = torch.zeros(nstage,nstage,device=self.device)
        # I = torch.arange(nstage,device=self.device)
        # d[I,I] = gm
        # d[1,0] =-9.3750000000000000e-01
        # d[2,0] =-9.7580572085994507e-02; d[2,1] =-5.8666328499964138e-01
        # d[3,0] =-4.9407065013256957e-01; d[3,1] =-5.6819726428975503e-01; d[3,2] = 5.0318949274167679e-01
        # d[4,0] =-1.2725031394709183e+00; d[4,1] =-1.2146444240989676e+00; d[4,2] = 1.5741357867872399e+00; d[4,3] = 6.0051177678264578e-01
        # d[5,0] = 6.9690744901421153e-01; d[5,1] = 6.2237005730756434e-01; d[5,2] =-1.1553701989197045e+00; d[5,3] = 1.8350029013386296e-01; d[5,4] =-6.5990759753593431e-01
        # m = torch.zeros(nstage,device=self.device)
        # m[0] = 5.1944159827788361e-01
        # m[1] = 3.9955867781540699e-02
        # m[2] =-4.7356407303732290e-01
        # m[3] = 9.4907420451383284e-01
        # m[4] =-3.4740759753593431e-01
        # m[5] = 3.1250000000000000e-01

        # nstage = 6
        # gm = 3.341423670680504e-01
        # a = torch.zeros(nstage,nstage,device=self.device)
        # a[1,0] = 0.66828473413610087e+000
        # a[2,0] = 0.58524803895736580e+000; a[2,1] =-0.48594008221492802e-001
        # a[3,0] =-0.61719233202999775e+000; a[3,1] =-0.83995264476522158e+000; a[3,2] = 0.62641917900148600e+000
        # a[4,0] = 0.35406887484552165e+001; a[4,1] = 0.65991497772646308e+000; a[4,2] =-0.63661180895697222e+000; a[4,3] =-0.11945984675295562e+001
        # a[5,0] = 0.80783664328582613e+000; a[5,1] = 0.10194631616818569e+000; a[5,2] =-0.78396778850607012e-001; a[5,3] =-0.44341977375427388e-001; a[5,4] = 0.13074732797453325e-001
        # d = torch.zeros(nstage,nstage,device=self.device)
        # I = torch.arange(nstage,device=self.device)
        # d[I,I] = gm
        # d[1,0] =-0.58308828523185086e+001
        # d[2,0] =-0.40175939515896193e+001; d[2,1] = 0.43970131925236112e+000
        # d[3,0] = 0.77228006257490299e+001; d[3,1] = 0.43368108251435758e+001; d[3,2] =-0.28219574578033366e+001
        # d[4,0] =-0.10516225114542007e+001; d[4,1] =-0.58853585181331353e+000; d[4,2] = 0.20433794587212771e+001; d[4,3] = 0.50098631723809151e+001
        # d[5,0] =-0.67357785372199458e+001; d[5,1] =-0.53593889506199845e+000; d[5,2] = 0.38622517020810987e+000; d[5,3] = 0.21066472713931598e+000; d[5,4] =-0.53546655670373728e-001
        # m = torch.zeros(nstage,device=self.device)
        # m[0] = 0.11358660043232931e+002
        # m[1] =-0.69896898855829058e+001
        # m[2] =-0.45967580421042947e+001
        # m[3] =-0.37220984696531517e+001
        # m[4] = 0.96012685868421520e+000
        # m[5] = 0.12953396234292936e+002

        # High-order W-method according to 
        # Arunasalam Rahunanthan and Dan Stanescu, Journal of Computational and Applied Mathematics, 2010
        # 6S4O(B)W-method
        nstage = 6
        gm = 0.25
        a = torch.zeros(nstage,nstage,device=self.device)
        a[1,0] = 0.032918605146
        a[2,0] =-0.573905274856; a[2,1] = 0.823256998200
        a[3,0] =-0.114172035574; a[3,1] = 0.199552791730; a[3,2] = 0.381530948900
        a[4,0] =-0.293732375805; a[4,1] = 0.443156103276; a[4,2] = 0.232514473389; a[4,3] = 0.200092213184
        a[5,0] = 1.973193167196; a[5,1] =-2.632303480923; a[5,2] = 2.113827764674; a[5,3] =-2.326045509879; a[5,4] = 1.718581042715
        d = torch.zeros(nstage,nstage,device=self.device)
        I = torch.arange(nstage,device=self.device)
        d[I,I] = gm
        d[1,0] =-0.060566928942
        d[2,0] =-0.317227025111
        d[3,0] =-1.420596433366; d[3,1] = 1.189454931628; d[3,2] =-0.161776149667
        d[4,0] = 0.702148458093; d[4,1] =-1.238114681516; d[4,2] = 0.645532689244; d[4,3] =-0.452154452468
        d[5,0] =-0.292949309489; d[5,1] = 0.238718474557; d[5,2] =-0.384426701068; d[5,3] = 1.461704809920; d[5,4] =-1.271281159685
        m = torch.zeros(nstage,device=self.device)
        m[0] = 0.971001746640
        m[1] =-1.272664996516
        m[2] = 1.282112737366
        m[3] =-1.209258255435
        m[4] = 0.958808767946
        m[5] = 0.27

        a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        d = d.unsqueeze(-1)
        m = m.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        torch.cuda.empty_cache()
        
        dq = torch.zeros(nstage,nvar,np,nx,ny,device=self.device)

        q_old = torch.matmul( self.Cmtx, qn[...,ids:ide,jds:jde].flatten() ).view(nvar,np,nx,ny)
        tend = self.spatial_operator(q)
        A, Jv = self.generate_jacobian(gm*dt)

        for istage in range(nstage):
            q_new = q_old + torch.sum( dq[:istage,...] * a[istage,:istage,...], dim=0 )
            q_new = torch.matmul( self.iCmtx, q_new.flatten() ).view(nvar,np,nx,ny)
            q[...,ids:ide,jds:jde] = q_new

            q = self.fill_ghost(q)
            tend = self.spatial_operator(q)
            dq_sum = torch.sum( dq[:istage,...].flatten(1) * d[istage,:istage,...] / gm, dim=0 )
            Jq = Jv.matmul(dq_sum)
            b = torch.matmul( self.Cmtx, tend.flatten() ) / ( gm * dt ) + Jq

            # Solve the linear system
            dq[istage,...] = torch.sparse.spsolve( A, b ).view(nvar,np,nx,ny)

        q_new = q_old + torch.sum( dq * m * dt, dim=0 )
        q_new = torch.matmul( self.iCmtx, q_new.flatten() ).view(nvar,np,nx,ny)
        q[...,ids:ide,jds:jde] = q_new
        return q
    
    def linear_approximation(self,q,q0,dt):
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde

        nvar, np, nx_halo, ny_halo = q0.shape
        nx, ny = self.mesh.nx, self.mesh.ny
    
        qn = q0.clone()
        q  = q.clone()

        tend = self.spatial_operator(q)
        A, _ = self.generate_jacobian(dt)
        b = self.calculate_rhs_b(q,qn,tend,dt)
        
        del qn
        torch.cuda.empty_cache()
        # # print(torch.cuda.memory_summary())  # 显存使用报告 
        # # print( 'CUDA memory used', torch.cuda.memory_allocated()/1024/1024/1024, ' GB' )  # 当前占用量 

        # Solve the linear system
        dq = torch.sparse.spsolve( A, b ).view(nvar,np,nx,ny)

        # dq = torch.zeros(nvar,np,nx,ny,device=self.device)
        # dq = self.gmres(A, b, dq, m=1024, tol=1e-16, max_iter=1).view(nvar,np,nx,ny)

        q_old = torch.matmul( self.Cmtx, q[...,ids:ide,jds:jde].flatten() ).view(nvar,np,nx,ny)

        q_new = q_old + dq
        q_new = torch.matmul( self.iCmtx, q_new.flatten() ).view(nvar,np,nx,ny)

        return q_new, dq
    
    # Generate the Jacobian matrix, qL, qR, qB, qT are generated in the spatial operator
    def generate_jacobian(self,dt):
        nvar = self.nvar
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        nx = self.mesh.nx
        ny = self.mesh.ny
        np = self.mesh.npanel
        nCOS = self.mesh.nCOS
        nQOC = self.mesh.nQOC
        nPOE = self.mesh.nPOE
        nEOC = self.mesh.nEOC
        hw = self.mesh.hw
        nflx = nvar

        # _ = self.spatial_operator(q)

        # Calculate the Flux Jacobian matrix
        qL = self.qL
        qR = self.qR
        qB = self.qB
        qT = self.qT

        Ax = self.calc_riemann_tangent_x(qL, qR, self.jabghsL, self.mesh.jabL, self.JiGL, self.convert_coefL)
        Ay = self.calc_riemann_tangent_y(qB, qT, self.jabghsB, self.mesh.jabB, self.JiGB, self.convert_coefB)

        dfdqL = Ax[0:3,...] # nflx, nvar, np, nPOE, nx+1, ny
        dfdqR = Ax[3:6,...] # nflx, nvar, np, nPOE, nx+1, ny
        dfdqB = Ay[0:3,...] # nflx, nvar, np, nPOE, nx, ny+1
        dfdqT = Ay[3:6,...] # nflx, nvar, np, nPOE, nx, ny+1

        dfdqL = dfdqL * self.mesh.gw.unsqueeze(-1).unsqueeze(-1)
        dfdqR = dfdqR * self.mesh.gw.unsqueeze(-1).unsqueeze(-1)
        dfdqB = dfdqB * self.mesh.gw.unsqueeze(-1).unsqueeze(-1)
        dfdqT = dfdqT * self.mesh.gw.unsqueeze(-1).unsqueeze(-1)

        dfdqL = dfdqL.permute(2,4,5,0,3,1).reshape(np*(nx+1)*(ny  ),nflx,nPOE*nvar)
        dfdqR = dfdqR.permute(2,4,5,0,3,1).reshape(np*(nx+1)*(ny  ),nflx,nPOE*nvar)
        dfdqB = dfdqB.permute(2,4,5,0,3,1).reshape(np*(nx  )*(ny+1),nflx,nPOE*nvar)
        dfdqT = dfdqT.permute(2,4,5,0,3,1).reshape(np*(nx  )*(ny+1),nflx,nPOE*nvar)

        row_blk = nflx
        col_blk = nPOE * nvar
        dfdqL_coo = self.jab_blk_to_coo(dfdqL,np,nx+1,ny  ,row_blk,col_blk)
        dfdqR_coo = self.jab_blk_to_coo(dfdqR,np,nx+1,ny  ,row_blk,col_blk)
        dfdqB_coo = self.jab_blk_to_coo(dfdqB,np,nx  ,ny+1,row_blk,col_blk)
        dfdqT_coo = self.jab_blk_to_coo(dfdqT,np,nx  ,ny+1,row_blk,col_blk)

        dfdqL = spspmm( dfdqL_coo, self.dqLdq ) # (np,nx+1,ny,nflx)*(np,nx+1,ny,nPOE,nvar) * (np,nx+1,ny,nPOE,nvar)*(nvar*np*nx*ny)
        dfdqR = spspmm( dfdqR_coo, self.dqRdq )
        dfdqB = spspmm( dfdqB_coo, self.dqBdq )
        dfdqT = spspmm( dfdqT_coo, self.dqTdq )

        dfxdq = ( dfdqL + dfdqR ) #.reshape(np*(nx+1)*ny*nflx,np*nx*ny*nvar)
        dfydq = ( dfdqB + dfdqT ) #.reshape(np*nx*(ny+1)*nflx,np*nx*ny*nvar)

        # Calculate flux derivative
        # dfdx = ( f_{i+1/2} - f_{i-1/2} ) / dx, ( dfdy = f_{j+1/2} - f_{j-1/2} ) / dy
        # dfdx = ( f_{i+1} - f_{i} ) / dx, ( dfdy = f_{j+1} - f_{j} ) / dy
        dfxdq = spspmm( self.dfxdq_idx_mtx, dfxdq ) # (np*nx*ny*nflx,np*(nx+1)*ny*nflx) * (np*(nx+1)*ny*nflx,np*nx*ny*nvar)
        dfydq = spspmm( self.dfydq_idx_mtx, dfydq )

        # Calculate Source term Jacobian matrix
        qQ = self.qQ
        As = self.calc_src_tangent(qQ) # nflx, nvar+2, np, nQOC, nx, ny
        dpsidq          = As[:,0:3,...] # nflx, nvar, np, nQOC, nx, ny
        dpsi_B_ddphitdx = As[:,  3,...] # nflx,       np, nQOC, nx, ny
        dpsi_B_ddphitdy = As[:,  4,...] # nflx,       np, nQOC, nx, ny

        dpsidq          = dpsidq          * self.mesh.gw2d.unsqueeze(-1).unsqueeze(-1)
        dpsi_B_ddphitdx = dpsi_B_ddphitdx * self.mesh.gw2d.unsqueeze(-1).unsqueeze(-1)
        dpsi_B_ddphitdy = dpsi_B_ddphitdy * self.mesh.gw2d.unsqueeze(-1).unsqueeze(-1)

        dpsidq = dpsidq.permute(2,4,5,0,3,1).reshape(np*nx*ny,nvar,nQOC*nvar)
        dpsi_B_ddphitdx = dpsi_B_ddphitdx.permute(1,3,4,0,2).reshape(np*nx*ny,nvar,nQOC)
        dpsi_B_ddphitdy = dpsi_B_ddphitdy.permute(1,3,4,0,2).reshape(np*nx*ny,nvar,nQOC)
        
        row_blk = nflx
        col_blk = nQOC * nvar
        dpsidq_coo = self.jab_blk_to_coo(dpsidq,np,nx,ny,row_blk,col_blk) # (np*nx*ny*nvar)*(np*nx*ny*nQOC*nvar) * (np,nx,ny,nQOC,nvar)*(nvar*np*nx_halo*ny_halo)

        row_blk = nflx
        col_blk = nQOC
        dpsi_B_ddphitdx_coo = self.jab_blk_to_coo(dpsi_B_ddphitdx,np,nx,ny,row_blk,col_blk)
        dpsi_B_ddphitdy_coo = self.jab_blk_to_coo(dpsi_B_ddphitdy,np,nx,ny,row_blk,col_blk)

        # Permute array
        # from (np*nx*ny*nvar)*(nvar*np*nx_halo*ny_halo)
        # to (nvar*np*nx*ny)*(nvar*np*nx_halo*ny_halo)
        nrow = nvar*np*nx*ny
        ncol = nrow
        I = torch.arange( nrow, device=self.device )
        J = torch.arange( nrow, device=self.device ).reshape(np,nx,ny,nvar)
        J = J.permute(3,0,1,2).flatten()
        V = torch.ones(I.shape[0], device=self.device)
        IJ = torch.stack((I, J), dim=0)
        permute_mtx = torch.sparse_coo_tensor(IJ, V, (nrow, nrow), device=self.device)
        dfxdq = spspmm(permute_mtx, dfxdq).coalesce()
        dfydq = spspmm(permute_mtx, dfydq).coalesce()
        dpsidq_coo = spspmm(permute_mtx, dpsidq_coo).coalesce()
        dpsi_B_ddphitdx_coo = spspmm(permute_mtx, dpsi_B_ddphitdx_coo).coalesce()
        dpsi_B_ddphitdx_coo = spspmm(permute_mtx, dpsi_B_ddphitdx_coo).coalesce()

        dsrcdq = spspmm( dpsidq_coo, self.dqQdq ) \
               + spspmm( dpsi_B_ddphitdx_coo, self.ddphitdxdq ) \
               + spspmm( dpsi_B_ddphitdy_coo, self.ddphitdydq )
        
        Jv = dfxdq + dfydq - dsrcdq
        Jv = spspmm( self.Cmtx, Jv ) # (nvar*np*nx*ny,nvar*np*nx*ny)
        Jv = spspmm( Jv, self.iCmtx ) # (nvar*np*nx*ny,nvar*np*nx*ny)
        Jv = self.remove_zero_values(Jv, tol=0)

        nrow = nvar*np*nx*ny
        ncol = nvar*np*nx*ny
        I = torch.arange(nrow, device=self.device)
        J = torch.arange(ncol, device=self.device)
        V = torch.ones(nrow, device=self.device) / dt
        IJ = torch.stack((I, J), dim=0)
        E = torch.sparse_coo_tensor( IJ, V, ( nrow, ncol ), device=self.device )

        A = ( E + Jv ).coalesce()
        # A = self.remove_zero_values(A, tol=1.e-4).to_sparse_csr()
        A = self.remove_zero_values(A, tol=0).to_sparse_csr()

        # # Check Jacobian matrix by comparing with torch.autograd.functional.jacobian
        # eye = torch.eye( nelement ).to(device=self.device)
        # # print('nelement',nelement)
        # Jv = torch.autograd.functional.jacobian( self.spatial_operator_with_fill_ghost, q[...,ids:ide,jds:jde] )
        # Jv = Jv.view(nelement,nelement)
        # # Jv = torch.matmul( self.Cmtx, Jv ) # (nvar*np*nx*ny,nvar*np*nx*ny)
        # # Jv = torch.matmul( Jv, self.iCmtx ) # (nvar*np*nx*ny,nvar*np*nx*ny)
        # AA = eye / dt - Jv

        # file_name = 'Amtx.nc'
        # mtx = A
        # Dataset = nc.Dataset(file_name,mode='w',format='NETCDF4')
        # Dataset.createDimension('row', mtx.shape[0])
        # Dataset.createDimension('col', mtx.shape[1])
        # A_var = Dataset.createVariable('A', 'f8', ('row', 'col'))
        # A_var[:] = mtx.to_dense().cpu().numpy()
        # B_var = Dataset.createVariable('b', 'f8', ('row'))
        # B_var[:] = b.flatten().to_dense().cpu().numpy()
        # AA_var = Dataset.createVariable('AA', 'f8', ('row', 'col'))
        # AA_var[:] = AA.to_dense().cpu().numpy()
        # Dataset.close()
        # raise Exception('stop')
        # AA = A

        return A, Jv
    
    def remove_zero_values(self, Jv, tol=0):
        J0 = Jv.coalesce()
        abs_value = J0.values().abs()
        max_value = abs_value.max()
        mask = abs_value / max_value > tol
        new_indices = J0.indices()[:, mask]
        new_values = J0.values()[mask]
        new_shape = J0.shape
        Jv = torch.sparse_coo_tensor(new_indices, new_values, new_shape, device=self.device)
        # print(Jv._nnz(),J0._nnz(),(Jv._nnz()-J0._nnz())/J0._nnz())
        return Jv
    
    def calculate_rhs_b(self,q,qn,tend,dt):
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde

        b = ( -( ( q[...,ids:ide,jds:jde] - qn[...,ids:ide,jds:jde] ) / dt - tend ) ).flatten()
        
        b = torch.matmul( self.Cmtx, b )
        return b
    
    def gmres(self, A, b, dq, m=1024, tol=1e-5, max_iter=None):
        # A: sparse matrix
        # b: right-hand side vector
        # dq0: initial guess for the solution
        # m: Krylov subspace size
        # tol: tolerance for convergence
        # max_iter: maximum number of iterations

        nelement = b.nelement()
        dq = dq.flatten()
        h = torch.zeros(m+1,m,device=self.device)
        v = torch.zeros(nelement,m+1,device=self.device)
        e1 = torch.zeros(m+1,device=self.device)

        for iter in range(max_iter):
            Ax = A.matmul(dq)
            r0 = ( b - Ax ).flatten()

            beta = torch.norm(r0,p=2)
            v[:,0] = r0 / beta

            for j in range(m):
                Avj = A.matmul(v[:,j])
                h[0:j+1,j] = torch.matmul( Avj, v[:,0:j+1] )
                w = Avj - torch.sum( h[0:j+1,j] * v[:,0:j+1], dim=1 )
                h[j+1,j] = torch.norm(w,p=2)
                v[:,j+1] = w / h[j+1,j]

            e1[0] = beta
            y,residual,rank,_ = torch.linalg.lstsq(h,e1)
            inc = torch.matmul( v[:,:m], y )
            dq = dq + inc

            inc_norm = torch.norm(inc)
            print( 'iter',iter, 'inc_norm', inc_norm, 'beta', beta )
            if beta < tol:
                break

        return dq

    # Combine the Jacobian blocks into a COO matrix
    def jab_blk_to_coo(self,dfdq,np,nx,ny,row_blk,col_blk):
        ncell = np * nx * ny
        nrow  = ncell * row_blk
        ncol  = ncell * col_blk
        icell = torch.linspace(0,ncell-1,ncell,dtype=torch.long,device=self.device)
        rows = icell * row_blk
        row_indices = rows.unsqueeze(-1) + torch.arange(row_blk, dtype=torch.long, device=self.device)  # (ncell, row_blk)
        cols = icell * col_blk
        col_indices = cols.unsqueeze(-1) + torch.arange(col_blk, dtype=torch.long, device=self.device)  # (ncell, col_blk)

        I_idx = row_indices.unsqueeze(-1).expand(-1,-1,col_blk).flatten()
        J_idx = col_indices.unsqueeze(-2).expand(-1,row_blk,-1).flatten()

        indices = torch.stack([I_idx,J_idx],dim=0)
        values = dfdq[icell,...].flatten()
        dfdq_mtx = torch.sparse_coo_tensor(indices,values,(nrow,ncol),device=self.device)
        return dfdq_mtx
    
    def fill_ghost_jacobian(self):
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        nx = self.mesh.nx
        ny = self.mesh.ny
        nx_halo = self.mesh.nx_halo
        ny_halo = self.mesh.ny_halo
        np = self.mesh.npanel
        nvar = self.nvar
        n_gst_cell_per_panel = self.mesh.n_gst_cell_per_panel
        
        # Calculate full ghost interp matrix
        nrow = nvar*np*nx_halo*ny_halo
        ncol = self.gst_mtx.shape[0]
        I = torch.arange(nrow,device=self.device).view(nvar,np,nx_halo,ny_halo)
        J = torch.arange(ncol,device=self.device).flatten()

        I = I[...,self.mesh.igs,self.mesh.jgs].flatten()
        IJ = torch.stack((I, J), dim=0)
        V = torch.ones(IJ.shape[1], device=self.device)
        E = torch.sparse_coo_tensor(IJ, V, (nrow, ncol), device=self.device)
    
        gst_mtx = self.gst_mtx.to_sparse_coo().to(self.device)
        C = spspmm(E, gst_mtx).coalesce()

        # Calculate eye matrix
        nrow = nvar*np*nx_halo*ny_halo
        ncol = nvar*np*nx*ny
        I = torch.arange(nrow,device=self.device).view(nvar,np,nx_halo,ny_halo)
        J = torch.arange(ncol,device=self.device).flatten()

        I = I[...,ids:ide,jds:jde].flatten()
        IJ = torch.stack((I, J), dim=0)
        V = torch.ones(IJ.shape[1], device=self.device)
        E = torch.sparse_coo_tensor(IJ, V, (nrow, ncol), device=self.device)

        # Combine the matrices
        J = E + C
        return J
    
    def calc_jacobian_qrec(self):
        # Calculate the Jacobian of q_edge with respect to qL, qR, qB, qT
        # This is a placeholder function and should be implemented based on the specific requirements
        # For now, we just return the identity matrix as a placeholder
        nvar = self.nvar
        np, nx, ny = self.mesh.npanel, self.mesh.nx, self.mesh.ny
        nEOC, nPOE, nPOC, nQOC = self.mesh.nEOC, self.mesh.nPOE, self.mesh.nPOC, self.mesh.nQOC
        pls, ple = self.mesh.pls, self.mesh.ple
        prs, pre = self.mesh.prs, self.mesh.pre
        pbs, pbe = self.mesh.pbs, self.mesh.pbe
        pts, pte = self.mesh.pts, self.mesh.pte
        pqs, pqe = self.mesh.pqs, self.mesh.pqe

        nrx = nx; nry =ny

        # device = 'cpu'
        device = self.device

        # Calculate Matrix A
        ncol = nvar*np*nPOC*nrx*nry # nelement_qrec
        J = torch.arange(ncol, dtype=torch.long, device=device).reshape(nvar,np,nPOC,nrx,nry)

        JL= J[...,prs:pre,:,:].flatten()
        JR= J[...,pls:ple,:,:].flatten()
        JB= J[...,pts:pte,:,:].flatten()
        JT= J[...,pbs:pbe,:,:].flatten()
        JQ= J[...,pqs:pqe,:,:].flatten()
        
        nrow_qL = self.qL.nelement()
        nrow_qR = self.qR.nelement()
        nrow_qB = self.qB.nelement()
        nrow_qT = self.qT.nelement()
        nrow_qQ = self.qQ.nelement()

        IL = torch.arange(nrow_qL, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx+1,ny)
        IR = torch.arange(nrow_qR, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx+1,ny)
        IB = torch.arange(nrow_qB, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx,ny+1)
        IT = torch.arange(nrow_qT, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx,ny+1)
        IQ = torch.arange(nrow_qQ, dtype=torch.long, device=device).reshape(nvar,np,nQOC,nx,ny  )

        IL = IL[...,1:  ,:   ].flatten()
        IR = IR[...,0:-1,:   ].flatten()
        IB = IB[...,:   ,1:  ].flatten()
        IT = IT[...,:   ,0:-1].flatten()
        IQ = IQ.flatten()

        IJL = torch.stack((IL, JL), dim=0)
        IJR = torch.stack((IR, JR), dim=0)
        IJB = torch.stack((IB, JB), dim=0)
        IJT = torch.stack((IT, JT), dim=0)
        IJQ = torch.stack((IQ, JQ), dim=0)

        vL = torch.ones(IJL.shape[1], device=device)
        vR = torch.ones(IJR.shape[1], device=device)
        vB = torch.ones(IJB.shape[1], device=device)
        vT = torch.ones(IJT.shape[1], device=device)
        vQ = torch.ones(IJQ.shape[1], device=device)

        AL = torch.sparse_coo_tensor(IJL, vL, (nrow_qL, ncol), device=device)
        AR = torch.sparse_coo_tensor(IJR, vR, (nrow_qR, ncol), device=device)
        AB = torch.sparse_coo_tensor(IJB, vB, (nrow_qB, ncol), device=device)
        AT = torch.sparse_coo_tensor(IJT, vT, (nrow_qT, ncol), device=device)
        AQ = torch.sparse_coo_tensor(IJQ, vQ, (nrow_qQ, ncol), device=device)

        # Calculate Matrix B
        ncol_qL = self.qL.nelement()
        ncol_qR = self.qR.nelement()
        ncol_qB = self.qB.nelement()
        ncol_qT = self.qT.nelement()
        JL = torch.arange(ncol_qL, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx+1,ny)
        JR = torch.arange(ncol_qR, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx+1,ny)
        JB = torch.arange(ncol_qB, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx,ny+1)
        JT = torch.arange(ncol_qT, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx,ny+1)

        JL = JL[...,-1,:].flatten()
        JR = JR[..., 0,:].flatten()
        JB = JB[...,:,-1].flatten()
        JT = JT[...,:, 0].flatten()

        nrow = nvar*np*nPOE*nEOC*nx
        I = torch.arange(nrow, dtype=torch.long, device=device).reshape(nvar,np,nPOE*nEOC,nx)

        IL = I[..., pls:ple, :].flatten()
        IR = I[..., prs:pre, :].flatten()
        IB = I[..., pbs:pbe, :].flatten()
        IT = I[..., pts:pte, :].flatten()

        IJL = torch.stack((IL, JL), dim=0)
        IJR = torch.stack((IR, JR), dim=0)
        IJB = torch.stack((IB, JB), dim=0)
        IJT = torch.stack((IT, JT), dim=0)

        vL = torch.ones(IJL.shape[1], device=device)
        vR = torch.ones(IJR.shape[1], device=device)
        vB = torch.ones(IJB.shape[1], device=device)
        vT = torch.ones(IJT.shape[1], device=device)

        BL = torch.sparse_coo_tensor(IJL, vL, (nrow, ncol_qL), device=device)
        BR = torch.sparse_coo_tensor(IJR, vR, (nrow, ncol_qR), device=device)
        BB = torch.sparse_coo_tensor(IJB, vB, (nrow, ncol_qB), device=device)
        BT = torch.sparse_coo_tensor(IJT, vT, (nrow, ncol_qT), device=device)
        
        psiL = spspmm(BL, AL).coalesce()
        psiR = spspmm(BR, AR).coalesce()
        psiB = spspmm(BB, AB).coalesce()
        psiT = spspmm(BT, AT).coalesce()

        psi = psiL + psiR + psiB + psiT

        unify_panel_bdy_flux_matrix = self.unify_panel_bdy_flux_matrix.to_sparse_coo().to(device)
        W = spspmm( unify_panel_bdy_flux_matrix, psi ).coalesce()
        
        # Calculate Matrix D
        ncol = nvar*np*nPOE*nEOC*nx
        J = torch.arange(ncol, dtype=torch.long, device=device).reshape(nvar,np,nPOE*nEOC,nx)
        JL = J[..., pls:ple, :].flatten()
        JR = J[..., prs:pre, :].flatten()
        JB = J[..., pbs:pbe, :].flatten()
        JT = J[..., pts:pte, :].flatten()

        nrow_qL = self.qL.nelement()
        nrow_qR = self.qR.nelement()
        nrow_qB = self.qB.nelement()
        nrow_qT = self.qT.nelement()
        IL = torch.arange(nrow_qL, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx+1,ny)
        IR = torch.arange(nrow_qR, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx+1,ny)
        IB = torch.arange(nrow_qB, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx,ny+1)
        IT = torch.arange(nrow_qT, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx,ny+1)

        IL = IL[..., 0,:].flatten()
        IR = IR[...,-1,:].flatten()
        IB = IB[...,:, 0].flatten()
        IT = IT[...,:,-1].flatten()

        IJL = torch.stack((IL, JL), dim=0)
        IJR = torch.stack((IR, JR), dim=0)
        IJB = torch.stack((IB, JB), dim=0)
        IJT = torch.stack((IT, JT), dim=0)

        vL = torch.ones(IJL.shape[1], device=device)
        vR = torch.ones(IJR.shape[1], device=device)
        vB = torch.ones(IJB.shape[1], device=device)
        vT = torch.ones(IJT.shape[1], device=device)

        DL = torch.sparse_coo_tensor(IJL, vL, (nrow_qL, ncol), device=device)
        DR = torch.sparse_coo_tensor(IJR, vR, (nrow_qR, ncol), device=device)
        DB = torch.sparse_coo_tensor(IJB, vB, (nrow_qB, ncol), device=device)
        DT = torch.sparse_coo_tensor(IJT, vT, (nrow_qT, ncol), device=device)

        DLW = spspmm(DL, W).coalesce()
        DRW = spspmm(DR, W).coalesce()
        DBW = spspmm(DB, W).coalesce()
        DTW = spspmm(DT, W).coalesce()

        dqLdqrec = AL + DLW
        dqRdqrec = AR + DRW
        dqBdqrec = AB + DBW
        dqTdqrec = AT + DTW
        dqQdqrec = AQ
        
        # Permute array
        # from (nvar,np,nPOE,nx+1,ny)*(nvar*np*nPOC*nx*ny)
        # to (np,nx+1,ny,nPOE,nvar)*(nvar*np*nPOC*nx*ny)
        nrow = nvar*np*nPOE*(nx+1)*ny
        ncol = nrow
        I = torch.arange(nrow, dtype=torch.long, device=device)
        J = torch.arange(ncol, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx+1,ny)
        J = J.permute(1,3,4,2,0).flatten()
        V = torch.ones(I.shape[0], device=device)
        IJ = torch.stack((I, J), dim=0)
        permute_mtx = torch.sparse_coo_tensor(IJ, V, (nrow, ncol), device=device)
        dqLdqrec = spspmm(permute_mtx, dqLdqrec).coalesce()
        dqRdqrec = spspmm(permute_mtx, dqRdqrec).coalesce()

        nrow = nvar*np*nPOE*nx*(ny+1)
        ncol = nrow
        I = torch.arange(nrow, dtype=torch.long, device=device)
        J = torch.arange(ncol, dtype=torch.long, device=device).reshape(nvar,np,nPOE,nx,ny+1)
        J = J.permute(1,3,4,2,0).flatten()
        V = torch.ones(I.shape[0], device=device)
        IJ = torch.stack((I, J), dim=0)
        permute_mtx = torch.sparse_coo_tensor(IJ, V, (nrow, ncol), device=device)
        dqBdqrec = spspmm(permute_mtx, dqBdqrec).coalesce()
        dqTdqrec = spspmm(permute_mtx, dqTdqrec).coalesce()

        nrow = nvar*np*nQOC*nx*ny
        ncol = nrow
        I = torch.arange(nrow, dtype=torch.long, device=device)
        J = torch.arange(ncol, dtype=torch.long, device=device).reshape(nvar,np,nQOC,nx,ny)
        J = J.permute(1,3,4,2,0).flatten()
        V = torch.ones(I.shape[0], device=device)
        IJ = torch.stack((I, J), dim=0)
        permute_mtx = torch.sparse_coo_tensor(IJ, V, (nrow, ncol), device=device)
        dqQdqrec = spspmm(permute_mtx, dqQdqrec).coalesce()

        return dqLdqrec, dqRdqrec, dqBdqrec, dqTdqrec, dqQdqrec

    def calc_jacobian_conv2d(self,mtx,nvar):
        np = self.mesh.npanel
        sw = self.mesh.sw
        nCOS = self.mesh.nCOS
        nrx, nry = self.mesh.nrx, self.mesh.nry
        nx_halo, ny_halo = self.mesh.nx_halo, self.mesh.ny_halo

        ncell      = np * nrx     * nry
        ncell_halo = np * nx_halo * ny_halo
        
        row_blk = mtx.shape[0]
        col_blk = mtx.shape[1]

        nrow = nvar * ncell * row_blk
        ncol = nvar * ncell_halo

        J = torch.arange(ncol, device=self.device).reshape(nvar,np,nx_halo,ny_halo)
        J = J.unfold(3,sw,1).unfold(2,sw,1).reshape(nvar*np*nrx*nry,nCOS)

        nelement = nvar * ncell
        icell = torch.linspace(0,nelement-1,nelement,dtype=torch.long,device=self.device)
        rows = icell * row_blk
        row_indices = rows.unsqueeze(-1) + torch.arange(row_blk, device=self.device) # (ncell, row_blk)
        col_indices = J[icell,...] # (ncell, col_blk)

        I_idx = row_indices.unsqueeze(-1).expand(-1,-1,col_blk).flatten()
        J_idx = col_indices.unsqueeze(-2).expand(-1,row_blk,-1).flatten()

        indices = torch.stack([I_idx,J_idx],dim=0)
        values = mtx.expand(nvar*np*nrx*nry,-1,-1).flatten()
        conv2d_jacobian = torch.sparse_coo_tensor(indices,values,(nrow,ncol),device=self.device).coalesce()
        
        # Permute array
        # from (nvar*np*nx*ny*row_blk)*(nvar*np*nx_halo*ny_halo)
        # to (nvar*np*row_blk*nx*ny)*(nvar*np*nx_halo*ny_halo)
        I = torch.arange( nrow, device=self.device )
        J = torch.arange( nrow, device=self.device ).reshape(nvar,np,nrx,nry,row_blk)
        J = J.permute(0,1,4,2,3).flatten()
        V = torch.ones(I.shape[0], device=self.device)
        IJ = torch.stack((I, J), dim=0)
        permute_mtx = torch.sparse_coo_tensor(IJ, V, (nrow, nrow), device=self.device)
        conv2d_jacobian = spspmm(permute_mtx, conv2d_jacobian).coalesce()

        return conv2d_jacobian # (nvar*np*nx*ny*row_blk)*(nvar*np*nx_halo*ny_halo)
    
    def calc_f_derivative_idx_matrix(self):
        nvar = self.nvar
        nflx = nvar
        np, nx, ny = self.mesh.npanel, self.mesh.nx, self.mesh.ny
        
        I = torch.arange(np*nx*ny*nflx,dtype=torch.long,device=self.device)
        J = torch.arange(np*(nx+1)*ny*nflx,dtype=torch.long,device=self.device).reshape(np,(nx+1),ny,nflx)
        J1 = J[:,1:,...].flatten()
        J0 = J[:,0:-1,...].flatten()
        v = torch.ones(np*nx*ny*nflx,device=self.device)
        dfxdq_idx_mtx0 = torch.sparse_coo_tensor(torch.stack([I,J0]),v,(np*nx*ny*nflx,np*(nx+1)*ny*nflx),device=self.device)
        dfxdq_idx_mtx1 = torch.sparse_coo_tensor(torch.stack([I,J1]),v,(np*nx*ny*nflx,np*(nx+1)*ny*nflx),device=self.device)
        dfxdq_idx_mtx = dfxdq_idx_mtx1 - dfxdq_idx_mtx0
        
        I = torch.arange(np*nx*ny*nflx,dtype=torch.long,device=self.device)
        J = torch.arange(np*nx*(ny+1)*nflx,dtype=torch.long,device=self.device).reshape(np,nx,(ny+1),nflx)
        J1 = J[:,:,1:,...].flatten()
        J0 = J[:,:,0:-1,...].flatten()
        v = torch.ones(np*nx*ny*nflx,device=self.device)
        dfydq_idx_mtx0 = torch.sparse_coo_tensor(torch.stack([I,J0]),v,(np*nx*ny*nflx,np*nx*(ny+1)*nflx),device=self.device)
        dfydq_idx_mtx1 = torch.sparse_coo_tensor(torch.stack([I,J1]),v,(np*nx*ny*nflx,np*nx*(ny+1)*nflx),device=self.device)
        dfydq_idx_mtx = dfydq_idx_mtx1 - dfydq_idx_mtx0
        return dfxdq_idx_mtx, dfydq_idx_mtx
    
    # def spatial_operator_with_fill_ghost_pure_var(self,q0):
    #     ids = self.mesh.ids
    #     ide = self.mesh.ide
    #     jds = self.mesh.jds
    #     jde = self.mesh.jde
    #     ims = self.mesh.ims
    #     ime = self.mesh.ime
    #     jms = self.mesh.jms
    #     jme = self.mesh.jme
    #     nx  = self.mesh.nx_halo # ime - ims
    #     ny  = self.mesh.ny_halo # jme - jms
    #     np  = self.mesh.npanel
    #     nvar = self.nvar

    #     jab = self.mesh.jab[:,ids:ide,jds:jde]
    #     sqrt_iG11 = torch.sqrt( self.mesh.iG[0,0,:,ids:ide,jds:jde] )
    #     sqrt_iG22 = torch.sqrt( self.mesh.iG[1,1,:,ids:ide,jds:jde] )

    #     q = torch.zeros(nvar,np,nx,ny,dtype=q0.dtype).to(self.device)
    #     q[0,:,ids:ide,jds:jde] = q0[0,...] * jab * self.h**2
    #     q[1,:,ids:ide,jds:jde] = q0[1,...] * jab * sqrt_iG11 * self.h**3
    #     q[2,:,ids:ide,jds:jde] = q0[2,...] * jab * sqrt_iG22 * self.h**3
        
    #     q = self.fill_ghost(q)
    #     dq = self.spatial_operator(q)

    #     dq[0,...] = dq[0,...] / jab / self.h**2
    #     dq[1,...] = dq[1,...] / jab / sqrt_iG11 / self.h**3
    #     dq[2,...] = dq[2,...] / jab / sqrt_iG22 / self.h**3
    #     return dq
    
    def spatial_operator_with_fill_ghost(self,q0):
        ids = self.mesh.ids
        ide = self.mesh.ide
        jds = self.mesh.jds
        jde = self.mesh.jde
        nx_halo = self.mesh.nx_halo # ime - ims
        ny_halo = self.mesh.ny_halo # jme - jms
        np  = self.mesh.npanel
        nvar = self.nvar

        q = torch.zeros(nvar,np,nx_halo,ny_halo,dtype=q0.dtype,device=q0.device)
        q[...,ids:ide,jds:jde] = q0
        
        q = self.fill_ghost(q)
        dq = self.spatial_operator(q)
        return dq
    
    def calc_riemann_tangent_x(self, qL, qR, jabghs, jab, JiG, convert_coef):
        qL1 = qL[0,...]
        qL2 = qL[1,...]
        qL3 = qL[2,...]
        qR1 = qR[0,...]
        qR2 = qR[1,...]
        qR3 = qR[2,...]
        JiG1 = JiG[0,...]
        JiG2 = JiG[1,...]

        nvar, np, npts, nx, ny = qL.shape

        t2 = convert_coef*convert_coef
        t3 = jabghs*2.0
        t4 = 1.0/convert_coef
        t5 = 1.0/jab
        t6 = -qL1
        t7 = -qR1
        t8 = -qR2
        t9 = -qR3
        t10 = jabghs+t6
        t11 = qL1*t5
        t12 = jabghs+t7
        t13 = qR1*t5
        t14 = qL1+t7
        t15 = qL2+t8
        t16 = qL3+t9
        t17 = t5/2.0
        t18 = t5/4.0
        t19 = t5*t7
        t20 = -t17
        t21 = -t18
        t22 = t11/2.0
        t23 = t11/4.0
        t24 = t13/2.0
        t25 = t13/4.0
        t26 = 1.0/t10
        t28 = 1.0/t12
        t30 = torch.sqrt(t11)
        t31 = torch.sqrt(t13)
        t38 = t11+t19
        t27 = t26*t26
        t29 = t28*t28
        t32 = 1.0/t30
        t33 = 1.0/t31
        t34 = t30/2.0
        t35 = t31/2.0
        t36 = qL2*t4*t26
        t37 = qR2*t4*t28
        t39 = t4*t8*t28
        t40 = t36/2.0
        t41 = (qL2*t4*t27)/2.0
        t42 = t37/2.0
        t43 = (qR2*t4*t29)/2.0
        t44 = t34+t35
        t47 = t36+t39
        t45 = 1.0/t44
        t50 = t41*t44
        t51 = (qL2*t4*t27*t44)/4.0
        t52 = t43*t44
        t53 = (qR2*t4*t29*t44)/4.0
        t56 = (t5*t32*t47)/8.0
        t57 = (t5*t32*t47)/1.6E+1
        t58 = (t5*t33*t47)/8.0
        t59 = (t5*t33*t47)/1.6E+1
        t62 = (t44*t47)/2.0
        t63 = (t44*t47)/4.0
        t46 = t45*t45
        t48 = t17*t45
        t49 = t5*t45*(-1.0/2.0)
        t54 = (t38*t45)/2.0
        t60 = -t58
        t61 = -t59
        t66 = -t62
        t67 = -t63
        t74 = t20+t50+t56
        t75 = t21+t51+t57
        t55 = -t54
        t64 = (t5*t32*t38*t46)/8.0
        t65 = (t5*t33*t38*t46)/8.0
        t68 = t22+t24+t66
        t69 = t23+t25+t67
        t76 = t17+t52+t60
        t77 = t18+t53+t61
        t70 = t40+t42+t55
        t81 = t43+t48+t65
        t82 = t41+t49+t64
        t71 = convert_coef*t70
        t72 = torch.abs(t71)
        t73 = torch.sign(t71)#(t71/torch.abs(t71))
        t78 = t72+1.0E-16
        t79 = 1.0/t78
        t80 = t79*t79
        t83 = t71*t79
        t84 = t14*t83
        t85 = t15*t83
        t86 = t16*t83
        t87 = -t85
        t88 = -t86
        t89 = qL2+qR2+t87
        t90 = qL3+qR3+t88
        
        A0 = torch.zeros(2*nvar, nvar, np, npts, nx, ny, device=self.device)

        A0[0,0,...] = (t71*(t83+convert_coef*t14*t79*t82-t2*t14*t70*t73*t80*t82-1.0))/2.0-(convert_coef*t82*(qL1+qR1-t3-t84))/2.0
        A0[0,1,...] = (t71*((t14*t26*t79)/2.0-(t14*t26*t71*t73*t80)/2.0))/2.0-(t26*(qL1+qR1-t3-t84))/4.0
        A0[1,0,...] = (t71*(convert_coef*t15*t79*t82-t2*t15*t70*t73*t80*t82))/2.0-JiG1*t68*t75-JiG1*t69*t74-(convert_coef*t82*t89)/2.0
        A0[1,1,...] = (t71*(t83+(t15*t26*t79)/2.0-(t15*t26*t71*t73*t80)/2.0-1.0))/2.0-(t26*t89)/4.0-(JiG1*t4*t26*t44*t68)/4.0-(JiG1*t4*t26*t44*t69)/2.0
        A0[2,0,...] = (t71*(convert_coef*t16*t79*t82-t2*t16*t70*t73*t80*t82))/2.0-JiG2*t68*t75-JiG2*t69*t74-(convert_coef*t82*t90)/2.0
        A0[2,1,...] = t26*t90*(-1.0/4.0)+(t71*((t16*t26*t79)/2.0-(t16*t26*t71*t73*t80)/2.0))/2.0-(JiG2*t4*t26*t44*t68)/4.0-(JiG2*t4*t26*t44*t69)/2.0
        A0[2,2,...] = (t71*(t83-1.0))/2.0
        A0[3,0,...] = t71*(t83-convert_coef*t14*t79*t81+t2*t14*t70*t73*t80*t81+1.0)*(-1.0/2.0)-(convert_coef*t81*(qL1+qR1-t3-t84))/2.0
        A0[3,1,...] = (t71*((t14*t28*t79)/2.0-(t14*t28*t71*t73*t80)/2.0))/2.0-(t28*(qL1+qR1-t3-t84))/4.0
        A0[4,0,...] = (t71*(convert_coef*t15*t79*t81-t2*t15*t70*t73*t80*t81))/2.0+JiG1*t68*t77+JiG1*t69*t76-(convert_coef*t81*t89)/2.0
        A0[4,1,...] = t71*(t83-(t15*t28*t79)/2.0+(t15*t28*t71*t73*t80)/2.0+1.0)*(-1.0/2.0)-(t28*t89)/4.0+(JiG1*t4*t28*t44*t68)/4.0+(JiG1*t4*t28*t44*t69)/2.0
        A0[5,0,...] = (t71*(convert_coef*t16*t79*t81-t2*t16*t70*t73*t80*t81))/2.0+JiG2*t68*t77+JiG2*t69*t76-(convert_coef*t81*t90)/2.0
        A0[5,1,...] = t28*t90*(-1.0/4.0)+(t71*((t16*t28*t79)/2.0-(t16*t28*t71*t73*t80)/2.0))/2.0+(JiG2*t4*t28*t44*t68)/4.0+(JiG2*t4*t28*t44*t69)/2.0
        A0[5,2,...] = t71*(t83+1.0)*(-1.0/2.0)
        
        return A0

    def calc_riemann_tangent_y(self, qL, qR, jabghs, jab, JiG, convert_coef):
        qL1 = qL[0,...]
        qL2 = qL[1,...]
        qL3 = qL[2,...]
        qR1 = qR[0,...]
        qR2 = qR[1,...]
        qR3 = qR[2,...]
        JiG1 = JiG[0,...]
        JiG2 = JiG[1,...]

        nvar, np, npts, nx, ny = qL.shape

        t2 = convert_coef*convert_coef
        t3 = jabghs*2.0
        t4 = 1.0/convert_coef
        t5 = 1.0/jab
        t6 = -qL1
        t7 = -qR1
        t8 = -qR2
        t9 = -qR3
        t10 = jabghs+t6
        t11 = qL1*t5
        t12 = jabghs+t7
        t13 = qR1*t5
        t14 = qL1+t7
        t15 = qL2+t8
        t16 = qL3+t9
        t17 = t5/2.0
        t18 = t5/4.0
        t19 = t5*t7
        t20 = -t17
        t21 = -t18
        t22 = t11/2.0
        t23 = t11/4.0
        t24 = t13/2.0
        t25 = t13/4.0
        t26 = 1.0/t10
        t28 = 1.0/t12
        t30 = torch.sqrt(t11)
        t31 = torch.sqrt(t13)
        t38 = t11+t19
        t27 = t26*t26
        t29 = t28*t28
        t32 = 1.0/t30
        t33 = 1.0/t31
        t34 = t30/2.0
        t35 = t31/2.0
        t36 = qL3*t4*t26
        t37 = qR3*t4*t28
        t39 = t4*t9*t28
        t40 = t36/2.0
        t41 = (qL3*t4*t27)/2.0
        t42 = t37/2.0
        t43 = (qR3*t4*t29)/2.0
        t44 = t34+t35
        t47 = t36+t39
        t45 = 1.0/t44
        t50 = t41*t44
        t51 = (qL3*t4*t27*t44)/4.0
        t52 = t43*t44
        t53 = (qR3*t4*t29*t44)/4.0
        t56 = (t5*t32*t47)/8.0
        t57 = (t5*t32*t47)/1.6E+1
        t58 = (t5*t33*t47)/8.0
        t59 = (t5*t33*t47)/1.6E+1
        t62 = (t44*t47)/2.0
        t63 = (t44*t47)/4.0
        t46 = t45*t45
        t48 = t17*t45
        t49 = t5*t45*(-1.0/2.0)
        t54 = (t38*t45)/2.0
        t60 = -t58
        t61 = -t59
        t66 = -t62
        t67 = -t63
        t74 = t20+t50+t56
        t75 = t21+t51+t57
        t55 = -t54
        t64 = (t5*t32*t38*t46)/8.0
        t65 = (t5*t33*t38*t46)/8.0
        t68 = t22+t24+t66
        t69 = t23+t25+t67
        t76 = t17+t52+t60
        t77 = t18+t53+t61
        t70 = t40+t42+t55
        t81 = t43+t48+t65
        t82 = t41+t49+t64
        t71 = convert_coef*t70
        t72 = torch.abs(t71)
        t73 = torch.sign(t71)#(t71/torch.abs(t71))
        t78 = t72+1.0E-16
        t79 = 1.0/t78
        t80 = t79*t79
        t83 = t71*t79
        t84 = t14*t83
        t85 = t15*t83
        t86 = t16*t83
        t87 = -t85
        t88 = -t86
        t89 = qL2+qR2+t87
        t90 = qL3+qR3+t88
        
        A0 = torch.zeros(2*nvar, nvar, np, npts, nx, ny, device=self.device)

        A0[0,0,...] = (t71*(t83+convert_coef*t14*t79*t82-t2*t14*t70*t73*t80*t82-1.0))/2.0-(convert_coef*t82*(qL1+qR1-t3-t84))/2.0
        A0[0,2,...] = (t71*((t14*t26*t79)/2.0-(t14*t26*t71*t73*t80)/2.0))/2.0-(t26*(qL1+qR1-t3-t84))/4.0
        A0[1,0,...] = (t71*(convert_coef*t15*t79*t82-t2*t15*t70*t73*t80*t82))/2.0-JiG1*t68*t75-JiG1*t69*t74-(convert_coef*t82*t89)/2.0
        A0[1,1,...] = (t71*(t83-1.0))/2.0
        A0[1,2,...] = t26*t89*(-1.0/4.0)+(t71*((t15*t26*t79)/2.0-(t15*t26*t71*t73*t80)/2.0))/2.0-(JiG1*t4*t26*t44*t68)/4.0-(JiG1*t4*t26*t44*t69)/2.0
        A0[2,0,...] = (t71*(convert_coef*t16*t79*t82-t2*t16*t70*t73*t80*t82))/2.0-JiG2*t68*t75-JiG2*t69*t74-(convert_coef*t82*t90)/2.0
        A0[2,2,...] = (t71*(t83+(t16*t26*t79)/2.0-(t16*t26*t71*t73*t80)/2.0-1.0))/2.0-(t26*t90)/4.0-(JiG2*t4*t26*t44*t68)/4.0-(JiG2*t4*t26*t44*t69)/2.0
        A0[3,0,...] = t71*(t83-convert_coef*t14*t79*t81+t2*t14*t70*t73*t80*t81+1.0)*(-1.0/2.0)-(convert_coef*t81*(qL1+qR1-t3-t84))/2.0
        A0[3,2,...] = (t71*((t14*t28*t79)/2.0-(t14*t28*t71*t73*t80)/2.0))/2.0-(t28*(qL1+qR1-t3-t84))/4.0
        A0[4,0,...] = (t71*(convert_coef*t15*t79*t81-t2*t15*t70*t73*t80*t81))/2.0+JiG1*t68*t77+JiG1*t69*t76-(convert_coef*t81*t89)/2.0
        A0[4,1,...] = t71*(t83+1.0)*(-1.0/2.0)
        A0[4,2,...] = t28*t89*(-1.0/4.0)+(t71*((t15*t28*t79)/2.0-(t15*t28*t71*t73*t80)/2.0))/2.0+(JiG1*t4*t28*t44*t68)/4.0+(JiG1*t4*t28*t44*t69)/2.0
        A0[5,0,...] = (t71*(convert_coef*t16*t79*t81-t2*t16*t70*t73*t80*t81))/2.0+JiG2*t68*t77+JiG2*t69*t76-(convert_coef*t81*t90)/2.0
        A0[5,2,...] = t71*(t83-(t16*t28*t79)/2.0+(t16*t28*t71*t73*t80)/2.0+1.0)*(-1.0/2.0)-(t28*t90)/4.0+(JiG2*t4*t28*t44*t68)/4.0+(JiG2*t4*t28*t44*t69)/2.0

        return A0
    
    def calc_src_tangent(self, q):
        jabghsQ = self.jabghsQ
        iGQ_C   = self.iGQ_C
        iGQ_S   = self.iGQ_S
        coef_M  = self.coef_M

        coef_M1_1 = coef_M[0,0,...]
        coef_M1_2 = coef_M[0,1,...]
        coef_M2_1 = coef_M[1,0,...]
        coef_M2_2 = coef_M[1,1,...]

        iGQ_C1_1 = iGQ_C[0,0,...]
        iGQ_C1_2 = iGQ_C[0,1,...]
        iGQ_C2_1 = iGQ_C[1,0,...]
        iGQ_C2_2 = iGQ_C[1,1,...]

        iGQ_S1_1 = iGQ_S[0,0,...]
        iGQ_S1_2 = iGQ_S[0,1,...]
        iGQ_S2_1 = iGQ_S[1,0,...]
        iGQ_S2_2 = iGQ_S[1,1,...]

        nvar, np, npts, nx, ny = q.shape

        q1 = q[0,...]
        q2 = q[1,...]
        q3 = q[2,...]

        t2 = -q1
        t3 = jabghsQ+t2
        t4 = 1.0/t3
        t5 = t4*t4
        
        A0 = torch.zeros(nvar, nvar+2, np, npts, nx, ny, device=self.device)

        A0[1,0,...] = -t5*(coef_M1_1*(q2*q2)+coef_M1_2*q2*q3)
        A0[1,1,...] = -iGQ_C1_2-t4*(coef_M1_1*q2*2.0+coef_M1_2*q3)
        A0[1,2,...] = iGQ_C1_1-coef_M1_2*q2*t4
        A0[1,3,...] = iGQ_S1_1
        A0[1,4,...] = iGQ_S1_2
        A0[2,0,...] = -t5*(coef_M2_2*(q3*q3)+coef_M2_1*q2*q3)
        A0[2,1,...] = -iGQ_C2_2-coef_M2_1*q3*t4
        A0[2,2,...] = iGQ_C2_1-t4*(coef_M2_1*q2+coef_M2_2*q3*2.0)
        A0[2,3,...] = iGQ_S2_1
        A0[2,4,...] = iGQ_S2_2

        return A0
