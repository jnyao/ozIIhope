import os
import sys
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from quadrature import _precompute_grid
from diag import pause, plot_cube_field

class mesh_class(nn.Module):
    def __init__(self,mesh_type,parallel,nx,ny,nz,npanel,rw,r):
        # mesh_type: choose from 'cubed_sphere', 'lonlat'
        super(mesh_class, self).__init__()
        pi = math.pi
        R2D = 180. / pi

        print('Begin mesh init')
        self.mesh_type = mesh_type
        self.parallel = parallel
        self.device = parallel.device
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.npanel = npanel
        self.rw = rw
        self.hw = rw * 2
        self.sw = 2 * rw + 1
        self.r = r
        self.nq = self.rw + 1
        self.nrx = self.nx + 2 * self.rw
        self.nry = self.ny + 2 * self.rw

        # choose between original and stretched jacobian False for origin jab, True for reduced jab
        self.use_jab_stretching = True
        if self.use_jab_stretching:
            self.jab_stretching = r**2
        else:
            self.jab_stretching = 1

        self.nEOC = 4 # number of edges on cell
        self.nCOS = self.sw**2 # number of cells on stencil
        self.nPOE = 1 #self.nq # number of points on edge
        # self.nQOC = self.nPOE**2 # number of quadrature points on cell
        self.nQOC = self.nq**2 # number of quadrature points on cell
        self.nPOR = 1 + self.nEOC * self.nPOE # number of reconstruction points on cell
        self.nPOC = self.nPOR + self.nQOC # number of points on cell

        self.pc  = 0
        self.pls = self.pc  + self.nPOE
        self.ple = self.pls + self.nPOE
        self.prs = self.ple
        self.pre = self.prs + self.nPOE
        self.pbs = self.pre
        self.pbe = self.pbs + self.nPOE
        self.pts = self.pbe
        self.pte = self.pts + self.nPOE
        self.pqs = self.pte
        self.pqe = self.pqs + self.nQOC

        self.gx, self.gw = _precompute_grid(self.nq, grid="legendre-gauss", a=0.0, b=1.0, periodic=False)

        digit_type = torch.get_default_dtype()
        self.gx = torch.tensor(self.gx,dtype=digit_type,device=self.parallel.device)
        self.gw = torch.tensor(self.gw,dtype=digit_type,device=self.parallel.device)
        self.gw2d = torch.mm( self.gw.view(self.nq,1), self.gw.view(1,self.nq) ).view(self.nQOC)

        if self.mesh_type == 'cubed_sphere':
            self.set_cubed_sphere()
        else:
            print('Not supported mesh')

        print( 'rank, min/max value of lon %1d %5f, %.5f' % ( parallel.rank, torch.min(self.lon).item()*R2D, torch.max(self.lon).item()*R2D ) )
        print( 'rank, min/max value of lat %1d %5f, %.5f' % ( parallel.rank, torch.min(self.lat).item()*R2D, torch.max(self.lat).item()*R2D ) )

        print('Finish mesh init')

    def calc_cubed_sphere_coordinate(self,x,y):
        npts, nx, ny = x.shape

        lon = torch.zeros( self.npanel_local, npts, nx, ny, device=self.device )
        lat = torch.zeros( self.npanel_local, npts, nx, ny, device=self.device )
        for ip in range(self.npanel_local):
            ipanel = self.panel[ip]
            lon[ip,...], lat[ip,...] = self.pointProjPlane2Sphere( x, y, ipanel )
        lon = torch.where( lon<0, lon + 2.*math.pi, lon )
        return lon, lat

    def calc_cubed_sphere_metric(self,lon,lat,x,y):
        npanel, npts, nx, ny = lon.shape

        A  = torch.zeros(2, 2, npanel, npts, nx, ny, device=self.device)
        iA = torch.zeros(2, 2, npanel, npts, nx, ny, device=self.device)
        for ip in range(npanel):
            ipanel = self.panel[ip]
            A [:,:,ip,...] = self.calc_A (lon[ip,...],lat[ip,...],self.r,ipanel)
            iA[:,:,ip,...] = self.calc_iA(lon[ip,...],lat[ip,...],self.r,ipanel)
    
        G  = self.calc_G (x,y,self.r).unsqueeze(2).repeat(1,1,npanel,1,1,1).to(self.device)
        iG = self.calc_iG(x,y,self.r).unsqueeze(2).repeat(1,1,npanel,1,1,1).to(self.device)

        # # Calculate G and iG by G=AT*A, iG=G^-1
        # AA = A.permute(2,3,4,5,0,1)
        # AAT = AA.transpose(4,5)
        # G = torch.matmul( AAT, AA )
        # iG = torch.linalg.inv( G )
        # G = G.permute(4,5,0,1,2,3)
        # iG = iG.permute(4,5,0,1,2,3)

        jab = torch.zeros(npanel, npts, nx, ny, device=self.device)
        jab[0,...] = self.calc_jab(x,y,self.r)
        jab[1:,...] = jab[0,...]
        return A, iA, G, iG, jab

    def set_cubed_sphere(self):
        self.x_min = -0.25 * math.pi
        self.x_max = 0.25 * math.pi
        self.y_min = -0.25 * math.pi
        self.y_max = 0.25 * math.pi
        self.dx = ( self.x_max - self.x_min ) / self.nx
        self.dy = ( self.y_max - self.y_min ) / self.ny
        self.inv_dx = 1 / self.dx
        self.inv_dy = 1 / self.dy

        self.x_min_halo = self.x_min - self.hw * self.dx
        self.x_max_halo = self.x_max + self.hw * self.dx
        self.y_min_halo = self.y_min - self.hw * self.dy
        self.y_max_halo = self.y_max + self.hw * self.dy

        self.nx_halo = self.nx + self.hw * 2
        self.ny_halo = self.ny + self.hw * 2

        # ips, ipe, self.npanel_local = self.parallel.round_robin(self.npanel)
        ips = 1
        ipe = 6
        self.npanel_local = 6
        self.panel = torch.arange(ips,ipe+1,1,dtype=torch.int16)

        self.ims = 0
        self.ime = self.nx_halo
        self.jms = 0
        self.jme = self.ny_halo

        self.ids = self.hw
        self.ide = self.nx + self.hw
        self.jds = self.hw
        self.jde = self.ny + self.hw

        self.irs = self.rw
        self.ire = self.nx_halo - self.rw
        self.jrs = self.rw
        self.jre = self.ny_halo - self.rw

        self.nCells = self.nx_halo * self.ny_halo * self.npanel
        self.nCells_local = self.nx_halo * self.ny_halo * self.npanel_local

        ###############
        # Cell Center #
        ###############
        x = torch.linspace( self.x_min_halo+0.5*self.dx, self.x_max_halo-0.5*self.dx, self.nx_halo ).to(self.device)
        y = torch.linspace( self.y_min_halo+0.5*self.dy, self.y_max_halo-0.5*self.dy, self.ny_halo ).to(self.device)

        self.x, self.y = torch.meshgrid( x, y, indexing='ij' )
        x = self.x.view(1,self.nx_halo,self.ny_halo)
        y = self.y.view(1,self.nx_halo,self.ny_halo)
        self.lon, self.lat = self.calc_cubed_sphere_coordinate(x,y)
        self.A, self.iA, self.G, self.iG, self.jab = self.calc_cubed_sphere_metric(self.lon,self.lat,x,y)

        self.lon = torch.squeeze( self.lon )
        self.lat = torch.squeeze( self.lat )
        self.A = torch.squeeze( self.A )
        self.iA = torch.squeeze( self.iA )
        self.G = torch.squeeze( self.G )
        self.iG = torch.squeeze( self.iG )
        self.jab = torch.squeeze( self.jab )

        ################
        # bottom edges #
        ################
        npts = self.nPOE
        nx = self.nrx
        ny = self.ny+1

        self.xB = torch.zeros(npts,nx,ny,device=self.device)
        self.yB = torch.zeros(npts,nx,ny,device=self.device)

        nc = nx * ny
        I1 = torch.arange(0,nx,device=self.device)
        J1 = torch.arange(0,ny,device=self.device)
        I, J = torch.meshgrid( I1, J1, indexing='ij' )
        I = I.reshape(nc).contiguous()
        J = J.reshape(nc).contiguous()
        
        if npts == 1:
            gx = torch.tensor([0.5],device=self.device)
        else:
            gx = self.gx

        for p in range(npts):
            self.xB[p,I,J] = ( I + gx[p] - self.rw ) * self.dx + self.x_min
            self.yB[p,I,J] = J * self.dy + self.y_min

        x = self.xB
        y = self.yB
        self.lonB, self.latB = self.calc_cubed_sphere_coordinate(x,y)
        lon = self.lonB
        lat = self.latB
        self.AB, self.iAB, self.GB, self.iGB, self.jabB = self.calc_cubed_sphere_metric(lon,lat,x,y)
        
        ##############
        # left edges #
        ##############
        npts = self.nPOE
        nx = self.nx+1
        ny = self.nry

        self.xL = torch.zeros(npts,nx,ny,device=self.device)
        self.yL = torch.zeros(npts,nx,ny,device=self.device)

        nc = nx * ny
        I1 = torch.arange(0,nx,device=self.device)
        J1 = torch.arange(0,ny,device=self.device)
        I, J = torch.meshgrid( I1, J1, indexing='ij' )
        I = I.reshape(nc).contiguous()
        J = J.reshape(nc).contiguous()
        
        if npts == 1:
            gx = torch.tensor([0.5],device=self.device)
        else:
            gx = self.gx
        
        for p in range(npts):
            self.xL[p,I,J] = I * self.dx + self.x_min
            self.yL[p,I,J] = ( J + gx[p] - self.rw ) * self.dy + self.y_min
        
        x = self.xL
        y = self.yL
        self.lonL, self.latL = self.calc_cubed_sphere_coordinate(x,y)
        lon = self.lonL
        lat = self.latL
        self.AL, self.iAL, self.GL, self.iGL, self.jabL = self.calc_cubed_sphere_metric(lon,lat,x,y)

        #####################
        # quadrature points #
        #####################
        npts = self.nQOC
        nx = self.nx_halo
        ny = self.ny_halo

        self.xQ = torch.zeros(npts,nx,ny,device=self.device)
        self.yQ = torch.zeros(npts,nx,ny,device=self.device)

        nc = nx * ny
        I1 = torch.arange(0,nx,device=self.device)
        J1 = torch.arange(0,ny,device=self.device)
        I, J = torch.meshgrid( I1, J1, indexing='ij' )
        I = I.reshape(nc).contiguous()
        J = J.reshape(nc).contiguous()

        ic = 0
        for jl in range(self.nq):
            for il in range(self.nq):
                self.xQ[ic,I,J] = ( I + self.gx[il] ) * self.dx + self.x_min_halo
                self.yQ[ic,I,J] = ( J + self.gx[jl] ) * self.dy + self.y_min_halo
                ic += 1

        x = self.xQ
        y = self.yQ
        self.lonQ, self.latQ = self.calc_cubed_sphere_coordinate(x,y)
        lon = self.lonQ
        lat = self.latQ
        self.AQ, self.iAQ, self.GQ, self.iGQ, self.jabQ = self.calc_cubed_sphere_metric(lon,lat,x,y)

        ################
        # Cell Average #
        ################
        # self.jabCell = torch.einsum('npij,p->nij',self.jabQ,self.gw2d)

        x = ( self.x - 0.5 * self.dx ).expand(self.npanel_local, -1, -1)
        y = ( self.y - 0.5 * self.dy ).expand(self.npanel_local, -1, -1)
        if self.use_jab_stretching:
            r = self.r
        else:
            r = 1
        self.jabCell = self.EquiangularElementArea(x, self.dx, y, self.dy, r) / (self.dx * self.dy)

        ###############################
        # Metric terms on cell center #
        ###############################
        ids = self.ids
        ide = self.ide
        jds = self.jds
        jde = self.jde
        self.tanx = torch.tan(self.x)
        self.tany = torch.tan(self.y)
        self.delta = torch.sqrt( 1. + self.tanx**2 + self.tany**2 )

        coef_M = 2 * torch.ones_like(self.jab) / ( self.delta * self.delta )
        npanel,nx,ny = coef_M.shape
        self.coef_M = torch.zeros(2,2,npanel,nx,ny,device=self.parallel.device)
        x = self.tanx
        y = self.tany
        xx = x * x
        yy = y * y
        self.coef_M[0,0,...] = coef_M * ( - x * yy )
        self.coef_M[0,1,...] = coef_M * ( y * ( 1. + yy ) )
        self.coef_M[1,0,...] = coef_M * ( x * ( 1. + xx ) )
        self.coef_M[1,1,...] = coef_M * ( - y * xx )
        
        self.set_cubed_sphere_ghost_points()

        # Set metric tensor for unify panel boundary flux
        self.jabL_bdy_adj = torch.zeros(       self.npanel_local, self.nPOE, self.nry, device=self.parallel.device )
        self.jabR_bdy_adj = torch.zeros(       self.npanel_local, self.nPOE, self.nry, device=self.parallel.device )
        self.jabB_bdy_adj = torch.zeros(       self.npanel_local, self.nPOE, self.nrx, device=self.parallel.device )
        self.jabT_bdy_adj = torch.zeros(       self.npanel_local, self.nPOE, self.nrx, device=self.parallel.device )
        self.AL_bdy_adj   = torch.zeros( 2, 2, self.npanel_local, self.nPOE, self.nry, device=self.parallel.device )
        self.AR_bdy_adj   = torch.zeros( 2, 2, self.npanel_local, self.nPOE, self.nry, device=self.parallel.device )
        self.AB_bdy_adj   = torch.zeros( 2, 2, self.npanel_local, self.nPOE, self.nrx, device=self.parallel.device )
        self.AT_bdy_adj   = torch.zeros( 2, 2, self.npanel_local, self.nPOE, self.nrx, device=self.parallel.device )

        # ! Panel 1
        self.jabL_bdy_adj[    0,...] = self.jabL[    3,:,-1, :] #! Left
        self.jabR_bdy_adj[    0,...] = self.jabL[    1,:, 0, :] #! Right
        self.jabB_bdy_adj[    0,...] = self.jabB[    5,:, :,-1] #! below
        self.jabT_bdy_adj[    0,...] = self.jabB[    4,:, :, 0] #! over
        self.AL_bdy_adj  [:,:,0,...] = self.AL  [:,:,3,:,-1, :]
        self.AR_bdy_adj  [:,:,0,...] = self.AL  [:,:,1,:, 0, :]
        self.AB_bdy_adj  [:,:,0,...] = self.AB  [:,:,5,:, :,-1]
        self.AT_bdy_adj  [:,:,0,...] = self.AB  [:,:,4,:, :, 0]
        # ! Panel 2
        self.jabL_bdy_adj[    1,...] = self.jabL[    0,:,-1,:] #! Left
        self.jabR_bdy_adj[    1,...] = self.jabL[    2,:, 0,:] #! Right
        self.jabB_bdy_adj[    1,...] = self.jabL[    5,:,-1,:].flip(-1,-2) #! below
        self.jabT_bdy_adj[    1,...] = self.jabL[    4,:,-1,:] #! over
        self.AL_bdy_adj  [:,:,1,...] = self.AL  [:,:,0,:,-1,:]
        self.AR_bdy_adj  [:,:,1,...] = self.AL  [:,:,2,:, 0,:]
        self.AB_bdy_adj  [:,:,1,...] = self.AL  [:,:,5,:,-1,:].flip(-1,-2)
        self.AT_bdy_adj  [:,:,1,...] = self.AL  [:,:,4,:,-1,:]
        # ! Panel 3
        self.jabL_bdy_adj[    2,...] = self.jabL[    1,:,-1, :] #! Left
        self.jabR_bdy_adj[    2,...] = self.jabL[    3,:, 0, :] #! Right
        self.jabB_bdy_adj[    2,...] = self.jabB[    5,:,: , 0].flip(-1,-2) #! below
        self.jabT_bdy_adj[    2,...] = self.jabB[    4,:,: ,-1].flip(-1,-2) #! over
        self.AL_bdy_adj  [:,:,2,...] = self.AL  [:,:,1,:,-1, :]
        self.AR_bdy_adj  [:,:,2,...] = self.AL  [:,:,3,:, 0, :]
        self.AB_bdy_adj  [:,:,2,...] = self.AB  [:,:,5,:,: , 0].flip(-1,-2)
        self.AT_bdy_adj  [:,:,2,...] = self.AB  [:,:,4,:,: ,-1].flip(-1,-2)
        # ! Panel 4
        self.jabL_bdy_adj[    3,...] = self.jabL[    2,:,-1,:] #! Left
        self.jabR_bdy_adj[    3,...] = self.jabL[    0,:, 0,:] #! Right
        self.jabB_bdy_adj[    3,...] = self.jabL[    5,:, 0,:] #! below
        self.jabT_bdy_adj[    3,...] = self.jabL[    4,:, 0,:].flip(-1,-2) #! over
        self.AL_bdy_adj  [:,:,3,...] = self.AL  [:,:,2,:,-1,:]
        self.AR_bdy_adj  [:,:,3,...] = self.AL  [:,:,0,:, 0,:]
        self.AB_bdy_adj  [:,:,3,...] = self.AL  [:,:,5,:, 0,:]
        self.AT_bdy_adj  [:,:,3,...] = self.AL  [:,:,4,:, 0,:].flip(-1,-2)
        # ! Panel 5
        self.jabL_bdy_adj[    4,...] = self.jabB[    3,:,:,-1].flip(-1,-2) #! Left
        self.jabR_bdy_adj[    4,...] = self.jabB[    1,:,:,-1] #! Right
        self.jabB_bdy_adj[    4,...] = self.jabB[    0,:,:,-1] #! below
        self.jabT_bdy_adj[    4,...] = self.jabB[    2,:,:,-1].flip(-1,-2) #! over
        self.AL_bdy_adj  [:,:,4,...] = self.AB  [:,:,3,:,:,-1].flip(-1,-2)
        self.AR_bdy_adj  [:,:,4,...] = self.AB  [:,:,1,:,:,-1]
        self.AB_bdy_adj  [:,:,4,...] = self.AB  [:,:,0,:,:,-1]
        self.AT_bdy_adj  [:,:,4,...] = self.AB  [:,:,2,:,:,-1].flip(-1,-2)
        # ! Panel 6
        self.jabL_bdy_adj[    5,...] = self.jabB[    3,:,:,0] #! Left
        self.jabR_bdy_adj[    5,...] = self.jabB[    1,:,:,0].flip(-1,-2) #! Right
        self.jabB_bdy_adj[    5,...] = self.jabB[    2,:,:,0].flip(-1,-2) #! below
        self.jabT_bdy_adj[    5,...] = self.jabB[    0,:,:,0] #! over
        self.AL_bdy_adj  [:,:,5,...] = self.AB  [:,:,3,:,:,0]
        self.AR_bdy_adj  [:,:,5,...] = self.AB  [:,:,1,:,:,0].flip(-1,-2)
        self.AB_bdy_adj  [:,:,5,...] = self.AB  [:,:,2,:,:,0].flip(-1,-2)
        self.AT_bdy_adj  [:,:,5,...] = self.AB  [:,:,0,:,:,0]
        
        self.jabL_bdy_dom = torch.zeros(       self.npanel_local, self.nPOE, self.nry, device=self.parallel.device )
        self.jabR_bdy_dom = torch.zeros(       self.npanel_local, self.nPOE, self.nry, device=self.parallel.device )
        self.jabB_bdy_dom = torch.zeros(       self.npanel_local, self.nPOE, self.nrx, device=self.parallel.device )
        self.jabT_bdy_dom = torch.zeros(       self.npanel_local, self.nPOE, self.nrx, device=self.parallel.device )
        self.iAL_bdy_dom  = torch.zeros( 2, 2, self.npanel_local, self.nPOE, self.nry, device=self.parallel.device )
        self.iAR_bdy_dom  = torch.zeros( 2, 2, self.npanel_local, self.nPOE, self.nry, device=self.parallel.device )
        self.iAB_bdy_dom  = torch.zeros( 2, 2, self.npanel_local, self.nPOE, self.nrx, device=self.parallel.device )
        self.iAT_bdy_dom  = torch.zeros( 2, 2, self.npanel_local, self.nPOE, self.nrx, device=self.parallel.device )

        self.jabL_bdy_dom = self.jabL[..., 0, :]
        self.jabR_bdy_dom = self.jabL[...,-1, :]
        self.jabB_bdy_dom = self.jabB[..., :, 0]
        self.jabT_bdy_dom = self.jabB[..., :,-1]
        self.iAL_bdy_dom  = self.iAL [..., 0, :]
        self.iAR_bdy_dom  = self.iAL [...,-1, :]
        self.iAB_bdy_dom  = self.iAB [..., :, 0]
        self.iAT_bdy_dom  = self.iAB [..., :,-1]

        # permute for matmul
        self.AL_bdy_adj = self.AL_bdy_adj.permute(2,3,4,0,1)
        self.AR_bdy_adj = self.AR_bdy_adj.permute(2,3,4,0,1)
        self.AB_bdy_adj = self.AB_bdy_adj.permute(2,3,4,0,1)
        self.AT_bdy_adj = self.AT_bdy_adj.permute(2,3,4,0,1)
        self.iAL_bdy_dom = self.iAL_bdy_dom.permute(2,3,4,0,1)
        self.iAR_bdy_dom = self.iAR_bdy_dom.permute(2,3,4,0,1)
        self.iAB_bdy_dom = self.iAB_bdy_dom.permute(2,3,4,0,1)
        self.iAT_bdy_dom = self.iAT_bdy_dom.permute(2,3,4,0,1)

        self.jabL_bdy_cvt = self.jabL_bdy_dom / self.jabL_bdy_adj
        self.jabR_bdy_cvt = self.jabR_bdy_dom / self.jabR_bdy_adj
        self.jabB_bdy_cvt = self.jabB_bdy_dom / self.jabB_bdy_adj
        self.jabT_bdy_cvt = self.jabT_bdy_dom / self.jabT_bdy_adj

        self.AL_bdy_cvt = torch.matmul( self.iAL_bdy_dom, self.AL_bdy_adj ).permute(3,4,0,1,2)
        self.AR_bdy_cvt = torch.matmul( self.iAR_bdy_dom, self.AR_bdy_adj ).permute(3,4,0,1,2)
        self.AB_bdy_cvt = torch.matmul( self.iAB_bdy_dom, self.AB_bdy_adj ).permute(3,4,0,1,2)
        self.AT_bdy_cvt = torch.matmul( self.iAT_bdy_dom, self.AT_bdy_adj ).permute(3,4,0,1,2)

        del self.jabL_bdy_dom, self.jabL_bdy_adj, self.iAL_bdy_dom, self.AL_bdy_adj
        del self.jabR_bdy_dom, self.jabR_bdy_adj, self.iAR_bdy_dom, self.AR_bdy_adj
        del self.jabB_bdy_dom, self.jabB_bdy_adj, self.iAB_bdy_dom, self.AB_bdy_adj
        del self.jabT_bdy_dom, self.jabT_bdy_adj, self.iAT_bdy_dom, self.AT_bdy_adj

    #------------------------------------------------------------------------------
    # SUBROUTINE EquiangularElementArea
    #
    # Description:
    #   Compute the area of a single equiangular cubed sphere grid cell.
    #
    # Parameters: 
    #   alpha - Alpha coordinate of lower-left corner of grid cell
    #   da - Delta alpha
    #   beta - Beta coordinate of lower-left corner of grid cell
    #   db - Delta beta
    #------------------------------------------------------------------------------
    def EquiangularElementArea(self,alpha,da,beta,db,r):
        pi = math.pi

        # Calculate interior grid angles
        a1 =      self.EquiangularGridAngle(alpha   , beta   )
        a2 = pi - self.EquiangularGridAngle(alpha+da, beta   )
        a3 = pi - self.EquiangularGridAngle(alpha   , beta+db)
        a4 =      self.EquiangularGridAngle(alpha+da, beta+db)

        area = ( a1 + a2 + a3 + a4 - 2. * pi ) * r**2
        return area
    
    def EquiangularGridAngle(self,alpha,beta):
        return torch.acos(-torch.sin(alpha) * torch.sin(beta))

    def pointProjPlane2Sphere(self, x, y, ipanel):
        if self.mesh_type == 'cubed_sphere':
            one = torch.tensor(1.,dtype=torch.float64,device=x.device)
            if ipanel<=4:
                lon = x + 0.5 * ( ipanel - 1 ) * math.pi
                lat = torch.atan2( torch.tan(y) * torch.cos(x), one )
            elif ipanel==5:
                X = torch.tan(x)
                Y = torch.tan(y)
                lon = torch.atan2( X, -Y )
                lat = torch.atan2( one, torch.sqrt( X*X + Y*Y ) )
            elif ipanel==6:
                X = torch.tan(x)
                Y = torch.tan(y)
                lon = torch.atan2( X, Y )
                lat =-torch.atan2( one, torch.sqrt( X*X + Y*Y ) )
        elif self.mesh_type == 'lonlat':
            lon = x
            lat = y

        return lon, lat

    def calc_G(self,x,y,r):
        G = torch.zeros_like( x )
        G = torch.stack([G,G],dim=0)
        G = torch.stack([G,G],dim=0)
        X = torch.tan(x)
        Y = torch.tan(y)
        X2 = X**2
        Y2 = Y**2
        rho = torch.sqrt( 1. + X2 + Y2 )

        G[0,0,...] = 1. + X2
        G[0,1,...] = -X*Y
        G[1,0,...] = G[0,1,...]
        G[1,1,...] = 1. + Y2
        G = r**2 / ( rho**4 * torch.cos(x)**2 * torch.cos(y)**2 ) * G
        return G

    def calc_iG(self,x,y,r):
        iG = torch.zeros_like( x )
        iG = torch.stack([iG,iG],dim=0)
        iG = torch.stack([iG,iG],dim=0)

        X = torch.tan(x)
        Y = torch.tan(y)
        X2 = X**2
        Y2 = Y**2
        rho = torch.sqrt( 1. + X2 + Y2 )

        iG[0,0,...] = 1. + Y2
        iG[0,1,...] = X*Y
        iG[1,0,...] = iG[0,1,...]
        iG[1,1,...] = 1. + X2
        iG = ( rho**2 * torch.cos(x)**2 * torch.cos(y)**2 ) / r**2 * iG
        return iG
    
    def calc_jab(self,x,y,r):
        X = torch.tan(x)
        Y = torch.tan(y)
        X2 = X**2
        Y2 = Y**2
        rho = torch.sqrt( 1. + X2 + Y2 )

        jab = r**2 / self.jab_stretching / ( torch.cos(x)**2 * torch.cos(y)**2 * rho**3 )
        return jab
    
    def calc_iA(self, lon, lat, r, ipanel):
        iA = torch.zeros_like( lon )
        iA = torch.stack([iA,iA],dim=0)
        iA = torch.stack([iA,iA],dim=0)
        if ipanel <= 4:
          alambda = lon - ( ipanel - 1 ) * math.pi / 2.
          atheta=lat
          a = torch.sin(alambda)
          b = torch.cos(alambda)
          c = torch.sin(atheta)
          d = torch.cos(atheta)
          temp = d*d*b*b+c*c
          iA[0,0,...] = 1./d
          iA[0,1,...] = 0.
          iA[1,0,...] = a*c/temp
          iA[1,1,...] = b/temp
        elif ipanel==5 :
          alambda = lon
          atheta = lat
          a = torch.sin(alambda)
          b = torch.cos(alambda)
          c = torch.sin(atheta)
          d = torch.cos(atheta)
          temp = c+a*a*d*d/c
          iA[0,0,...] = b/temp
          iA[0,1,...] = -a/c/temp
          temp = c+b*b*d*d/c
          iA[1,0,...] = a/temp
          iA[1,1,...] = b/c/temp
        else:
          alambda = lon
          atheta = lat
          a = torch.sin(alambda)
          b = torch.cos(alambda)
          c = torch.sin(atheta)
          d = torch.cos(atheta)
          temp = c+a*a*d*d/c
          iA[0,0,...] = -b/temp
          iA[0,1,...] = a/c/temp
          temp = c+b*b*d*d/c
          iA[1,0,...] = a/temp
          iA[1,1,...] = b/c/temp

        iA = iA / r

        return iA

    def calc_A(self, lon, lat, r, ipanel):
        A = lon
        A = torch.stack([A,A],dim=0)
        A = torch.stack([A,A],dim=0)
        if ipanel <= 4:
          alambda = lon - ( ipanel - 1 ) * math.pi / 2.
          atheta = lat
          a = torch.sin(alambda)
          b = torch.cos(alambda)
          c = torch.sin(atheta)
          d = torch.cos(atheta)
          A[0,0,...] = d
          A[0,1,...] = 0.
          A[1,0,...] = -c*d*a/b
          A[1,1,...] = b*d*d+c*c/b
        elif ipanel==5:
          alambda = lon
          atheta = lat
          a = torch.sin(alambda)
          b = torch.cos(alambda)
          c = torch.sin(atheta)
          d = torch.cos(atheta)
          temp = 1.+a*a*d*d/c/c
          A[0,0,...] = b*c*temp
          A[1,0,...] = -c*c*a*temp
          temp = 1.+b*b*d*d/c/c
          A[0,1,...] = a*c*temp
          A[1,1,...] = b*c*c*temp
        else:
          alambda = lon
          atheta = lat
          a = torch.sin(alambda)
          b = torch.cos(alambda)
          c = torch.sin(atheta)
          d = torch.cos(atheta)
          temp = 1.+a*a*d*d/c/c
          A[0,0,...] = -b*c*temp
          A[1,0,...] = c*c*a*temp
          temp = 1.+b*b*d*d/c/c
          A[0,1,...] = a*c*temp
          A[1,1,...] = b*c*c*temp

        A = A * r

        return A
  
    def contravProjSphere2Plane(self, iA, sv1, sv2):
        contrav1 = iA[0,0,...] * sv1 + iA[0,1,...] * sv2
        contrav2 = iA[1,0,...] * sv1 + iA[1,1,...] * sv2
        return contrav1, contrav2
  
    def contravProjPlane2Sphere(self, A, contrav1, contrav2):
        sv1 = A[0,0,...] * contrav1 + A[0,1,...] * contrav2
        sv2 = A[1,0,...] * contrav1 + A[1,1,...] * contrav2
        return sv1, sv2
    
    def contrav2cov(self, G, contrav1, contrav2):
        cov1 = G[0,0,...] * contrav1 + G[0,1,...] * contrav2
        cov2 = G[1,0,...] * contrav1 + G[1,1,...] * contrav2
        return cov1, cov2

    def pointProjSphere2Plane(self,lon,lat,iPatch):
        pi = math.pi
        
        x14 = torch.atan( torch.tan( lon-(iPatch-1)*pi/2. ) )
        y14 = torch.atan( torch.tan(lat) / torch.cos(lon-(iPatch-1)*pi/2.) )
        x56 = torch.atan((-1.)**(iPatch+1)*torch.sin(lon)/torch.tan(lat))
        y56 = torch.atan(-torch.cos(lon)/torch.tan(lat))

        x = torch.where( (iPatch>=1) & (iPatch <=4), x14, x56 )
        y = torch.where( (iPatch>=1) & (iPatch <=4), y14, y56 )

        return x, y
    
    def psp2ploc_cross_edge(self,lon_in,lat_in):
        pi = math.pi
        piq = 0.25 * pi
        pih = 0.5 * pi
        R2D = 180. / pi

        digit_type = torch.get_default_dtype()
        if digit_type==torch.float64:
            tolerance = 1.e-14
        elif digit_type==torch.float32:
            tolerance = 1.e-5

        lon = lon_in.clone()
        lat = lat_in.clone()

        cross_edge = False

        lon = torch.where( lon <   -piq, lon + 2*pi, lon )
        lon = torch.where( lon > 7.*piq, lon - 2*pi, lon )
        
        pnl = [1,2,3,4]
        iPatch1 = -torch.ones_like(lon, dtype=torch.long)
        iPatch2 = -torch.ones_like(lon, dtype=torch.long)
        for ip in range(4):
            ipanel = pnl[ip]
            angs = -piq + ip*pih
            ange =  piq + ip*pih
            ps = pnl[ ( 3 + ip ) % 4 ]
            pe = pnl[ ( 1 + ip ) % 4 ]
            iPatch1 = torch.where( (lon >= angs) & (lon <= ange), ipanel, iPatch1 )
            iPatch2 = torch.where( (lon >= angs) & (lon <= ange), ipanel, iPatch2 )
            iPatch2 = torch.where( (iPatch2==iPatch1) & (torch.abs( lon-angs )<=tolerance), ps, iPatch2 )
            iPatch2 = torch.where( (iPatch2==iPatch1) & (torch.abs( lon-ange )<=tolerance), pe, iPatch2 )
        
        x1, y1 = self.pointProjSphere2Plane( lon, lat, iPatch1 )
        x2, y2 = self.pointProjSphere2Plane( lon, lat, iPatch2 )

        iPatch2 = torch.where( torch.abs( y1 + piq ) < tolerance, 6, iPatch2 )
        iPatch2 = torch.where( torch.abs( y1 - piq ) < tolerance, 5, iPatch2 )

        iPatch1 = torch.where( y1<=-piq, 6, iPatch1 )
        iPatch2 = torch.where( y1<=-piq, 6, iPatch2 )
        iPatch1 = torch.where( y1>= piq, 5, iPatch1 )
        iPatch2 = torch.where( y1>= piq, 5, iPatch2 )

        x1, y1 = self.pointProjSphere2Plane( lon, lat, iPatch1 )

        iPatch2 = torch.where( (iPatch1==6) & (torch.abs( x1 + piq )<tolerance), 4, iPatch2 )
        iPatch2 = torch.where( (iPatch1==6) & (torch.abs( x1 - piq )<tolerance), 2, iPatch2 )
        iPatch2 = torch.where( (iPatch1==6) & (torch.abs( y1 + piq )<tolerance), 3, iPatch2 )
        iPatch2 = torch.where( (iPatch1==6) & (torch.abs( y1 - piq )<tolerance), 1, iPatch2 )

        iPatch2 = torch.where( (iPatch1==5) & (torch.abs( x1 + piq )<tolerance), 4, iPatch2 )
        iPatch2 = torch.where( (iPatch1==5) & (torch.abs( x1 - piq )<tolerance), 2, iPatch2 )
        iPatch2 = torch.where( (iPatch1==5) & (torch.abs( y1 + piq )<tolerance), 1, iPatch2 )
        iPatch2 = torch.where( (iPatch1==5) & (torch.abs( y1 - piq )<tolerance), 3, iPatch2 )

        x2, y2 = self.pointProjSphere2Plane( lon, lat, iPatch2 )
        
        cross_edge = torch.where( iPatch1!=iPatch2, True, False )
        
        return x1, y1, iPatch1, x2, y2, iPatch2, cross_edge
    
    def set_cubed_sphere_ghost_points(self):
        ids = self.ids
        ide = self.ide
        jds = self.jds
        jde = self.jde
        ims = self.ims
        ime = self.ime
        jms = self.jms
        jme = self.jme
        R2D = 180. / math.pi

        self.out_domain = torch.ones(self.nx_halo,self.ny_halo,dtype=torch.long,device=self.device)
        self.out_domain[ids:ide,jds:jde] = 0

        idx = self.out_domain.nonzero()
        igs = idx[:,0]
        jgs = idx[:,1]
        self.igs = igs.to(self.device)
        self.jgs = jgs.to(self.device)
        self.n_gst_cell_per_panel = self.igs.shape[0]

        lonQ = self.lonQ[...,igs,jgs].permute(0,2,1)
        latQ = self.latQ[...,igs,jgs].permute(0,2,1)
        x1, y1, p1, x2, y2, p2, cross_edge = self.psp2ploc_cross_edge( lonQ, latQ )

        self.gst_c = cross_edge

        _, self.n_out_dom_cell_per_panel, _ = x1.shape # npanel, nodcpp, nQOC
        self.n_gst_pts = x1.nelement()

        gst_x = torch.stack( [x1,x2], dim=0 )
        gst_y = torch.stack( [y1,y2], dim=0 )
        gst_p = torch.stack( [p1,p2], dim=0 )

        gst_x = torch.where( gst_x<self.x_min, self.x_min, gst_x )
        gst_x = torch.where( gst_x>self.x_max, self.x_max, gst_x )
        gst_y = torch.where( gst_y<self.y_min, self.y_min, gst_y )
        gst_y = torch.where( gst_y>self.y_max, self.y_max, gst_y )

        p0 = gst_p
        pp = -torch.ones_like(gst_p,device=self.device)
        for ip in range(self.npanel_local):
            ipn = ( p0==self.panel[ip] ).nonzero()
            pp[ ipn[:,0], ipn[:,1], ipn[:,2], ipn[:,3] ] = ip
        gst_p = pp

        gst_i = torch.floor( ( gst_x - self.x_min ) / self.dx )
        gst_j = torch.floor( ( gst_y - self.y_min ) / self.dy )
        gst_i = torch.where( gst_i<0        , 0        , gst_i )
        gst_i = torch.where( gst_i>self.nx-1, self.nx-1, gst_i )
        gst_j = torch.where( gst_j<0        , 0        , gst_j )
        gst_j = torch.where( gst_j>self.ny-1, self.ny-1, gst_j )
        gst_i += self.hw
        gst_j += self.hw
        gst_i = gst_i.to(torch.long)
        gst_j = gst_j.to(torch.long)

        self.gst_x = gst_x[0,...]
        self.gst_y = gst_y[0,...]
        self.gst_p = gst_p[0,...]
        self.gst_i = gst_i[0,...]
        self.gst_j = gst_j[0,...]

        self.gst_x = self.gst_x.reshape(self.n_gst_pts).contiguous()
        self.gst_y = self.gst_y.reshape(self.n_gst_pts).contiguous()
        self.gst_p = self.gst_p.reshape(self.n_gst_pts).contiguous()
        self.gst_c = self.gst_c.reshape(self.n_gst_pts).contiguous()
        self.gst_i = self.gst_i.reshape(self.n_gst_pts).contiguous()
        self.gst_j = self.gst_j.reshape(self.n_gst_pts).contiguous()

        self.gst_ir = torch.zeros( self.n_gst_pts, self.nCOS, dtype=torch.long, device=self.device )
        self.gst_jr = torch.zeros( self.n_gst_pts, self.nCOS, dtype=torch.long, device=self.device )
        self.gst_pr = torch.zeros( self.n_gst_pts, self.nCOS, dtype=torch.long, device=self.device )
        iCOS = 0
        for j in range(-self.rw,self.rw+1):
            for i in range(-self.rw,self.rw+1):
                self.gst_ir[...,iCOS] = self.gst_i + i
                self.gst_jr[...,iCOS] = self.gst_j + j
                self.gst_pr[...,iCOS] = self.gst_p
                iCOS += 1

        # Overlap points
        self.olp_idx = torch.squeeze( self.gst_c.nonzero() )
        self.n_olp_pts = torch.squeeze( self.gst_c.count_nonzero() )
        gst_x = gst_x.reshape(2,self.n_gst_pts).contiguous()
        gst_y = gst_y.reshape(2,self.n_gst_pts).contiguous()
        gst_p = gst_p.reshape(2,self.n_gst_pts).contiguous()
        gst_i = gst_i.reshape(2,self.n_gst_pts).contiguous()
        gst_j = gst_j.reshape(2,self.n_gst_pts).contiguous()
        self.gst_x2 = gst_x[:,self.olp_idx].view(2*self.n_olp_pts)
        self.gst_y2 = gst_y[:,self.olp_idx].view(2*self.n_olp_pts)
        self.gst_p2 = gst_p[:,self.olp_idx].view(2*self.n_olp_pts)
        self.gst_i2 = gst_i[:,self.olp_idx].view(2*self.n_olp_pts)
        self.gst_j2 = gst_j[:,self.olp_idx].view(2*self.n_olp_pts)

        self.gst_ir2 = torch.zeros( 2 * self.n_olp_pts, self.nCOS, dtype=torch.long, device=self.device )
        self.gst_jr2 = torch.zeros( 2 * self.n_olp_pts, self.nCOS, dtype=torch.long, device=self.device )
        self.gst_pr2 = torch.zeros( 2 * self.n_olp_pts, self.nCOS, dtype=torch.long, device=self.device )
        iCOS = 0
        for j in range(-self.rw,self.rw+1):
            for i in range(-self.rw,self.rw+1):
                self.gst_ir2[...,iCOS] = self.gst_i2 + i
                self.gst_jr2[...,iCOS] = self.gst_j2 + j
                self.gst_pr2[...,iCOS] = self.gst_p2
                iCOS += 1

        # Calculate panel convert metric
        lon = lonQ.reshape(self.n_gst_pts).contiguous()
        lat = latQ.reshape(self.n_gst_pts).contiguous()

        gst_p = self.gst_p
        self.jabG = torch.zeros( self.n_gst_pts, device=self.parallel.device )
        self.AG = torch.zeros( 2, 2, self.n_gst_pts, device=self.parallel.device )
        for ip in range(self.npanel_local):
            ipanel = self.panel[ip]
            idx = torch.squeeze( ( gst_p==ip ).nonzero() )
            self.jabG[idx] = self.calc_jab(self.gst_x[idx],self.gst_y[idx],self.r)
            self.AG[:,:,idx] = self.calc_A( lon[idx], lat[idx], self.r, ipanel )

        lon = lon[self.olp_idx]
        lat = lat[self.olp_idx]
        
        gst_x2 = self.gst_x2.view(2,self.n_olp_pts)
        gst_y2 = self.gst_y2.view(2,self.n_olp_pts)
        gst_p2 = self.gst_p2.view(2,self.n_olp_pts)
        self.jabG2 = torch.zeros( 2, self.n_olp_pts, device=self.parallel.device )
        self.AG2 = torch.zeros( 2, 2, 2, self.n_olp_pts, device=self.parallel.device )
        for ip in range(self.npanel_local):
            ipanel = self.panel[ip]

            icover = 0
            idx = torch.squeeze( ( gst_p2[icover,...]==ip ).nonzero() )
            self.jabG2[icover,idx] = self.calc_jab(gst_x2[icover,idx],gst_y2[icover,idx],self.r)
            self.AG2[...,icover,idx] = self.calc_A( lon[idx], lat[idx], self.r, ipanel )

            icover = 1
            idx = torch.squeeze( ( gst_p2[icover,...]==ip ).nonzero() )
            self.jabG2[icover,idx] = self.calc_jab(gst_x2[icover,idx],gst_y2[icover,idx],self.r)
            self.AG2[...,icover,idx] = self.calc_A( lon[idx], lat[idx], self.r, ipanel )

        self.jabQG = self.jabQ[...,self.igs,self.jgs].permute(0,2,1).reshape(self.n_gst_pts).contiguous()
        self.iAQG = self.iAQ[...,self.igs,self.jgs].permute(0,1,2,4,3).reshape(2,2,self.n_gst_pts).contiguous()

        self.jabQG2 = self.jabQG[self.olp_idx]
        self.iAQG2 = self.iAQG[...,self.olp_idx]
        self.iAQG2 = torch.stack( [self.iAQG2,self.iAQG2], dim=2 )

        # permute for matmul
        self.AG = self.AG.permute(2,0,1)
        self.iAQG = self.iAQG.permute(2,0,1)

        self.AG2 = self.AG2.permute(2,3,0,1)
        self.iAQG2 = self.iAQG2.permute(2,3,0,1)
        
        self.jabG_cvt = self.jabQG / self.jabG
        self.jabG2_cvt = self.jabQG2 / self.jabG2

        self.AG_cvt = torch.matmul( self.iAQG, self.AG ).permute(1,2,0)
        self.AG2_cvt = torch.matmul( self.iAQG2, self.AG2 ).permute(2,3,0,1)

        del self.AG, self.iAQG, self.AG2, self.iAQG2, self.jabG, self.jabQG, self.jabG2, self.jabQG2