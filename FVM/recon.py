import torch
import torch.nn.functional as F
import numpy as np
import cudnn
import os
import gp
numsplit = int(os.environ["NUMSPLIT"])
bits_per_slice = int(os.environ["BITS_PER_SLICE"])
OZCUDNN_CHNLS = 16
OZCUDNN_OUTCHNLS = int(os.environ["OZCUDNN_OUTCHNLS"])
OZ_NORMAL = 0
OZ_BSHALF = 1
OZ_MODE = int(os.environ["OZ_MODE"])
prec_mode = int(os.environ["PREC_MODE"])

def calc_1D_poly_matrix(nx,xi):
    device = xi.device
    m = xi.shape[0]
    p = torch.zeros(m,nx,device=device)
    for i in range(nx):
        p[:,i] = xi**i
    return p

def calc_1D_poly_integration(nx,x_min,x_max,device):
    I = torch.arange(0,nx,device=device)
    c = ( x_max**(I+1) - x_min**(I+1) ) / ( I + 1 )
    return c

def calc_rectangle_poly_matrix(nx,ny,xi,eta):
    device = xi.device
    m = xi.shape[0]
    p = torch.zeros(m,nx*ny,device=device)
    x = xi
    y = eta
    iCOS = 0
    for j in range(ny):
        for i in range(nx):
            p[:,iCOS] = x**i * y**j
            iCOS = iCOS + 1
    return p
    
def calc_rectangle_poly_deriv_matrix(nx,ny,xi,eta):
    device = xi.device
    m = xi.shape[0]
    dpdx = torch.zeros(m,nx*ny,device=device)
    dpdy = torch.zeros(m,nx*ny,device=device)
    for iPOC in range(m):
        x = xi [iPOC]
        y = eta[iPOC]

        iCOS = 0
        for j in range( ny ):
            for i in range( nx ):
                if i-1<0:
                    cx = 0
                    px = 0
                else:
                    cx = i
                    px = i-1
                
                if j-1<0:
                    cy = 0
                    py = 0
                else:
                    cy = j
                    py = j-1

                dpdx[iPOC,iCOS] = cx * x**px * y**j
                dpdy[iPOC,iCOS] = cy * x**i * y**py
                iCOS = iCOS + 1
    return dpdx, dpdy

def calc_rectangle_poly_integration(nx,ny,x_min,x_max,y_min,y_max,device):
    I = torch.arange(0,nx,device=device).repeat(ny,1).flatten()
    J = torch.arange(0,ny,device=device).repeat(nx,1).transpose(0,1).flatten()
    c = ( x_max**(I+1) - x_min**(I+1) ) * ( y_max**(J+1) - y_min**(J+1) ) / ( ( I + 1 ) * ( J + 1 ) )

    # iCOS = 0
    # c = torch.zeros(nx*ny)
    # for j in range(ny):
    #     for i in range(nx):
    #         c[iCOS] = ( x_max**(i+1) - x_min**(i+1) ) * ( y_max**(j+1) - y_min**(j+1) ) / ( ( i + 1 ) * ( j + 1 ) )
    #         iCOS = iCOS + 1
    return c

class weno_smooth_indicator_2(torch.nn.Module):
    def __init__(self):
        super(weno_smooth_indicator_2, self).__init__()
        self.order = 2
    #@torch.compile
    def forward(self,a):
        WENO_smooth_indicator = a[...,1]**2 + a[...,2]**2 + 7. * a[...,3]**2 / 6.
        return WENO_smooth_indicator
    
class weno_smooth_indicator_3(torch.nn.Module):
    def __init__(self):
        super(weno_smooth_indicator_3, self).__init__()
        self.order = 3
    #@torch.compile
    def forward(self,a):
        WENO_smooth_indicator = ( 720  * a[...,1] * a[...,1] + 3120 * a[...,2] * a[...,2] + 720  * a[...,3] * a[...,3] + 840   * a[...,4] * a[...,4] \
                                + 120  * a[...,3] * a[...,5] + 3389 * a[...,5] * a[...,5] + 3120 * a[...,6] * a[...,6] + 120   * a[...,1] * a[...,7] \
                                + 3389 * a[...,7] * a[...,7] + 520  * a[...,2] * a[...,8] + 520  * a[...,6] * a[...,8] + 13598 * a[...,8] * a[...,8] ) / 720
        return WENO_smooth_indicator
    
class weno_smooth_indicator_4(torch.nn.Module):
    def __init__(self):
        super(weno_smooth_indicator_4, self).__init__()
        self.order = 4
    #@torch.compile
    def forward(self,a):
        WENO_smooth_indicator = a[...,1]**2 + (13*a[...,2]**2)/3 + (3129*a[...,3]**2)/80 + a[...,4]**2 + (7*a[...,5]**2)/6 + (1./6)*a[...,4]*a[...,6] + (3389*a[...,6]**2)/720      \
                                + (17./30)*a[...,5]*a[...,7] + (47459*a[...,7]**2)/1120 + (13*a[...,8]**2)/3 + (1./24)*a[...,3]*a[...,9] + (3389*a[...,9]**2)/720                   \
                                + (13./18)*a[...,2]*a[...,10] + (13./18)*a[...,8]*a[...,10] + (6799*a[...,10]**2)/360 + (1043./160)*a[...,3]*a[...,11] + (73./32)*a[...,9]*a[...,11]\
                                + (22846129*a[...,11]**2)/134400 + (1./24)*a[...,1]*(12*a[...,3] + 4*a[...,9] + a[...,11]) + (1./2)*a[...,4]*a[...,12] + (1./24)*a[...,6]*a[...,12] \
                                + (3129*a[...,12]**2)/80 + (17./30)*a[...,5]*a[...,13] + (11./80)*a[...,7]*a[...,13] + (47459*a[...,13]**2)/1120 + (1./24)*a[...,4]*a[...,14]       \
                                + (73./32)*a[...,6]*a[...,14] + (1043./160)*a[...,12]*a[...,14] + (22846129*a[...,14]**2)/134400 + (11./80)*a[...,5]*a[...,15]                      \
                                + (114997*a[...,7]*a[...,15])/5600 + (114997*a[...,13]*a[...,15])/5600 + (19583517*a[...,15]**2)/12800
        return WENO_smooth_indicator
        
class weno_smooth_indicator_5(torch.nn.Module):
    def __init__(self):
        super(weno_smooth_indicator_5, self).__init__()
        self.order = 5
    #@torch.compile
    def forward(self,a):
        WENO_smooth_indicator = a[...,1]**2. + (13.*a[...,2]**2)/3. + (3129.*a[...,3]**2)/80. + (87617.*a[...,4]**2)/140. + a[...,5]**2. + (7.*a[...,6]**2)/6. + (1./6)*a[...,5]*a[...,7] + (3389.*a[...,7]**2)/720. + (17./30)*a[...,6]*a[...,8] + (47459.*a[...,8]**2)/1120. + (1./40)*a[...,5]*a[...,9] +    \
                                (5101.*a[...,7]*a[...,9])/1120. + (54673043.*a[...,9]**2)/80640. + (13.*a[...,10]**2)/3. + (1./24)*a[...,3]*a[...,11] + (3389.*a[...,11]**2)/720. + (7./20)*a[...,4]*a[...,12] + (13./18)*a[...,10]*a[...,12] + (6799.*a[...,12]**2)/360. + (1043./160)*a[...,3]*a[...,13] +   \
                                (73./32)*a[...,11]*a[...,13] + (22846129.*a[...,13]**2)/134400. + (87617./840)*a[...,4]*a[...,14] + (13./120)*a[...,10]*a[...,14] + (306967.*a[...,12]*a[...,14])/16800. + (469977913.*a[...,14]**2)/172800. + (1./2)*a[...,5]*a[...,15] + (1./24)*a[...,7]*a[...,15] +          \
                                (1./160)*a[...,9]*a[...,15] + (3129.*a[...,15]**2)/80. + (17./30)*a[...,6]*a[...,16] + (11./80)*a[...,8]*a[...,16] + (47459.*a[...,16]**2)/1120. + (1./24)*a[...,5]*a[...,17] + (73./32)*a[...,7]*a[...,17] + (24721.*a[...,9]*a[...,17])/22400. +                             \
                                (1043./160)*a[...,15]*a[...,17] + (22846129.*a[...,17]**2)/134400. + (11./80)*a[...,6]*a[...,18] + (114997.*a[...,8]*a[...,18])/5600. + (114997.*a[...,16]*a[...,18])/5600. + (19583517.*a[...,18]**2)/12800. + (1./160)*a[...,5]*a[...,19] +                            \
                                (24721.*a[...,7]*a[...,19])/22400. + (37850569.*a[...,9]*a[...,19])/115200. + (3129.*a[...,15]*a[...,19])/3200. + (2105043.*a[...,17]*a[...,19])/12800. + (368483712607.*a[...,19]**2)/15052800. + (21./5)*a[...,10]*a[...,20] + (7./20)*a[...,12]*a[...,20] +              \
                                (21./400)*a[...,14]*a[...,20] + (87617.*a[...,20]**2)/140. + (1./160)*a[...,3]*a[...,21] + (5101.*a[...,11]*a[...,21])/1120. + (24721.*a[...,13]*a[...,21])/22400. + (54673043.*a[...,21]**2)/80640. + (21./400)*a[...,4]*a[...,22] + (7./20)*a[...,10]*a[...,22] +              \
                                (306967.*a[...,12]*a[...,22])/16800. + (7071./800)*a[...,14]*a[...,22] + (87617./840)*a[...,20]*a[...,22] + (469977913.*a[...,22]**2)/172800. + (3129.*a[...,3]*a[...,23])/3200. + (24721.*a[...,11]*a[...,23])/22400. + (2105043.*a[...,13]*a[...,23])/12800. +             \
                                (37850569.*a[...,21]*a[...,23])/115200. + (368483712607.*a[...,23]**2)/15052800. + (1./480)*a[...,1]*(240.*a[...,3] + 80.*a[...,11] + 20.*a[...,13] + 12.*a[...,21] + 3.*a[...,23]) + (87617.*a[...,4]*a[...,24])/5600. + (21./400)*a[...,10]*a[...,24] +                    \
                                (7071./800)*a[...,12]*a[...,24] + (2475532433.*a[...,14]*a[...,24])/940800. + (87617.*a[...,20]*a[...,24])/5600. + (2475532433.*a[...,22]*a[...,24])/940800. + (2210903809027.*a[...,24]**2)/5644800. +                                                      \
                                (a[...,2]*(15120.*a[...,4] + 2600.*a[...,12] + 1260.*a[...,14] + 390.*a[...,22] + 189.*a[...,24]))/3600
        return WENO_smooth_indicator
    
class weno_JS_nonlinear_weights(torch.nn.Module):
    def __init__(self):
        super(weno_JS_nonlinear_weights, self).__init__()
        return
    #@torch.compile
    def forward(self,beta,c,eps):
        alpha = c / ( beta + eps )**2
        wts = torch.nn.functional.normalize( alpha, p=1, dim=-1, eps=0 )
        return wts

class weno_M_nonlinear_weights(torch.nn.Module):
    def __init__(self):
        super(weno_M_nonlinear_weights, self).__init__()
        self.JS_weights = weno_JS_nonlinear_weights()
    #@torch.compile
    def forward(self,beta,c,eps):
        w_js = self.JS_weights(beta,c,eps)
        g = w_js * ( c + c**2 - 3 * c * w_js + w_js**2 ) / ( c**2 + w_js * ( 1. - 2. * c ) )
        wts = torch.nn.functional.normalize( g, p=1, dim=-1, eps=0 )
        
        # nelement, nstencil = beta.shape
        # wts = c.unsqueeze(0).repeat(nelement,1,1)
        return wts

class recon_class(torch.nn.Module):
    def __init__(self,mesh,recon_scheme,nvar,device):
        super(recon_class, self).__init__()
        self.mesh = mesh
        self.sw = self.mesh.sw # Stencil width
        self.rw = self.mesh.rw # recon radius
        self.gx = self.mesh.gx - 0.5 # gaussian legendre quadrature points position
        self.gw = self.mesh.gw # gaussian legendre quadrature weights
        self.device = device

        dx = self.mesh.dx
        dy = self.mesh.dy
        
        self.nq   = self.mesh.nq
        self.nCOS = self.mesh.nCOS
        self.nPOE = self.mesh.nPOE
        self.nPOR = self.mesh.nPOR
        self.nQOC = self.mesh.nQOC
        self.nPOC = self.mesh.nPOC

        self.pc  = self.mesh.pc
        self.pls = self.mesh.pls
        self.ple = self.mesh.ple
        self.prs = self.mesh.prs
        self.pre = self.mesh.pre
        self.pbs = self.mesh.pbs
        self.pbe = self.mesh.pbe
        self.pts = self.mesh.pts
        self.pte = self.mesh.pte
        self.pqs = self.mesh.pqs
        self.pqe = self.mesh.pqe
        
        # Set coordinates for polynomial coefficients
        self.xi  = torch.zeros(self.nPOR,device=self.device)
        self.eta = torch.zeros(self.nPOR,device=self.device)
        
        self.xi[self.pc          ] = 0 # Center
        self.xi[self.pls:self.ple] = -0.5 # Left
        self.xi[self.prs:self.pre] =  0.5 # Right
        self.xi[self.pbs:self.pbe] = 0 # Bottom
        self.xi[self.pts:self.pte] = 0 # Top
        
        self.eta[self.pc          ] = 0 # Center
        self.eta[self.pls:self.ple] = 0 # Left
        self.eta[self.prs:self.pre] = 0 # Right
        self.eta[self.pbs:self.pbe] = -0.5 # Bottom
        self.eta[self.pts:self.pte] =  0.5 # Top

        self.Amtx = torch.zeros(self.nCOS,self.nCOS,device=self.device)
        iCOS = 0
        for j in range(-self.rw,self.rw+1):
            for i in range(-self.rw,self.rw+1):
                x_min = i - 0.5
                x_max = x_min + 1
                y_min = j - 0.5
                y_max = y_min + 1
                self.Amtx[iCOS,:] = calc_rectangle_poly_integration(nx=self.sw,ny=self.sw,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max,device=self.device)
                iCOS = iCOS + 1

        self.iAmtx = torch.inverse(self.Amtx)
        self.Pmtx = calc_rectangle_poly_matrix(self.sw,self.sw,self.xi,self.eta)
        self.PxMtx, self.PyMtx = calc_rectangle_poly_deriv_matrix(self.sw,self.sw,self.xi,self.eta)
        self.Rmtx = torch.mm( self.Pmtx, self.iAmtx )
        self.Rmtx_conv = self.Rmtx.reshape(self.nPOR,1,self.sw,self.sw).transpose(-1,-2)
        self.DxMtx = torch.mm( self.PxMtx, self.iAmtx )
        self.DyMtx = torch.mm( self.PyMtx, self.iAmtx )

        self.DxMtxC  = self.DxMtx[self.pc,...] / dx
        self.DyMtxC  = self.DyMtx[self.pc,...] / dy
        self.DxyMtxC = torch.cat( [self.DxMtxC,self.DyMtxC], dim=0 )

        self.DxMtxC_conv  = self.DxMtxC.reshape(1,1,self.sw,self.sw).transpose(-1,-2)
        self.DyMtxC_conv  = self.DyMtxC.reshape(1,1,self.sw,self.sw).transpose(-1,-2)
        self.DxyMtxC_conv = self.DxyMtxC.reshape(2,1,self.sw,self.sw).transpose(-1,-2)

        # Recon ghost points
        xloc = ( self.mesh.gst_x - self.mesh.x[self.mesh.gst_i,self.mesh.gst_j] ) / dx
        yloc = ( self.mesh.gst_y - self.mesh.y[self.mesh.gst_i,self.mesh.gst_j] ) / dy

        n_gst_pts = self.mesh.n_gst_pts
        xloc = xloc.reshape(n_gst_pts)
        yloc = yloc.reshape(n_gst_pts)

        self.Pmtx_gst = calc_rectangle_poly_matrix(self.sw,self.sw,xloc,yloc)
        self.Rmtx_gst = torch.matmul( self.Pmtx_gst, self.iAmtx )

        # Overlap points
        xloc = ( self.mesh.gst_x2 - self.mesh.x[self.mesh.gst_i2,self.mesh.gst_j2] ) / dx
        yloc = ( self.mesh.gst_y2 - self.mesh.y[self.mesh.gst_i2,self.mesh.gst_j2] ) / dy

        n_olp_pts = 2 * self.mesh.n_olp_pts
        xloc = xloc.reshape(n_olp_pts)
        yloc = yloc.reshape(n_olp_pts)

        self.Pmtx_olp = calc_rectangle_poly_matrix(self.sw,self.sw,xloc,yloc)
        self.Rmtx_olp = torch.matmul( self.Pmtx_olp, self.iAmtx )

        # Set convolution matrices for edge integration
        xi = torch.arange(-self.rw,self.rw+1,device=self.device)
        Amtx = calc_1D_poly_matrix(self.sw,xi)
        Pmtx = calc_1D_poly_integration(self.sw,-0.5,0.5,self.device)
        iAmtx = torch.inverse(Amtx)
        self.conv_edge = torch.matmul( Pmtx, iAmtx ).view(1,1,self.sw)#.flip(-1)

        # Set convolution matrices for cell integration
        I1 = torch.arange(-self.rw,self.rw+1,device=self.device)
        J1 = torch.arange(-self.rw,self.rw+1,device=self.device)
        I, J = torch.meshgrid( I1, J1, indexing='ij' )
        I = I.reshape(self.sw**2).contiguous()
        J = J.reshape(self.sw**2).contiguous()
        Amtx = calc_rectangle_poly_matrix(self.sw,self.sw,I,J)
        iAmtx = torch.inverse(Amtx)
        Pmtx = calc_rectangle_poly_integration(self.sw,self.sw,-0.5,0.5,-0.5,0.5,self.device)
        self.conv_cell = torch.matmul( Pmtx, iAmtx ).view(1,1,self.sw,self.sw).transpose(-1,-2)

        # Set up WENO reconstruction
        if recon_scheme == 'WENO':
            n_cells_on_full_stencil = self.sw**2
            sw_sub = round( ( self.sw + 1 ) / 2 )
            full_stencil_cell_idx = torch.arange( n_cells_on_full_stencil, device=self.device ).view(self.sw,self.sw).transpose(0,1)
            ic = self.rw
            jc = ic
            x_min_cell = torch.zeros(n_cells_on_full_stencil, device=self.device)
            x_max_cell = torch.zeros(n_cells_on_full_stencil, device=self.device)
            y_min_cell = torch.zeros(n_cells_on_full_stencil, device=self.device)
            y_max_cell = torch.zeros(n_cells_on_full_stencil, device=self.device)
            icell = 0
            for j in range(self.sw):
                for i in range(self.sw):
                    x_min_cell[icell] = i - ic - 0.5
                    x_max_cell[icell] = x_min_cell[icell] + 1
                    y_min_cell[icell] = j - jc - 0.5
                    y_max_cell[icell] = y_min_cell[icell] + 1
                    icell += 1

            n_sub_stencil = sw_sub**2
            n_cells_on_sub_stencil = n_sub_stencil
            sub_is = 0
            sub_ie = ic
            sub_js = 0
            sub_je = jc
            sub_stencil_cell_idx = torch.zeros( n_sub_stencil, n_cells_on_sub_stencil, dtype=torch.int, device=self.device )
            i_sub_stencil = -1
            for j in range(sub_js,sub_je+1):
                for i in range(sub_is,sub_ie+1):
                    i_sub_stencil += 1
                    icell_on_sub_stencil = 0
                    for jsc in range(sw_sub):
                        for isc in range(sw_sub):
                            sub_stencil_cell_idx[i_sub_stencil,icell_on_sub_stencil] = full_stencil_cell_idx[i+isc,j+jsc]
                            icell_on_sub_stencil += 1
            
            A = torch.zeros( n_sub_stencil, n_cells_on_sub_stencil, n_cells_on_sub_stencil, device=self.device )
            for i_sub_stencil in range(n_sub_stencil):
                for icell_on_sub_stencil in range(n_cells_on_sub_stencil):
                    idx = sub_stencil_cell_idx[i_sub_stencil,icell_on_sub_stencil]
                    x_min = x_min_cell[idx]
                    x_max = x_max_cell[idx]
                    y_min = y_min_cell[idx]
                    y_max = y_max_cell[idx]
                    A[i_sub_stencil,icell_on_sub_stencil,:] = calc_rectangle_poly_integration(sw_sub,sw_sub,x_min,x_max,y_min,y_max,self.device)
            
            iA = torch.inverse(A)
            weno_iA = iA

            Pmtx = calc_rectangle_poly_matrix(sw_sub,sw_sub,self.xi,self.eta)
            weno_Pmtx = Pmtx
            
            Rmtx_sub = torch.matmul( Pmtx, iA )

            R = torch.zeros( self.nPOR, n_sub_stencil, n_cells_on_full_stencil, device=self.device )
            for i_sub_stencil in range(n_sub_stencil):
                R[:,i_sub_stencil,sub_stencil_cell_idx[i_sub_stencil,:]] = Rmtx_sub[i_sub_stencil,:,:]
            RT = R.transpose(-1,-2)
            RTR = torch.matmul( R, RT )
            iRTR = torch.inverse( RTR )
            Amtx = torch.matmul( iRTR, R )

            b = self.Rmtx.view(self.nPOR,n_cells_on_full_stencil,1)

            c = torch.matmul( Amtx, b ).squeeze()

            weno_opt_coef = c

            # Post check
            opt_coef = c.unsqueeze(-1).expand(-1,-1,n_cells_on_full_stencil) * R
            opt_wgt = torch.sum( opt_coef, dim=1 )
            diff = torch.max( ( opt_wgt - self.Rmtx ) / self.Rmtx )
            # diff = torch.max( (opt_wgt - self.Rmtx).abs() )
            print('')
            print('Maximum WENO optimal coef residual ',diff.item())
            
            theta = 3
            weno_rp = 0.5 * ( c + theta * c.abs() )
            weno_rn = weno_rp - c
            weno_sigmap = torch.sum(weno_rp,dim=1,keepdim=True).expand(-1,n_sub_stencil)
            weno_sigman = torch.sum(weno_rn,dim=1,keepdim=True).expand(-1,n_sub_stencil)
            weno_rp = weno_rp / weno_sigmap
            weno_rn = weno_rn / weno_sigman
            
            if sw_sub==2:
                weno_smooth_indicator = weno_smooth_indicator_2()
            elif sw_sub==3:
                weno_smooth_indicator = weno_smooth_indicator_3()
            elif sw_sub==4:
                weno_smooth_indicator = weno_smooth_indicator_4()
            elif sw_sub==5:
                weno_smooth_indicator = weno_smooth_indicator_5()
            else:
                raise ValueError('Unknown stencil width, choose from 3,5,7,9')

            weno_nonlinear_weights = weno_M_nonlinear_weights()
    
        if recon_scheme == 'TPP':
            self.recon = self.tpp_recon_class(self.Rmtx_conv,nvar,mesh,self.device)
        elif recon_scheme == 'WENO':
            self.recon = self.weno_recon_class(sw_sub, sub_stencil_cell_idx, \
                                               weno_opt_coef, weno_iA, weno_Pmtx,\
                                               weno_rp, weno_rn, weno_sigmap, weno_sigman,\
                                               weno_smooth_indicator, weno_nonlinear_weights)

    class tpp_recon_class(torch.nn.Module):
        def __init__(self,Rmtx_conv,nvar,mesh,device,prec_mode=prec_mode):
            super().__init__()
            self.Rmtx_conv = Rmtx_conv

            yworksize = 2**16*10
            self.d_x = torch.zeros(yworksize, dtype=torch.float32, device=device)
            self.d_y = torch.zeros(yworksize, dtype=torch.float32, device=device)
            self.handle = handle = cudnn.create_handle()
            self.prec_mode = prec_mode
            self.input_type64 = torch.get_default_dtype()
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
            size_img = [nvar*mesh.npanel, 1, mesh.nx_halo, mesh.ny_halo]
            size_knl = list(self.Rmtx_conv.shape)
            size_out = [nvar*mesh.npanel, mesh.nPOR, mesh.nrx, mesh.nry]
            n = size_img[0]
            c = size_img[1]
            h = size_img[2]
            w = size_img[3]
            k = size_knl[0]
            r = size_knl[2]
            s = size_knl[3]
            self.tensor_float64 = torch.zeros(size_img, dtype=self.input_type64, device=device)
            self.vector_float64 = torch.zeros(size_knl, dtype=self.input_type64, device=device)
            self.y64 = torch.zeros(size_out, dtype=self.output_type64, device=device)

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
            self.workspace64 = torch.empty(self.convgraph64.get_workspace_size(), device=device, dtype=torch.uint8)

            # FP16-FP32 cudnn settings
            self.input_type  = torch.float16
            self.output_type = torch.float32
            self.convgraph = cudnn.pygraph(
                handle=handle,
                io_data_type=cudnn.data_type.HALF,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT
            )

            if OZ_MODE == OZ_BSHALF:
                        size_img = [(nvar*mesh.npanel)//2, OZCUDNN_CHNLS, mesh.nx_halo, mesh.ny_halo]
            elif OZ_MODE == OZ_NORMAL:
                        size_img = [nvar*mesh.npanel, OZCUDNN_CHNLS, mesh.nx_halo, mesh.ny_halo]
            size_knl = list(self.Rmtx_conv.shape)
            size_knl[0] = OZCUDNN_OUTCHNLS
            size_knl[1] = OZCUDNN_CHNLS
            size_out    = [nvar*mesh.npanel, OZCUDNN_OUTCHNLS, mesh.nrx, mesh.nry]
            size_out_dp = [nvar*mesh.npanel,        mesh.nPOR, mesh.nrx, mesh.nry]
            n = size_img[0]
            c = size_img[1]
            h = size_img[2]
            w = size_img[3]
            k = size_knl[0]
            r = size_knl[2]
            s = size_knl[3]
            size_scl = [1,  k, 1, 1]
            self.tensor_float32 = torch.zeros(size_img, dtype=self.input_type, device=device)
            self.vector_float32 = torch.zeros(size_knl, dtype=self.input_type, device=device)
            self.y    = torch.empty([n, k, h-(size_knl[2]-1), h-(size_knl[3]-1)], dtype=self.output_type, device=device, memory_format=torch.channels_last)
            self.y.zero_()
            self.y_dp = torch.empty(size_out_dp, dtype=torch.double, device=device)
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
            self.workspace = torch.empty(self.convgraph.get_workspace_size(), device=device, dtype=torch.uint8)
            self.tensor_float32 = self.tensor_float32.to(memory_format=torch.channels_last)
            self.vector_float32 = self.vector_float32.to(memory_format=torch.channels_last)

        def oz_conv(self, A, C):
            outbs = C.shape[0]
            knlsize = [C.shape[2], C.shape[3]]
            gp.start("split")
            torch.ops.my_ops.custom_ozcudnn(A, C, self.y_dp, self.tensor_float32, self.d_y, torch.tensor([OZ_MODE],dtype=torch.int))
            d_yreshp = self.d_y[:numsplit*outbs*OZCUDNN_CHNLS*knlsize[0]*knlsize[1]].reshape([numsplit*outbs,OZCUDNN_CHNLS,knlsize[0],knlsize[1]])
            curr_ind = 0
            for i in range(numsplit):
                self.vector_float32[outbs*i:outbs*(i+1),curr_ind:curr_ind+(i+1),:,:] = d_yreshp[outbs*i:outbs*(i+1),curr_ind:curr_ind+(i+1),:,:]
            if OZ_MODE == OZ_BSHALF:
                self.vector_float32[outbs*numsplit:2*outbs*numsplit, numsplit:2*numsplit,:,:] = self.vector_float32[:outbs*numsplit, 0:numsplit,:,:]
            gp.stop("split")
            gp.start("compute")
            variant_pack = {
                self.x_cudnn_tensor: self.tensor_float32,
                self.w_cudnn_tensor: self.vector_float32,
                self.y_cudnn_tensor_casted: self.y,
            }
            self.convgraph.execute(variant_pack, self.workspace)
            gp.stop("compute")
            gp.start("accum")
            B = torch.ops.my_ops.custom_accumulate_ozcudnn(A, C, self.y_dp, self.d_x, self.d_y, self.y, torch.tensor([OZ_MODE],dtype=torch.int))
            Brshp = B.reshape([B.shape[0],B.shape[2],B.shape[3],outbs])
            B = Brshp.permute([0,3,1,2])
            gp.stop("accum")
            return B

        def forward(self,field):
            if self.prec_mode == 0:
                field1 = field.to(memory_format=torch.channels_last)
                Rmtx_conv1 = self.Rmtx_conv.to(memory_format=torch.channels_last)
                variant_pack = {
                    self.x_cudnn_tensor64: field1,
                    self.w_cudnn_tensor64: Rmtx_conv1,
                    self.y_cudnn_tensor64: self.y64,
                }
                self.convgraph64.execute(variant_pack, self.workspace64)
                q = self.y64.reshape([field.shape[0],self.y64.shape[2],self.y64.shape[3],self.Rmtx_conv.shape[0]]).permute([0,3,1,2])
            elif self.prec_mode == 1:
                field1 = field.to(memory_format=torch.channels_last)
                Rmtx_conv1 = self.Rmtx_conv.to(memory_format=torch.channels_last)
                field1[0:6] = field[0:6] / 2**16
                Rmtx_conv1 = Rmtx_conv1 / 4
                q = self.oz_conv(field1, Rmtx_conv1)
                q[0:6] = q[0:6] * 2**16
                q = q * 4 /  4 **(bits_per_slice-7) 
            return q
    
    class weno_recon_class(torch.nn.Module):
        def __init__(self, sw_sub, sub_stencil_cell_idx, \
                     opt_coef, iA, Pmtx,\
                     rp, rn, sigmap, sigman,\
                     smooth_indicator, nonlinear_weights):
            super().__init__()
            self.Pmtx = Pmtx
            self.iA = iA
            self.n_sub_stencil, self.n_cells_on_sub_stencil, self.n_terms_on_sub_stencil = iA.shape
            self.nPOR = Pmtx.shape[0]
            self.sw_sub = sw_sub
            self.sw = 2 * sw_sub - 1
            self.sub_stencil_cell_idx = sub_stencil_cell_idx
            self.opt_coef = opt_coef
            self.rp = rp
            self.rn = rn
            self.sigmap = sigmap
            self.sigman = sigman
            self.smooth_indicator = smooth_indicator
            self.nonlinear_weights = nonlinear_weights

            digit_type = torch.get_default_dtype()
            if digit_type==torch.float64:
                self.eps = 1.e-14
            elif digit_type==torch.float32:
                self.eps = 1.e-5
            
        #@torch.compile
        def forward(self,field):
            qc = field.unfold(3,self.sw,1).unfold(2,self.sw,1) # im2col
            nvar_npanel, nx, ny, swx, swy = qc.squeeze().shape
            nelement = nvar_npanel*nx*ny
            npts = self.Pmtx.shape[0]
            qc = qc.reshape(nelement,swx*swy)
            qc = qc[:,self.sub_stencil_cell_idx].unsqueeze(-1)
            a = torch.matmul( self.iA, qc ).squeeze() # nelments, nstencils, nterms
            beta = self.smooth_indicator(a).unsqueeze(-2).expand(-1,npts,-1)
            alpha_p = self.nonlinear_weights(beta,self.rp,self.eps)
            alpha_n = self.nonlinear_weights(beta,self.rn,self.eps)
            wp = torch.nn.functional.normalize( alpha_p, p=1, dim=-1, eps=0 )
            wn = torch.nn.functional.normalize( alpha_n, p=1, dim=-1, eps=0 )
            wa_p = torch.matmul( wp, a ) * self.sigmap
            wa_n = torch.matmul( wn, a ) * self.sigman
            w = wa_p - wa_n
            q = torch.sum( w * self.Pmtx, dim=-1 ).view(nvar_npanel,nx,ny,npts)
            q = q.permute(0,3,1,2)
            return q

    #@torch.compile
    def forward(self,field):
        q = self.recon(field)
        return q
