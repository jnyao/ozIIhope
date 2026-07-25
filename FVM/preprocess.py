import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy
import math
from hinterp import interp_latlon_to_cube_class
from recon import recon_class
from diag import pause, plot_cube_field
from ncio import netcdf_read
from mesh import mesh_class
from quadrature import _precompute_grid

class preprocess_class(torch.nn.Module):
    def __init__(self,case_num,nbatch,nt,nx,ny,nz,npanel,rw,parallel):
        super(preprocess_class, self).__init__()
        self.nbatch = nbatch
        self.nt = nt
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.npanel = npanel
        self.rw = rw
        self.parallel = parallel
        self.case_num = case_num
        self.device = self.parallel.device

        if self.case_num==0:
            self.case = case0_class(nbatch,nt,nx,ny,nz,npanel,rw,parallel)
        elif self.case_num==2:
            self.case = case2_class(nbatch,nt,nx,ny,nz,npanel,rw,parallel)
        elif self.case_num==5:
            self.case = case5_class(nbatch,nt,nx,ny,nz,npanel,rw,parallel)
        elif self.case_num==6:
            self.case = case6_class(nbatch,nt,nx,ny,nz,npanel,rw,parallel)
        elif self.case_num==8:
            self.case = case8_class(nbatch,nt,nx,ny,nz,npanel,rw,parallel)
        elif self.case_num==9:
            self.case = case9_class(nbatch,nt,nx,ny,nz,npanel,rw,parallel)
        elif self.case_num==-1:
            self.case = casen1_class(nbatch,nt,nx,ny,nz,npanel,rw,parallel)
        
        self.gravity = self.case.gravity
        self.radius = self.case.radius
        self.Omega = self.case.Omega
        self.mesh = self.case.mesh
        self.Coriolis = self.case.Coriolis
        self.ghsL = self.case.ghsL
        self.ghsB = self.case.ghsB
        self.ghsQ = self.case.ghsQ
        self.ghs  = self.case.ghs
    
    def forward(self):
        q = self.case()
        return q
    
class casen1_class(torch.nn.Module):
    def __init__(self,nbatch,nt,nx,ny,nz,npanel,rw,parallel):
        super(casen1_class, self).__init__()
        self.nbatch = nbatch
        self.nt = nt
        self.parallel = parallel
        self.radius = 6371220.
        self.gravity = 9.80616
        self.Omega = 7.292E-5

        self.mesh = mesh_class('cubed_sphere',parallel,nx,ny,nz,npanel,rw,self.radius).to(parallel.device)

        lon = self.mesh.lon
        lat = self.mesh.lat
        
        self.Coriolis = 2. * self.Omega * torch.sin(lat)

        # alpha = 2.
        # tau = 0.0001
        # self.gh_GRF = GaussianRF(dim=2, size=nx, alpha=alpha, tau=tau, device=parallel.device)

        # alpha = 2.
        # tau = 0.0001
        # self.u_GRF = GaussianRF(dim=2, size=nx, alpha=alpha, tau=tau, device=parallel.device)

        # alpha = 2.
        # tau = 0.0001
        # self.v_GRF = GaussianRF(dim=2, size=nx, alpha=alpha, tau=tau, device=parallel.device)

        self.ghs  = torch.zeros_like(self.mesh.lon )
        self.ghsL = torch.zeros_like(self.mesh.lonL)
        self.ghsB = torch.zeros_like(self.mesh.lonB)
        self.ghsQ = torch.zeros_like(self.mesh.lonQ)
    
    def forward(self):
        q = torch.zeros_like( self.mesh.lon, device=self.parallel.device )
        # gh = self.gh_GRF(self.nbatch)
        # u  = self.u_GRF (self.nbatch)
        # v  = self.v_GRF (self.nbatch)
        # ght = gh + self.ghsQ

        # ghtQ = ght[0,...]
        # ghQ = ghtQ - self.ghsQ
        # usQ = u[1,...]
        # vsQ = v[2,...]
        # # Set variables on cubed sphere
        # q1 = self.mesh.jabQ * ghtQ
        # uc, vc = self.mesh.contravProjSphere2Plane(self.mesh.iAQ, usQ, vsQ)
        # q2 = self.mesh.jabQ * ghQ * uc
        # q3 = self.mesh.jabQ * ghQ * vc
        # q = torch.stack([q1,q2,q3],dim=0) # nvar, npanel, npts, nx, ny
        # q = torch.einsum('vnpij,p->vnij',q,self.mesh.gw2d).requires_grad_(True)
        return q

class case0_class(torch.nn.Module):
    def __init__(self,nbatch,nt,nx,ny,nz,npanel,rw,parallel):
        super(case0_class, self).__init__()
        pi = math.pi
        self.D2R = pi / 180.
        self.parallel = parallel
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.npanel = npanel
        self.rw = rw
        self.radius = 6371220
        self.Omega = 7.292E-5
        self.gravity = 9.80616
        self.stride = 4
        self.ilev = 22
        self.nc_file = 'ERA5-pl_2022041100.nc'
        self.gh_file = 'ERA5_gh.nc'
        self.mesh = mesh_class('cubed_sphere',parallel,nx,ny,nz,npanel,rw,self.radius).to(parallel.device)
        self.Coriolis = 2. * self.Omega * torch.sin(self.mesh.lat)

        lon = netcdf_read(self.nc_file, 'longitude', self.parallel.device)
        lat = torch.flip( netcdf_read(self.nc_file, 'latitude', self.parallel.device), dims=[0] )
        self.lon = lon[::self.stride] * self.D2R
        self.lat = lat[::self.stride] * self.D2R
        dlon = self.lon[1] - self.lon[0]
        dlat = self.lat[1] - self.lat[0]
        self.interp  = interp_latlon_to_cube_class(self.lat,self.lon,dlat,dlon,self.mesh.lat ,self.mesh.lon )
        self.interpL = interp_latlon_to_cube_class(self.lat,self.lon,dlat,dlon,self.mesh.latL,self.mesh.lonL)
        self.interpB = interp_latlon_to_cube_class(self.lat,self.lon,dlat,dlon,self.mesh.latB,self.mesh.lonB)
        self.interpQ = interp_latlon_to_cube_class(self.lat,self.lon,dlat,dlon,self.mesh.latQ,self.mesh.lonQ)
        ghs = torch.squeeze( torch.flip( netcdf_read(self.gh_file, 'z', self.parallel.device), dims=[1] ).type(torch.float32) )
        ghs = ghs[::self.stride,::self.stride] * 0

        self.ghs  = torch.squeeze( self.interp (ghs) )
        self.ghsL = torch.squeeze( self.interpL(ghs) )
        self.ghsB = torch.squeeze( self.interpB(ghs) )
        self.ghsQ = torch.squeeze( self.interpQ(ghs) )

    def forward(self):
        gh  = torch.flip( netcdf_read(self.nc_file, 'z', self.parallel.device), dims=[2] )#.requires_grad_(True)
        u   = torch.flip( netcdf_read(self.nc_file, 'u', self.parallel.device), dims=[2] )#.requires_grad_(True)
        v   = torch.flip( netcdf_read(self.nc_file, 'v', self.parallel.device), dims=[2] )#.requires_grad_(True)
        ght = torch.squeeze( gh[0,self.ilev-1,::self.stride,::self.stride] ).type(torch.float32)
        u   = torch.squeeze( u [0,self.ilev-1,::self.stride,::self.stride] ).type(torch.float32)
        v   = torch.squeeze( v [0,self.ilev-1,::self.stride,::self.stride] ).type(torch.float32)
        print( 'min/max value of u',torch.min(u).item(), torch.max(u).item() )
        print( 'min/max value of v',torch.min(v).item(), torch.max(v).item() )

        qc = torch.stack( [ ght, u, v ], dim=0)
        qc = self.interp( qc )
        ght = qc[0,...]
        gh = ght - self.ghs
        us = qc[1,...]
        vs = qc[2,...]

        # Set variables on cubed sphere
        jab = self.mesh.jab
        q1 = jab * ght
        uc, vc = self.mesh.contravProjSphere2Plane(self.mesh.iA, us, vs)
        q2 = jab * gh * uc
        q3 = jab * gh * vc
        q = torch.stack([q1,q2,q3],dim=0) # nvar, npanel, npts, nx, ny
        return q
    
class case2_class(torch.nn.Module):
    def __init__(self,nbatch,nt,nx,ny,nz,npanel,rw,parallel):
        super(case2_class, self).__init__()
        pi = math.pi
        self.parallel = parallel
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.npanel = npanel
        self.rw = rw
        self.radius = 6371220.
        self.u0 = 2. * pi * self.radius / ( 12. * 86400. )
        self.gh0 = 29400
        self.alpha = torch.tensor(0.)
        self.Omega = 7.292E-5
        self.gravity = 9.80616

        self.mesh = mesh_class('cubed_sphere',parallel,nx,ny,nz,npanel,rw,self.radius).to(parallel.device)
        self.lon = self.mesh.lon
        self.lat = self.mesh.lat

        lon = self.mesh.lon
        lat = self.mesh.lat
        Omega = self.Omega
        alpha = self.alpha
        self.Coriolis = 2. * Omega * ( -torch.cos(lon)*torch.cos(lat)*torch.sin(alpha) + torch.sin(lat)*torch.cos(alpha) )

        self.ghs  = torch.zeros_like(self.mesh.lon )
        self.ghsL = torch.zeros_like(self.mesh.lonL)
        self.ghsB = torch.zeros_like(self.mesh.lonB)
        self.ghsQ = torch.zeros_like(self.mesh.lonQ)

    def forward(self):
        u0 = self.u0
        alpha = self.alpha
        gh0 = self.gh0
        Omega = self.Omega
        radius = self.mesh.r
        lon = self.lon
        lat = self.lat
        mesh = self.mesh

        gh = gh0 - (radius * Omega * u0 + u0**2 / 2.) * ( -torch.cos(lon)*torch.cos(lat)*torch.sin(alpha) + torch.sin(lat)*torch.cos(alpha) )**2
        u = u0 * ( torch.cos(lat)*torch.cos(alpha) + torch.cos(lon)*torch.sin(lat)*torch.sin(alpha) )
        v = -u0 * torch.sin(lon) * torch.sin(alpha)
        ght = gh + self.ghs

        qc = torch.stack( [ ght, u, v ], dim=0)
        ght = qc[0,...]
        gh = ght - self.ghs
        us = qc[1,...]
        vs = qc[2,...]
        # Set variables on cubed sphere
        jab = mesh.jab
        q1 = jab * ght
        uc, vc = mesh.contravProjSphere2Plane(mesh.iA, us, vs)
        q2 = jab * gh * uc
        q3 = jab * gh * vc
        q = torch.stack([q1,q2,q3],dim=0) # nvar, npanel, npts, nx, ny
        
        return q
    
class case5_class(torch.nn.Module):
    def __init__(self,nbatch,nt,nx,ny,nz,npanel,rw,parallel):
        super(case5_class, self).__init__()
        pi = math.pi
        self.parallel = parallel
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.npanel = npanel
        self.rw = rw
        self.radius = 6371220.
        self.Omega = 7.292E-5

        self.mesh = mesh_class('cubed_sphere',parallel,nx,ny,nz,npanel,rw,self.radius).to(parallel.device)
        self.lon = self.mesh.lon
        self.lat = self.mesh.lat
        self.gravity = 9.80616

        self.alpha = torch.tensor(0.)
        self.ghs0 = 2000. * self.gravity
        self.gh0 = 5960. * self.gravity
        self.u0  = 20.

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

        rr = torch.tensor( pi / 9. )
        lambda_c = 1.5 * pi
        theta_c = pi / 6.

        lon = self.mesh.lon
        lat = self.mesh.lat
        Omega = self.Omega
        alpha = self.alpha
        self.Coriolis = 2. * Omega * ( -torch.cos(lon)*torch.cos(lat)*torch.sin(alpha) + torch.sin(lat)*torch.cos(alpha) )
        r = torch.sqrt( torch.min(rr**2,(lon-lambda_c)**2+(lat-theta_c)**2) )
        self.ghs = self.ghs0 * ( 1. - r / rr )

        lon = self.mesh.lonL
        lat = self.mesh.latL
        r = torch.sqrt( torch.min(rr**2,(lon-lambda_c)**2+(lat-theta_c)**2) )
        self.ghsL = self.ghs0 * ( 1. - r / rr )

        lon = self.mesh.lonB
        lat = self.mesh.latB
        r = torch.sqrt( torch.min(rr**2,(lon-lambda_c)**2+(lat-theta_c)**2) )
        self.ghsB = self.ghs0 * ( 1. - r / rr )

        lon = self.mesh.lonQ
        lat = self.mesh.latQ
        r = torch.sqrt( torch.min(rr**2,(lon-lambda_c)**2+(lat-theta_c)**2) )
        self.ghsQ = self.ghs0 * ( 1. - r / rr )

    def forward(self):
        u0 = self.u0
        alpha = self.alpha
        gh0 = self.gh0
        Omega = self.Omega
        radius = self.mesh.r
        lon = self.lon
        lat = self.lat
        mesh = self.mesh
        ghs = self.ghs
        
        u = u0*(torch.cos(lat)*torch.cos(alpha)+torch.cos(lon)*torch.sin(lat)*torch.sin(alpha))
        v = -u0*torch.sin(lon)*torch.sin(alpha)
        gh = gh0 - (radius * Omega * u0 + u0**2 / 2.) * ( -torch.cos(lon)*torch.cos(lat)*torch.sin(alpha) + torch.sin(lat)*torch.cos(alpha) )**2 - ghs
        ght = gh + ghs

        qc = torch.stack( [ ght, u, v ], dim=0)
        ght = qc[0,...]
        gh = ght - self.ghs
        us = qc[1,...]
        vs = qc[2,...]
        # Set variables on cubed sphere
        jab = mesh.jab
        q1 = jab * ght
        uc, vc = mesh.contravProjSphere2Plane(mesh.iA, us, vs)
        q2 = jab * gh * uc
        q3 = jab * gh * vc
        q = torch.stack([q1,q2,q3],dim=0) # nvar, npanel, npts, nx, ny
        return q
    
class case6_class(torch.nn.Module):
    def __init__(self,nbatch,nt,nx,ny,nz,npanel,rw,parallel):
        super(case6_class, self).__init__()
        pi = math.pi
        self.parallel = parallel
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.npanel = npanel
        self.rw = rw
        self.radius = 6371220.
        self.Omega = 7.292E-5
        self.gravity = 9.80616

        self.mesh = mesh_class('cubed_sphere',parallel,nx,ny,nz,npanel,rw,self.radius).to(parallel.device)
        self.lon = self.mesh.lon
        self.lat = self.mesh.lat

        h0 = 8000.
        self.omg = 7.848e-6
        self.R = 4
        self.gh0 = h0 * self.gravity

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

        lon = self.lon
        lat = self.lat
        Omega = self.Omega
        self.Coriolis = 2. * Omega * torch.sin(lat)

        self.ghs  = torch.zeros_like(self.mesh.lon )
        self.ghsL = torch.zeros_like(self.mesh.lonL)
        self.ghsB = torch.zeros_like(self.mesh.lonB)
        self.ghsQ = torch.zeros_like(self.mesh.lonQ)

    def forward(self):
        mesh = self.mesh
        gh0 = self.gh0
        omg = self.omg
        Omega = self.Omega
        R = self.R
        radius = self.mesh.r
        lon = self.lon
        lat = self.lat
        ghs = self.ghs

        # FVM version
        u1 = torch.cos(lat)
        u2 = R*torch.cos(lat)**(R-1)*torch.sin(lat)**2*torch.cos(R*lon)
        u3 = torch.cos(lat)**(R+1)*torch.cos(R*lon)
        u  = radius*omg*(u1+u2-u3)
        
        v  = -radius*omg*R*torch.cos(lat)**(R-1)*torch.sin(lat)*torch.sin(R*lon)
        
        AA1 = omg*0.5*(2.*Omega+omg)*torch.cos(lat)**2
        Ac  = 0.25*omg**2
        A21 = (R+1.)*torch.cos(lat)**(2.*R+2.)
        A22 = (2.*R**2-R-2.)*torch.cos(lat)**(2.*R)
        A23 = 2.*R**2*torch.cos(lat)**(2.*R-2)
        Ah  = AA1+Ac*(A21+A22-A23)
        
        Bc  = 2.*(Omega+omg)*omg/((R+1)*(R+2))*torch.cos(lat)**R
        BB1 = R**2+2.*R+2.
        BB2 = (R+1.)**2.*torch.cos(lat)**2.
        Bh  = Bc*(BB1-BB2)
        
        CC  = 0.25*omg**2*torch.cos(lat)**(2.*R)
        CC1 = (R+1.)*torch.cos(lat)**2
        CC2 = R+2.
        Ch  = CC*(CC1-CC2)
        gh  = gh0+radius**2*(Ah + Bh*torch.cos(R*lon) + Ch*torch.cos(2.0*R*lon))

        ght = gh + ghs

        qc = torch.stack( [ ght, u, v ], dim=0)
        ght = qc[0,...]
        gh = ght - self.ghs
        us = qc[1,...]
        vs = qc[2,...]
        # Set variables on cubed sphere
        jab = mesh.jab
        q1 = jab * ght
        uc, vc = mesh.contravProjSphere2Plane(mesh.iA, us, vs)
        q2 = jab * gh * uc
        q3 = jab * gh * vc
        q = torch.stack([q1,q2,q3],dim=0) # nvar, npanel, npts, nx, ny
        return q
    
class case8_class(torch.nn.Module):
    def __init__(self,nbatch,nt,nx,ny,nz,npanel,rw,parallel):
        super(case8_class, self).__init__()
        pi = math.pi
        self.parallel = parallel
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.npanel = npanel
        self.rw = rw

        self.radius = 6371220.
        self.Omega = 7.292E-5
        self.gravity = 9.80616

        self.mesh = mesh_class('cubed_sphere',parallel,nx,ny,nz,npanel,rw,self.radius).to(parallel.device)
        self.lon = self.mesh.lon
        self.lat = self.mesh.lat

        h0 = 10000.
        self.gh0 = h0 * self.gravity
        self.umax = 80.
        self.lat0 = pi / 7.
        self.lat1 = pi / 2. - self.lat0
        self.lat2 = pi / 4.
        self.en = torch.exp( torch.tensor( -4. / ( self.lat1 - self.lat0 )**2 ) )
        self.ghd = self.gravity * 120.
        self.alpha = 1. / 3.
        self.beta = 1. / 15.

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

        lon = self.lon
        lat = self.lat
        Omega = self.Omega
        self.Coriolis = 2. * Omega * torch.sin(lat)

        self.ghs  = torch.zeros_like(self.mesh.lon )
        self.ghsL = torch.zeros_like(self.mesh.lonL)
        self.ghsB = torch.zeros_like(self.mesh.lonB)
        self.ghsQ = torch.zeros_like(self.mesh.lonQ)

        self.digit_type = torch.get_default_dtype()
        if self.digit_type==torch.float64:
            self.relative_tol = 1.e-9
        elif self.digit_type==torch.float32:
            self.relative_tol = 1.e-1

        self.ng = 50
        self.gx, self.gw = _precompute_grid(self.ng, grid="legendre-gauss", a=0.0, b=1.0, periodic=False)
        digit_type = torch.get_default_dtype()
        self.gx = torch.tensor( self.gx, dtype=digit_type, device=self.parallel.device )
        self.gw = torch.tensor( self.gw, dtype=digit_type, device=self.parallel.device )

    def u_function(self,lat):
        umax = self.umax
        en = self.en
        lat0 = self.lat0
        lat1 = self.lat1

        u = umax / en * torch.exp(1. / (lat - lat0) / (lat - lat1))
        u = torch.where( lat<lat0, 0., u )
        u = torch.where( lat>lat1, 0., u )
        return u
    
    def gh_integrand(self,lat):
        Omega = self.Omega
        radius = self.mesh.r
        f = 2 * self.Omega * torch.sin(lat)

        u = self.u_function(lat)
        res = radius * u * ( f + torch.tan(lat) / radius * u )
        return res
    
    def integration_1d_local(self,func,xs,xe):
        x1 = xs.unsqueeze(-1).expand(-1,-1,-1,-1,self.ng)
        x2 = xe.unsqueeze(-1).expand(-1,-1,-1,-1,self.ng)
        x = x1 + ( x2 - x1 ) * self.gx
        q = func(x)
        quad = torch.matmul( q, self.gw )
        return quad
    
    def integration_1d(self,func,xs,xe,relative_tol=1e-10):
        quad_prev = 0
        residual = 1
        n_divide = 1

        while residual > relative_tol:
            n_divide = n_divide * 2
            dx = (xe-xs) / n_divide
            x = torch.ones_like(xe) * dx
            x = x.unsqueeze(0).expand(n_divide+1,-1,-1,-1,-1)
            x = xs + torch.cumsum( x, dim=0 ) - dx
            quad = torch.zeros_like(x[0,...])
            for i in range(n_divide):
                xs_local = x[i  ,...]
                xe_local = x[i+1,...]
                quad_local = self.integration_1d_local(func,xs_local,xe_local)
                quad = quad + quad_local
            quad = quad * dx
            residual = torch.max( torch.abs( quad - quad_prev ) )
            quad_prev = quad
            print('case8 integral n_divide, residual',n_divide,residual)
        return quad

    def forward(self):
        pi = math.pi
        mesh = self.mesh
        gh0 = self.gh0
        ghd = self.ghd
        alpha = self.alpha
        beta = self.beta
        lat1 = self.lat1
        lat2 = self.lat2
        Omega = self.Omega
        radius = self.mesh.r
        lon = self.lon.clone()
        lat = self.lat
        ghs = self.ghs
        
        lon = torch.where( lon>pi, lon-2*pi, lon )
        lon = torch.where( lon<-pi, lon+2*pi, lon )

        u = self.u_function(lat)
        v = torch.zeros_like(lat)
        
        lat_s = -pi/2 * torch.ones_like(lat)
        lat_e = lat
        gh = gh0 - self.integration_1d( self.gh_integrand, lat_s, lat_e, self.relative_tol )
        ght = gh + ghs + ghd * torch.cos(lat) * torch.exp(-(lon / alpha)**2) * torch.exp(-((lat2 - lat) / beta)**2)

        qc = torch.stack( [ torch.squeeze(ght), u, v ], dim=0)
        ght = qc[0,...]
        gh = ght - self.ghs
        us = qc[1,...]
        vs = qc[2,...]
        # Set variables on cubed sphere
        jab = mesh.jab
        q1 = jab * ght
        uc, vc = mesh.contravProjSphere2Plane(mesh.iA, us, vs)
        q2 = jab * gh * uc
        q3 = jab * gh * vc
        q = torch.stack([q1,q2,q3],dim=0) # nvar, npanel, npts, nx, ny
        return q
    
class case9_class(torch.nn.Module):
    def __init__(self,nbatch,nt,nx,ny,nz,npanel,rw,parallel):
        super(case9_class, self).__init__()
        pi = math.pi
        D2R = pi / 180.
        R2D = 180 / pi
        self.parallel = parallel
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.npanel = npanel
        self.rw = rw

        self.radius = 6371220.
        self.Omega = 7.292E-5
        self.gravity = 9.80616

        self.mesh = mesh_class('cubed_sphere',parallel,nx,ny,nz,npanel,rw,self.radius).to(parallel.device)
        self.lon = self.mesh.lon
        self.lat = self.mesh.lat

        self.gh0 = 30000.
        self.rc = pi / 9.
        self.lon_c = 180. * D2R
        self.lat_c = 0

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

        lon = self.lon
        lat = self.lat
        Omega = self.Omega
        self.Coriolis = torch.zeros_like(lat)

        self.ghs  = torch.zeros_like(self.mesh.lon )
        self.ghsL = torch.zeros_like(self.mesh.lonL)
        self.ghsB = torch.zeros_like(self.mesh.lonB)
        self.ghsQ = torch.zeros_like(self.mesh.lonQ)

    def forward(self):
        pi = math.pi
        mesh = self.mesh
        rc = self.rc
        gh0 = self.gh0
        lon_c = self.lon_c
        lat_c = self.lat_c
        lon = self.lon
        lat = self.lat
        
        u = torch.zeros_like(lon)
        v = torch.zeros_like(lon)
        
        r = torch.sqrt( (lon-lon_c)**2 + (lat-lat_c)**2 )
        ght = torch.where( r>rc, gh0, 2*gh0 )

        qc = torch.stack( [ ght, u, v ], dim=0)
        ght = qc[0,...]
        gh = ght - self.ghs
        us = qc[1,...]
        vs = qc[2,...]
        # Set variables on cubed sphere
        jab = mesh.jab
        q1 = jab * ght
        uc, vc = mesh.contravProjSphere2Plane(mesh.iA, us, vs)
        q2 = jab * gh * uc
        q3 = jab * gh * vc
        q = torch.stack([q1,q2,q3],dim=0) # nvar, npanel, npts, nx, ny
        return q

class GaussianRF(torch.nn.Module):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        super(GaussianRF, self).__init__()
        self.dim = dim
        self.device = device
        pi = math.pi

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        if dim == 1:
            k_max = size[0] // 2
            k = torch.cat((torch.arange(start=0     , end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0    , step=1, device=device)), 0)

            self.sqrt_eig = size[0]*math.sqrt(2.)*sigma*((4*(pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            k = torch.zeros_like(size)
            k[0] = size[0] // 2
            k[1:] = size[1:]
            w1 = torch.cat((torch.arange(start=0    , end=k[0]+1, step=1, device=device), \
                            torch.arange(start=-k[0], end=0     , step=1, device=device)), 0).repeat(size[1],1)
            w2 = torch.arange(start=0, end=k[1], step=1, device=device).repeat(size[0],1)

            k_x = w1.transpose(0,1)
            k_y = w2

            self.sqrt_eig = size[0]*size[1]*math.sqrt(2.)*sigma*((4*(pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        # elif dim == 3:
        #     w1 = torch.cat((torch.arange(start=0     , end=k_max, step=1, device=device), \
        #                     torch.arange(start=-k_max, end=0    , step=1, device=device)), 0).repeat(size,size,1)

        #     k_x = wavenumers.transpose(1,2)
        #     k_y = wavenumers
        #     k_z = wavenumers.transpose(0,2)

        #     self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
        #     self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size[j])

        self.size = tuple(self.size)

    def forward(self,nsample):
                
        coeff = torch.randn(nsample, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff
        
        u = torch.fft.irfftn(coeff, self.size)
        
        # Reset u in [0,1]
        u_min = u.min()
        u_max = u.max()
        u_range = u_max - u_min
        u = ( u - u_min ) / u_range

        return u
    
# class GaussianRF(object):
#     def __init__(self, dim, size, length=1.0, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", constant_eig=False, device=None):

#         self.dim = dim
#         self.device = device

#         if sigma is None:
#             sigma = tau**(0.5*(2*alpha - self.dim))

#         k_max = size//2

#         const = (4*(math.pi**2))/(length**2)

#         if dim == 1:
#             k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
#                            torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

#             self.sqrt_eig = size*math.sqrt(2.0)*sigma*((const*(k**2) + tau**2)**(-alpha/2.0))

#             if constant_eig:
#                 self.sqrt_eig[0] = size*sigma*(tau**(-alpha))
#             else:
#                 self.sqrt_eig[0] = 0.0

#         elif dim == 2:
#             wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
#                                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

#             k_x = wavenumers.transpose(0,1)
#             k_y = wavenumers

#             self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))

#             if constant_eig:
#                 self.sqrt_eig[0,0] = (size**2)*sigma*(tau**(-alpha))
#             else:
#                 self.sqrt_eig[0,0] = 0.0

#         elif dim == 3:
#             wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
#                                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

#             k_x = wavenumers.transpose(1,2)
#             k_y = wavenumers
#             k_z = wavenumers.transpose(0,2)

#             self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))

#             if constant_eig:
#                 self.sqrt_eig[0,0,0] = (size**3)*sigma*(tau**(-alpha))
#             else:
#                 self.sqrt_eig[0,0,0] = 0.0

#         self.size = []
#         for j in range(self.dim):
#             self.size.append(size)

#         self.size = tuple(self.size)

#     def sample(self, N):

#         coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
#         coeff = self.sqrt_eig*coeff

#         u = torch.fft.irfftn(coeff, self.size, norm="backward")
#         return u
    
# class GaussianRF2d(object):

#     def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

#         self.s1 = s1
#         self.s2 = s2

#         self.mean = mean

#         self.device = device
#         self.dtype = dtype

#         if sigma is None:
#             self.sigma = tau**(0.5*(2*alpha - 2.0))
#         else:
#             self.sigma = sigma

#         const1 = (4*(math.pi**2))/(L1**2)
#         const2 = (4*(math.pi**2))/(L2**2)

#         freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
#                                 torch.arange(start=-s1//2, end=0, step=1)), 0)
#         k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

#         freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)

#         k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

#         self.sqrt_eig = s1*s2*self.sigma*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
#         self.sqrt_eig[0,0] = 0.0

#     def sample(self, N, xi=None):
#         if xi is None:
#             xi  = torch.randn(N, self.s1, self.s2//2 + 1, 2, dtype=self.dtype, device=self.device)
        
#         xi[...,0] = self.sqrt_eig*xi [...,0]
#         xi[...,1] = self.sqrt_eig*xi [...,1]
        
#         u = torch.fft.irfft2(torch.view_as_complex(xi), s=(self.s1, self.s2))

#         if self.mean is not None:
#             u += self.mean
        
#         return u