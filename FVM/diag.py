import pdb
import math
import torch
import torch.nn.functional as F

class diag_class(torch.nn.Module):
    def __init__(self):
        super(diag_class, self).__init__()
    
def calc_total_mass(q,mesh):
    ids = mesh.ids
    ide = mesh.ide
    jds = mesh.jds
    jde = mesh.jde
    mass = torch.sum( q[0,:,ids:ide,jds:jde] )
    return mass

def calc_total_energy(q,mesh,case):
    ids = mesh.ids
    ide = mesh.ide
    jds = mesh.jds
    jde = mesh.jde
    nc = mesh.nx * mesh.ny * mesh.npanel

    Jgh = q[0,:,ids:ide,jds:jde] - mesh.jab[:,ids:ide,jds:jde] * case.ghs[:,ids:ide,jds:jde]
    uc = q[1,:,ids:ide,jds:jde] / Jgh
    vc = q[2,:,ids:ide,jds:jde] / Jgh
    Gmtx = mesh.G[...,ids:ide,jds:jde]
    u, v = mesh.contrav2cov(Gmtx,uc,vc)
    KE = 0.5 * Jgh * ( u * uc + v * vc ) / mesh.r**2 / nc
    PE = 0.5 * q[0,:,ids:ide,jds:jde]**2 / mesh.r**2 / nc
    energy = torch.sum( KE + PE ) * mesh.dx * mesh.dy
    return energy
    
def calc_mass(q,mesh):
    ids = mesh.ids
    ide = mesh.ide
    jds = mesh.jds
    jde = mesh.jde
    mass = q[0,:,ids:ide,jds:jde] / mesh.jab[...,ids:ide,jds:jde]
    return mass

def calc_energy(q,mesh,case):
    ids = mesh.ids
    ide = mesh.ide
    jds = mesh.jds
    jde = mesh.jde

    Jgh = q[0,:,ids:ide,jds:jde] - mesh.jab[:,ids:ide,jds:jde] * case.ghs[:,ids:ide,jds:jde]
    uc = q[1,:,ids:ide,jds:jde] / Jgh
    vc = q[2,:,ids:ide,jds:jde] / Jgh
    Gmtx = mesh.G[...,ids:ide,jds:jde]
    u, v = mesh.contrav2cov(Gmtx,uc,vc)
    KE = 0.5 * Jgh * ( u * uc + v * vc ) / mesh.r**2
    PE = 0.5 * q[0,:,ids:ide,jds:jde]**2 / mesh.r**2
    energy = ( KE + PE ) * mesh.dx * mesh.dy
    return energy

def calc_vor(q,mesh,recon,fill_ghost):
    np      = mesh.npanel
    nPOR    = mesh.nPOR
    nx      = mesh.nx
    ny      = mesh.ny
    nrx     = mesh.nrx
    nry     = mesh.nry
    nx_halo = mesh.nx_halo
    ny_halo = mesh.ny_halo
    ids     = mesh.ids
    ide     = mesh.ide
    jds     = mesh.jds
    jde     = mesh.jde
    dx      = mesh.dx
    dy      = mesh.dy
    irs     = mesh.irs
    ire     = mesh.ire
    jrs     = mesh.jrs
    jre     = mesh.jre
    pc      = mesh.pc

    qc = q.clone()
    qc = fill_ghost(qc)

    nvar, _, _, _ = qc.shape
    qc = qc.view(nvar*np,1,nx_halo,ny_halo)
    qrec = recon(qc).view(nvar,np,nPOR,nrx,nry)
    qc = qrec[...,pc,:,:]
    
    uc = qc[1,...] / qc[0,...]
    vc = qc[2,...] / qc[0,...]
    Gmtx = mesh.G[...,irs:ire,jrs:jre]
    u, v = mesh.contrav2cov(Gmtx,uc,vc)
    
    # # 2nd order
    # vx = 0.5 * ( v[...,ids+1:ide+1,jds:jde] - v[...,ids-1:ide-1,jds:jde] ) / dx
    # uy = 0.5 * ( u[...,ids:ide,jds+1:jde+1] - u[...,ids:ide,jds-1:jde-1] ) / dy
    # vor = ( vx - uy ) / ( jab * mesh.jab_stretching )
    # # 4th order
    # vx = ( v[...,ids-2:ide-2,jds:jde] - 8*v[...,ids-1:ide-1,jds:jde] + 8*v[...,ids+1:ide+1,jds:jde] - v[...,ids+2:ide+2,jds:jde] ) / dx / 12.
    # uy = ( u[...,ids-2:ide-2,jds:jde] - 8*u[...,ids:ide,jds-1:jde-1] + 8*u[...,ids:ide,jds+1:jde+1] - u[...,ids:ide,jds+2:jde+2] ) / dy / 12.
    # vor = ( vx - uy ) / ( jab * mesh.jab_stretching )

    u = u.view(np,1,nrx,nry)
    v = v.view(np,1,nrx,nry)
    vx = F.conv2d( v, recon.DxMtxC_conv ).view(np,nx,ny)
    uy = F.conv2d( u, recon.DyMtxC_conv ).view(np,nx,ny)
    jab = mesh.jab [...,ids:ide,jds:jde]
    vor_pts = ( vx - uy ) / ( jab * mesh.jab_stretching )
    vor = vor_pts # vorticity on cell center,2nd order approximation of vorticity
    # vor = torch.nn.functional.conv2d(vor_pts.view(np,1,nrx,nry), recon.conv_cell).squeeze().view(np,nx,ny)

    return vor

def calc_gh_norm_error(q,q0,mesh,case):
    ids = mesh.ids
    ide = mesh.ide
    jds = mesh.jds
    jde = mesh.jde

    jab = mesh.jabCell[:,ids:ide,jds:jde]
    ghs = case.ghs[:,ids:ide,jds:jde]

    gh  = q [0,:,ids:ide,jds:jde] / jab - ghs
    gh0 = q0[0,:,ids:ide,jds:jde] / jab - ghs

    L2 = calc_L2_error(gh,gh0,jab)

    return L2

def calc_L2_error(f,f0,area):
    res = torch.sqrt( torch.sum( ( f - f0 )**2 * area ) / torch.sum( f0**2 * area ) )
    return res

def pause(name):
    print('pause ',name)
    pdb.set_trace()

def plot_cube_field(file,lon,lat,var,vmin=None,vmax=None):
    R2D = 180. / math.pi
    point_size = 2./ 1.
    nelement = lon.nelement()

    lon_plt = lon.reshape(nelement).cpu().detach().numpy()*R2D
    lat_plt = lat.reshape(nelement).cpu().detach().numpy()*R2D
    var_plt = var.reshape(nelement).cpu().detach().numpy()

    plt.figure()
    marker = matplotlib.markers.MarkerStyle('o', fillstyle='full')
    plt.scatter(lon_plt,lat_plt,c=var_plt,s=point_size,cmap='jet',marker=marker,edgecolors='none',vmin=vmin,vmax=vmax)
    plt.xlim((0,360))
    plt.ylim((-90,90))
    plt.colorbar()
    plt.savefig(file, dpi=600)
    plt.close()
