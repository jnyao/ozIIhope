import torch
import math

class interp_latlon_to_cube_class(torch.nn.Module):
    def __init__(self,lat_src,lon_src,dlat_src,dlon_src,lat_tgt,lon_tgt):
        super(interp_latlon_to_cube_class, self).__init__()
        R2D = 180. / math.pi

        lon_min_src = torch.min(lon_src)
        lon_max_src = torch.max(lon_src)
        lat_min_src = torch.min(lat_src)
        lat_max_src = torch.max(lat_src)

        self.nlon = lon_src.size()[0]
        self.nlat = lat_src.size()[0]

        self.idx = torch.floor( ( lon_tgt - lon_min_src ) / dlon_src ).type(torch.int32)
        self.jdx = torch.floor( ( lat_tgt - lat_min_src ) / dlat_src ).type(torch.int32)

        self.idxp1 = torch.where( self.idx+1<=self.nlon-1, self.idx+1, 0 )
        self.jdxp1 = torch.where( self.jdx+1<=self.nlat-1, self.jdx+1, 0 )

        cx = ( lon_tgt - lon_min_src ) / dlon_src - self.idx
        cy = ( lat_tgt - lat_min_src ) / dlat_src - self.jdx

        c1 = ( 1. - cx ) * ( 1. - cy )
        c2 = cx * ( 1. - cy )
        c3 = cx * cy
        c4 = ( 1.-  cx ) * cy

        self.c = torch.stack( [c1,c2,c3,c4], dim=0 )
        
    def forward(self,var):
        q = self.c[0,...] * var[...,self.jdx  ,self.idx  ] \
          + self.c[1,...] * var[...,self.jdx  ,self.idxp1] \
          + self.c[2,...] * var[...,self.jdxp1,self.idxp1] \
          + self.c[3,...] * var[...,self.jdxp1,self.idx  ]
        return q