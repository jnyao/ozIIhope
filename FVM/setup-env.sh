export OZAKI_OP_DIR=/path/to/libozaki_op.so
export NUMSPLIT=7
export BITS_PER_SLICE=8

# Setting the memory layout strategy
# OZ_NORMAL = 0, for in-channel-only utilization
# OZ_BSHALF = 1, for batch-size dimension utilization
export OZ_MODE=1

# The switch parameter to control the precision used in the convolution computation of HOPE
# PREC_MODE = 0, for turning off Ozaki scheme in convolution
# PREC_MODE = 1, for turning  on Ozaki scheme in convolution
export PREC_MODE=1
