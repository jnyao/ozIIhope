#include <torch/torch.h>
#include <torch/script.h>
#include "pyt_ozaki_kernel.hh"


static torch::Tensor custom_ozcudnn(torch::Tensor input1,torch::Tensor input2,torch::Tensor input3, torch::Tensor d_x,torch::Tensor d_y,torch::Tensor d_conv1) {
    return ozcudnn_launcher(input1, input2, input3, d_x, d_y, d_conv1); 
}


static torch::Tensor custom_accumulate_ozcudnn(torch::Tensor input1,torch::Tensor input2,torch::Tensor input3, torch::Tensor d_x,torch::Tensor d_y,torch::Tensor d_conv1,torch::Tensor bw) {
    return accumulate_ozcudnn_launcher(input1, input2, input3, d_x, d_y, d_conv1,bw); 
}

TORCH_LIBRARY (my_ops, m){
    m.def("custom_ozcudnn", &custom_ozcudnn);
    m.def("custom_accumulate_ozcudnn", &custom_accumulate_ozcudnn);
}

