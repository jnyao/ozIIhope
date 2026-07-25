#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cudnn/Handle.h>
#include <torch/script.h>

torch::Tensor ozcudnn_launcher(torch::Tensor input1,  torch::Tensor input2,torch::Tensor input3,torch::Tensor d_x,torch::Tensor d_y, torch::Tensor d_conv1);

torch::Tensor accumulate_ozcudnn_launcher(torch::Tensor input1,  torch::Tensor input2,torch::Tensor input3,torch::Tensor d_x,torch::Tensor d_y, torch::Tensor d_conv1,torch::Tensor bw);
