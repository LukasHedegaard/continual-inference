#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <iterator>

using at::IntArrayRef;
using at::Tensor;
using torch::indexing::None;
using torch::indexing::Slice;

// Given an input tensor and an expected number of spatial dimensions, checks that the
// input is a valid shape and returns the batched form of the input.
//
// Args:
//     input (Tensor): Input tensor
//     num_spatial_dims (int): Number of spatial dimensions expected for the input
//     func_name (string): Function name to produce a nice error message for invalid input
//
// Returns a std::tuple containing:
//     batched_input (Tensor): Input with a batch dimension
//     is_batched (bool): Indicates whether the original input was already batched
static std::tuple<Tensor, bool> batchify(
    const Tensor &input,
    const int64_t num_spatial_dims,
    const std::string &func_name)
{
    const auto dim_count_no_batch = num_spatial_dims + 1;
    const auto dim_count_batch = dim_count_no_batch + 1;
    const auto is_batched = (input.dim() == dim_count_batch);
    TORCH_CHECK(input.dim() == dim_count_no_batch || is_batched,
                "Expected ", dim_count_no_batch, "D (unbatched) or ", dim_count_batch,
                "D (batched) input to ", func_name, ", but got input of size: ", input.sizes());
    return std::make_tuple(is_batched ? input : input.unsqueeze(0), is_batched);
}

static std::tuple<Tensor, int64_t, int64_t>
init_state(
    const Tensor &step_propotype,
    const int64_t temporal_size,
    const int64_t temporal_stride,
    const int64_t temporal_padding,
    const int64_t dims)
{
    // Determine buffer size
    std::vector<int64_t> buf_shape_{temporal_size - 1};
    buf_shape_.reserve(1 + dims);
    for (int64_t i : step_propotype.sizes())
        buf_shape_.push_back(i);

    // Allocate state buffer
    Tensor buffer = torch::zeros(buf_shape_,
                                 torch::TensorOptions()
                                     .dtype(step_propotype.dtype())
                                     .device(step_propotype.device().type(), step_propotype.device().index())
                                     .requires_grad(false));

    int64_t state_index = 0;
    int64_t stride_index = temporal_stride - temporal_size + temporal_padding;
    return std::make_tuple(buffer, state_index, stride_index);
}

// Returns result, buffer, state_index, stride_index
std::tuple<c10::optional<Tensor>, std::tuple<Tensor, int64_t, int64_t>>
coconv_forward_step(
    const Tensor &input_,
    const Tensor &weight,
    const c10::optional<Tensor> &bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef step_padding,
    IntArrayRef dilation,
    const int64_t groups,
    // c10::optional<std::tuple<Tensor &, int64_t, int64_t>> &prev_state_opt, # NB: tensor device isn't transferred properly if packed in a tuple
    c10::optional<Tensor> &buffer_opt,
    c10::optional<int64_t> &state_index_opt,
    c10::optional<int64_t> &stride_index_opt)
{
    const int64_t temporal_size = weight.sizes()[2];
    const int64_t dims = weight.sizes().size() - 2;

    Tensor input = input_.unsqueeze(2); // e.g. B, C -> B, C, 1
    bool is_batched;
    std::tie(input, is_batched) = batchify(input, /*num_spatial_dims=*/dims, "co_conv");

    // Perform convolutions for current input
    Tensor x_cur;
    switch (dims)
    {
    case 1:
        x_cur = at::conv1d(input, weight, torch::Tensor(), stride, step_padding, dilation, groups);
        break;
    case 2:
        x_cur = at::conv2d(input, weight, torch::Tensor(), stride, step_padding, dilation, groups);
        break;
    case 3:
        x_cur = at::conv3d(input, weight, torch::Tensor(), stride, step_padding, dilation, groups);
        break;
    }

    auto x_out = x_cur.index({Slice(), Slice(), 0});
    auto x_rest = x_cur.index({Slice(), Slice(), Slice(1, None)});

    // Check if prev_state is non-empty
    Tensor buffer;
    int64_t state_index, stride_index;
    // std::tie(buffer, state_index, stride_index) = prev_state_opt.has_value()
    //                                                   ? prev_state_opt.value()
    //                                                   : init_state(x_rest, temporal_size, stride[0], padding[0], dims);
    if (buffer_opt.has_value())
    {
        buffer = buffer_opt.value();
        state_index = state_index_opt.value();
        stride_index = stride_index_opt.value();
    }
    else
        std::tie(buffer, state_index, stride_index) = init_state(x_rest, temporal_size, stride[0], padding[0], dims);

    int64_t buffer_len = buffer.sizes().front();

    // Compute output
    c10::optional<Tensor> result;
    if (stride_index == stride[0] - 1)
    {
        result = x_out + torch::sum(
                             buffer.index({torch::remainder(torch::arange(buffer_len) + state_index, buffer_len),
                                           Slice(),
                                           Slice(),
                                           torch::arange(buffer_len - 1, -1, -1)}),
                             0);
        if (bias_opt.has_value())
            result = result.value() + bias_opt.value();
    }

    // Update states
    // if (weight.sizes().front() > 1)
    // {
    buffer[state_index] = x_rest;
    state_index = (state_index + 1) % buffer_len;
    // }
    stride_index = stride_index + 1;
    if (stride_index > 0)
        stride_index = stride_index % stride[0];

    return std::make_tuple(result, std::make_tuple(buffer, state_index, stride_index));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward_step", &coconv_forward_step, "CoConv forward_step");
}