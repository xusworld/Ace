
#include "saber/core/common.h"
#include "saber/core/tensor.h"
#include "saber/funcs/calibrate.h"
#include <cfloat>

namespace anakin {
namespace saber {

template <typename out_vtype, typename out_dtype, typename in_vtype, typename in_dtype>
__global__
void convert_data_type4(out_dtype* out_data, const in_dtype* in_data,
        int count, float scale) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < count) {
        in_vtype load = ((in_vtype*)in_data)[gid];
        out_vtype store;
        float load0 = static_cast<float>(load.x) * scale;
        float load1 = static_cast<float>(load.y) * scale;
        float load2 = static_cast<float>(load.z) * scale;
        float load3 = static_cast<float>(load.w) * scale;
        store.x = static_cast<out_dtype>(__float2int_rn(load0));
        store.y = static_cast<out_dtype>(__float2int_rn(load1));
        store.z = static_cast<out_dtype>(__float2int_rn(load2));
        store.w = static_cast<out_dtype>(__float2int_rn(load3));
        ((out_vtype*)out_data)[gid] = store;
    }
}

__global__
void transform_nchw_2_c4(char* out_data, const float* in_data,
        int valid_num, int valid_channel_4, int valid_height, int valid_width,
        int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
        int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
        float scale, int count, int out_channel) {

    int load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int write_w = (gid) % valid_width;
    int write_h = (gid / (out_h_stride)) % valid_height;
    int write_c = (gid / (out_c_stride)) % valid_channel_4;
    int write_n = (gid / (out_n_stride)) % valid_num;

    int in_offset = write_n * in_n_stride
                    + write_c * in_c_stride * 4
                    + write_h * in_h_stride
                    + write_w * in_w_stride;

    int out_offset = write_n * out_n_stride
                     + write_c * out_c_stride
                     + write_h * out_h_stride
                     + write_w;

    if (gid < count) {
        bool p0, p1, p2, p3;
        p0 = (4 * write_c) < out_channel;
        p1 = (4 * write_c) + 1 < out_channel;
        p2 = (4 * write_c) + 2 < out_channel;
        p3 = (4 * write_c) + 3 < out_channel;
        float r0;
        char4 write;
        if (p0) r0 = __ldg(&in_data[in_offset]);
        else r0 = 0;
        load0 = __float2int_rn(r0 * scale);
        write.x = static_cast<char>(load0);

        in_offset += in_c_stride;
        if (p1) r0 = __ldg(&in_data[in_offset]);
        else r0 = 0;
        load1 = __float2int_rn(r0 * scale);
        write.y = static_cast<char>(load1);

        in_offset += in_c_stride;
        if (p2) r0 = __ldg(&in_data[in_offset]);
        else r0 = 0;
        load2 = __float2int_rn(r0 * scale);
        write.z = static_cast<char>(load2);

        in_offset += in_c_stride;
        if (p3) r0 = __ldg(&in_data[in_offset]);
        else r0 = 0;
        load3 = __float2int_rn(r0 * scale);
        write.w = static_cast<char>(load3);

        ((char4*)out_data)[out_offset] = write;
    }
}

__global__ void transform_nchw_2_nchw(float * out_data,
        const float* in_data, const int count,
        int in_n, int in_c, int in_h, int in_w,
        int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
        int out_n, int out_c, int out_h, int out_w,
        int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
        const float *scale, const float input_scale) {

    CUDA_KERNEL_LOOP(tid, count){
        int read_w =  tid % in_w;
        int read_h = (tid / (in_w)) % in_h;
        int read_c = (tid / (in_h * in_w)) % in_c;
        int read_n = (tid / (in_c * in_h * in_w)) % in_n;

        int write_w =  tid % out_w;
        int write_h = (tid / (out_w)) % out_h;
        int write_c = (tid / (out_h * out_w)) % out_c;
        int write_n = (tid / (out_c * out_h * out_w)) % out_n;

        int in_idx = read_n * in_n_stride
                     + read_c * in_c_stride
                     + read_h * in_h_stride
                     + read_w * in_w_stride;

        int out_idx = write_n * out_n_stride
                      + write_c * out_c_stride
                      + write_h * out_h_stride
                      + write_w * out_w_stride;

        float in_var = in_data[in_idx];
        float in_scale = scale[read_c];
        out_data[out_idx] = in_var * in_scale * input_scale;
    }
}

__global__
void int8nchwc4_fp32nchw(float* out_data, const char* in_data,
        int valid_num, int valid_channel_4, int valid_height, int valid_width,
        int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
        int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
        const float* scale, int count) {

    float load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int read_w = (gid) % valid_width;
    int read_h = (gid / (in_h_stride)) % valid_height;
    int read_c = (gid / (in_c_stride)) % valid_channel_4;
    int read_n = (gid / (in_n_stride)) % valid_num;
    int scale_index = read_c << 2;

    int in_offset = read_n * in_n_stride
                    + read_c * in_c_stride
                    + read_h * in_h_stride
                    + read_w;

    int out_offset = read_n * out_n_stride
                     + read_c * (out_c_stride << 2)
                     + read_h * out_h_stride
                     + read_w * out_w_stride;

    if (gid < count) {

        char4 readin = __ldg(&((const char4*)in_data)[in_offset]);

        load0 = static_cast<float>(readin.x);
        load1 = static_cast<float>(readin.y);
        load2 = static_cast<float>(readin.z);
        load3 = static_cast<float>(readin.w);

        out_data[out_offset] = load0 * scale[scale_index]; out_offset += out_c_stride;
        out_data[out_offset] = load1 * scale[scale_index + 1]; out_offset += out_c_stride;
        out_data[out_offset] = load2 * scale[scale_index + 2]; out_offset += out_c_stride;
        out_data[out_offset] = load3 * scale[scale_index + 3];
    }
}

template <typename dtype>
__global__
void nchwc4_2_nchw(dtype* out_data, const char* in_data,
        int valid_num, int valid_channel_4, int valid_height, int valid_width,
        int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
        int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride, int count) {

    dtype load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int read_w = (gid) % valid_width;
    int read_h = (gid / (in_h_stride)) % valid_height;
    int read_c = (gid / (in_c_stride)) % valid_channel_4;
    int read_n = (gid / (in_n_stride)) % valid_num;

    int in_offset = read_n * in_n_stride
                    + read_c * in_c_stride
                    + read_h * in_h_stride
                    + read_w;

    int out_offset = read_n * out_n_stride
                     + read_c * (out_c_stride << 2)
                     + read_h * out_h_stride
                     + read_w * out_w_stride;

    if (gid < count) {

        char4 readin = __ldg(&((const char4*)in_data)[in_offset]);
        load0 = static_cast<dtype>(readin.x);
        load1 = static_cast<dtype>(readin.y);
        load2 = static_cast<dtype>(readin.z);
        load3 = static_cast<dtype>(readin.w);

        out_data[out_offset] = load0; out_offset += out_c_stride;
        out_data[out_offset] = load1; out_offset += out_c_stride;
        out_data[out_offset] = load2; out_offset += out_c_stride;
        out_data[out_offset] = load3;
    }
}

__global__
void int8nchwc4_fp32nchw_s(float* out_data, const char* in_data,
                         int valid_num, int valid_channel_4, int valid_height, int valid_width,
                         int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                         int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                         const float scale, int count) {

    float load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int read_w = (gid) % valid_width;
    int read_h = (gid / (in_h_stride)) % valid_height;
    int read_c = (gid / (in_c_stride)) % valid_channel_4;
    int read_n = (gid / (in_n_stride)) % valid_num;

    int in_offset = read_n * in_n_stride
                    + read_c * in_c_stride
                    + read_h * in_h_stride
                    + read_w;

    int out_offset = read_n * out_n_stride
                     + read_c * (out_c_stride << 2)
                     + read_h * out_h_stride
                     + read_w * out_w_stride;

    if (gid < count) {

        char4 readin = __ldg(&((const char4*)in_data)[in_offset]);

        load0 = static_cast<float>(readin.x);
        load1 = static_cast<float>(readin.y);
        load2 = static_cast<float>(readin.z);
        load3 = static_cast<float>(readin.w);

        out_data[out_offset] = load0 * scale; out_offset += out_c_stride;
        out_data[out_offset] = load1 * scale; out_offset += out_c_stride;
        out_data[out_offset] = load2 * scale; out_offset += out_c_stride;
        out_data[out_offset] = load3 * scale;
    }
}

#define JUDGESIGN(x) (((x) >= 0) ? +1 : -1)
__global__
void calibrate_float2char_col(signed char* dst, const float* src,
        float * scale, int height, int width) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    float col_max = 0.0f;
    const float *data = src + gid;
    for(int idx = 0; idx < height; ++idx){
        if (gid < width) {
            float temp = fabsf(data[idx * width]);
            col_max = (col_max >= temp)? col_max : temp;
        }
    }
    signed char* target = dst + gid;
    float col_scale = (float)((1 << 7) - 1) / col_max;
    for(int idx = 0; idx < height; ++idx) {
        if(gid < width) {
            float temp = data[idx * width];
            if(temp >= col_max - FLT_EPSILON) {
                target[idx * width] = (signed char)((1 << 7) - 1);
            } else if(temp <= -col_max + FLT_EPSILON) {
                target[idx * width] = (signed char)(-(1 << 7));
            } else {
                target[idx * width] = (signed char)(temp * col_scale + JUDGESIGN(temp) * 0.5);
            }
        }
    }
    scale[gid] = 1.f / col_scale;
}

__global__
void calibrate_float2char_row(signed char* dst, const float* src,
        float * scale, int height, int width) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    float row_max = 0.0f;
    const float * data = src + width * gid;
    for(int idx = 0; idx < width; ++idx) {
        if(gid < height){
            float temp = fabsf(data[idx]);
            row_max = (row_max >= temp) ? row_max : temp;
        }
    }
    signed char * target = dst + width * gid;
    float row_scale = (float)((1 << 7) - 1) / row_max;
    for(int idx = 0; idx < width; ++idx) {
        if(gid < height) {
            float temp = data[idx];
            if(temp >= row_max - FLT_EPSILON) {
                target[idx] = (signed char)((1 << 7) - 1);
            } else if(temp <= -row_max + FLT_EPSILON) {
                target[idx] = (signed char)(-(1 << 7));
            } else {
                target[idx] = (signed char)(temp * row_scale + JUDGESIGN(temp) * 0.5);
            }
        }
    }
    scale[gid] = 1.f / row_scale;
}

__global__ void calibrate_fix2float(float * dst,
                                    const float* sA, const float* sB,
                                    float alpha, float beta, int height,
                                    int width, int threads) {
    int ri = blockIdx.x;
    int tid = threadIdx.x;
    int loop = (width / threads) + ((width % threads == 0) ? 0 : 1);

    float rscale = (sA[ri] == 0.0f) ? 1.0f : sA[ri];
    float * data = dst + width * ri;
    int idx = 0;
    for (int i = 0; i < loop; ++i) {
        if(idx + tid < width){
            float temp = data[idx + tid];
            float cscale = (sB[idx + tid] == 0.0f) ? 255.0f : sB[idx + tid];
            data[idx + tid] = beta  * temp + alpha * temp * rscale * cscale;
        }
        idx += threads;
    }
}

template <>
SaberStatus conv_data_calibrate<NV, Layout_NCHW, char, Layout_NCHW, float>(
        Tensor<NV> &out_tensor, const Tensor<NV> &in_tensor, const float in_scale,
        const float* weight_scale, Context<NV> ctx) {
    if (in_tensor.get_dtype() != AK_FLOAT) {
        LOG(FATAL) << "input tensor dtype error!";
    }
    if (out_tensor.get_dtype() != AK_INT8) {
        LOG(FATAL) << "output tensor dtype error!";
    }
    if (in_tensor.get_layout() != out_tensor.get_layout()) {
        LOG(FATAL) << "convert layout is not same!";
    }
    if (in_tensor.valid_size() != out_tensor.valid_size()) {
        LOG(FATAL) << "convert size is not same!";
    }
    char* out_data = (char*)out_tensor.mutable_data();
    const float* in_data = (const float*)in_tensor.data();
    float scale = 1 / (in_tensor.get_scale()[0]);
    int count = in_tensor.valid_size() / 4; // need to check if is multiple of 4
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    convert_data_type4<char4, char, float4, float>
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>> (
            out_data, in_data, count, scale);
    return SaberSuccess;
}

template <>
SaberStatus conv_data_calibrate<NV, Layout_NCHW, float, Layout_NCHW, char>(
        Tensor<NV> &out_tensor, const Tensor<NV> &in_tensor, const float in_scale,
        const float* weight_scale, Context<NV> ctx) {
    if (out_tensor.get_dtype() != AK_FLOAT) {
        LOG(FATAL) << "output tensor dtype error!";
    }
    if (in_tensor.get_dtype() != AK_INT8) {
        LOG(FATAL) << "input tensor dtype error!";
    }
    if (in_tensor.get_layout() != out_tensor.get_layout()) {
        LOG(FATAL) << "convert layout is not same!";
    }
    if (in_tensor.valid_size() != out_tensor.valid_size()) {
        LOG(FATAL) << "convert size is not same!";
    }
    float* out_data = (float*)out_tensor.mutable_data();
    const char* in_data = (const char*)in_tensor.data();
    float scale = in_tensor.get_scale()[0];
    int count = in_tensor.valid_size() / 4; // need to check if is multiple of 4
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    convert_data_type4<float4, float, char4, char>
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>> (
            out_data, in_data, count, scale);

    return SaberSuccess;
}

template <>
SaberStatus conv_data_calibrate<NV, Layout_NCHW, float, Layout_NCHW_C4, char>(
        Tensor<NV> &out_tensor, const Tensor<NV> &in_tensor, const float in_scale,
        const float* weight_scale, Context<NV> ctx) {
    Shape out_stride = out_tensor.get_stride();
    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();
    int count = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];

    const char * in_data = (const char*)in_tensor.data();
    float * out_data = (float*)out_tensor.mutable_data();

    cudaStream_t cuda_stream = ctx.get_compute_stream();
    int8nchwc4_fp32nchw<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, in_data,
            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
            in_shape[1] * in_shape[2] * in_shape[3],
            in_shape[2] * in_shape[3],
            in_shape[3], 1,
            out_stride[0], out_stride[1], out_stride[2], out_stride[3],
            weight_scale, count);

    return SaberSuccess;
}

template <>
SaberStatus conv_data_calibrate<NV, Layout_NCHW_C4, char, Layout_NCHW, float>(
        Tensor<NV> &out_tensor, const Tensor<NV> &in_tensor, const float in_scale,
        const float* weight_scale, Context<NV> ctx) {
    const float * in_data = (const float*)in_tensor.data();
    char * out_data = (char*)out_tensor.mutable_data();

    Shape in_stride = in_tensor.get_stride();

    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();

    int out_num = out_shape.num();
    int out_channel = out_shape.channel();
    int out_height = out_shape.height();
    int out_width = out_shape.width();
    int out_channel_4 = out_channel >> 2;
    bool multipler_4 = (out_channel & 0x3) != 0;
    out_channel_4 += multipler_4 ? 1 : 0;
    int count = out_num * out_channel_4 * out_height * out_width;
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    transform_nchw_2_c4<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS,
            0, cuda_stream>>>(out_data, in_data,
            out_num, out_channel_4, out_height, out_width,
            in_stride[0], in_stride[1], in_stride[2], in_stride[3],
            out_channel_4 * out_height * out_width,
            out_height * out_width, out_width, 1,
            (1.f / in_scale), count, out_channel);

    return SaberSuccess;
}

// This template is for calibrate!!!!
template <>
SaberStatus conv_data_calibrate<NV, Layout_NCHW, float, Layout_NCHW, float>(
        Tensor<NV> &out_tensor, const Tensor<NV> &in_tensor, const float in_scale,
        const float* weight_scale, Context<NV> ctx) {
    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();

    Shape stride_in = in_tensor.get_stride();
    Shape stride_out = out_tensor.get_stride();

    const float *in_data = (const float*)in_tensor.data();
    float *out_data = (float*)out_tensor.mutable_data();

    const int count = in_tensor.valid_size();
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    transform_nchw_2_nchw
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
            out_data, in_data, count,
                    in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                    out_shape[0], out_shape[1], out_shape[2], out_shape[3],
                    stride_out[0], stride_out[1], stride_out[2], stride_out[3],
                    weight_scale, in_scale);

    return SaberSuccess;
}

template <>
SaberStatus flatten_calibrate<NV, float, char>(
        Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor,
        Context<NV> &ctx) {

    if (out_tensor.get_dtype() != AK_FLOAT) {
        LOG(FATAL) << "output tensor dtype error!";
    }
    if (in_tensor.get_dtype() != AK_INT8) {
        LOG(FATAL) << "input tensor dtype error!";
    }
    if (in_tensor.get_layout() != out_tensor.get_layout()) {
        LOG(FATAL) << "convert layout is not same!";
    }
    if (in_tensor.valid_size() != out_tensor.valid_size()) {
        LOG(FATAL) << "convert size is not same!";
    }
    float* out_data = (float*)out_tensor.mutable_data();
    const char* in_data = (const char*)in_tensor.data();
    float scale = in_tensor.get_scale()[0];
    int count = in_tensor.valid_size() / 4; // need to check if is multiple of 4
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    convert_data_type4<float4, float, char4, char>
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>> (
            out_data, in_data, count, scale);

    return SaberSuccess;
}

template <>
SaberStatus flatten_calibrate<NV, char, float>(
        Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor,
        Context<NV> &ctx) {
    if (in_tensor.get_dtype() != AK_FLOAT) {
        LOG(FATAL) << "input tensor dtype error!";
    }
    if (out_tensor.get_dtype() != AK_INT8) {
        LOG(FATAL) << "output tensor dtype error!";
    }
    if (in_tensor.get_layout() != out_tensor.get_layout()) {
        LOG(FATAL) << "convert layout is not same!";
    }
    if (in_tensor.valid_size() != out_tensor.valid_size()) {
        LOG(FATAL) << "convert size is not same!";
    }
    char* out_data = (char*)out_tensor.mutable_data();
    const float* in_data = (const float*)in_tensor.data();
    float scale = 1 / (in_tensor.get_scale()[0]);
    int count = in_tensor.valid_size() / 4; // need to check if is multiple of 4
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    convert_data_type4<char4, char, float4, float>
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>> (
            out_data, in_data, count, scale);
    return SaberSuccess;
}

template<>
SaberStatus conv_calibrate_fp32_int8_c4<NV>(Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor, const float in_scale, Context<NV> ctx) {

    const float * in_data = (const float*)in_tensor.data();
    char * out_data = (char*)out_tensor.mutable_data();

    Shape in_stride = in_tensor.get_stride();

    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();

    int out_num = out_shape.num();
    int out_channel = in_shape.channel();
    int out_height = out_shape.height();
    int out_width = out_shape.width();
    int out_channel_4 = out_channel >> 2;
    bool multipler_4 = (out_channel & 0x3) != 0;
    out_channel_4 += multipler_4 ? 1 : 0;
    int count = out_num * out_channel_4 * out_height * out_width;
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    transform_nchw_2_c4<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS,
        0, cuda_stream>>>(out_data, in_data,
            out_num, out_channel_4, out_height, out_width,
            in_stride[0], in_stride[1], in_stride[2], in_stride[3],
            out_channel_4 * out_height * out_width,
            out_height * out_width, out_width, 1,
            (1.f / in_scale), count, out_channel);

    return SaberSuccess;
}

template<>
SaberStatus conv_calibrate_int32_fp32<NV>(
        Tensor<NV> &out_tensor, const Tensor<NV> &in_tensor,
        const float in_scale, const float* weight_scale, Context<NV> ctx) {

    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();

    Shape stride_in = in_tensor.get_stride();
    Shape stride_out = out_tensor.get_stride();

    const float *in_data = (const float*)in_tensor.data();
    float *out_data = (float*)out_tensor.mutable_data();

    const int count = in_tensor.valid_size();
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    transform_nchw_2_nchw
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
            out_data, in_data, count,
                    in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                    out_shape[0], out_shape[1], out_shape[2], out_shape[3],
                    stride_out[0], stride_out[1], stride_out[2], stride_out[3],
                    weight_scale, in_scale);

    return SaberSuccess;
}

template<>
SaberStatus conv_calibrate_int8_c4_fp32<NV>(
        Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor,
        const float* weight_scale,
        Context<NV> ctx) {

    Shape out_stride = out_tensor.get_stride();
    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();
    int count = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] / 4;

    const char * in_data = (const char*)in_tensor.data();
    float * out_data = (float*)out_tensor.mutable_data();

    cudaStream_t cuda_stream = ctx.get_compute_stream();
    int8nchwc4_fp32nchw<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, in_data,
            in_shape[0], in_shape[1] / 4, in_shape[2], in_shape[3],
            in_shape[1] * in_shape[2] * in_shape[3],
            in_shape[2] * in_shape[3],
            in_shape[3], 1,
            out_stride[0], out_stride[1], out_stride[2], out_stride[3],
            weight_scale, count);

    return SaberSuccess;
}

template <>
SaberStatus layout_trans_nchwc4_2_nchw<NV>(
        Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor,
        float scale,
        Context<NV> ctx) {

    Shape out_stride = out_tensor.get_stride();
    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();
    int count = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] / 4;

    cudaStream_t cuda_stream = ctx.get_compute_stream();
    if (in_tensor.get_dtype() == AK_FLOAT) {
        flatten_calibrate<NV, char, float>(out_tensor, in_tensor, ctx);
    } else if (in_tensor.get_dtype() == AK_INT8) {
        const char * in_data = (const char*)in_tensor.data();
        char * out_data = (char*)out_tensor.mutable_data();
        nchwc4_2_nchw<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, in_data,
                in_shape[0], in_shape[1] / 4, in_shape[2], in_shape[3],
                in_shape[1] * in_shape[2] * in_shape[3] / 4,
                in_shape[2] * in_shape[3], in_shape[3], 1,
                out_stride[0], out_stride[1], out_stride[2], out_stride[3], count);
    } else {
        LOG(FATAL) << "tensor dtype is wrong!!!";
    }

    return SaberSuccess;
}

template<>
SaberStatus calibrate_int8_c4_fp32<NV>(
        Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor,
        const float out_scale,
        Context<NV> ctx) {

    Shape out_stride = out_tensor.get_stride();
    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();
    int count = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] / 4;
    const char * in_data = (const char*)in_tensor.data();
    float * out_data = (float*)out_tensor.mutable_data();

    cudaStream_t cuda_stream = ctx.get_compute_stream();
    int8nchwc4_fp32nchw_s<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, in_data,
            in_shape[0], in_shape[1] / 4, in_shape[2], in_shape[3],
            in_shape[1] * in_shape[2] * in_shape[3] / 4,
            in_shape[2] * in_shape[3],
            in_shape[3], 1,
            out_stride[0], out_stride[1], out_stride[2], out_stride[3],
            out_scale, count);

    return SaberSuccess;
}

template <>
void float2char<NV>(bool col_direct, signed char* dst, const float* src,
                    float *scale, int height, int width, Context<NV> ctx) {
    int threads = 32;
    cudaStream_t cuda_stream = ctx.get_compute_stream();
    if (col_direct) {
        calibrate_float2char_col <<< (width / threads) + (((width % threads) == 0) ? 0 : 1), threads, 0,
                cuda_stream >>> (
                        dst, src, scale, height, width);
    } else {
        calibrate_float2char_row<<<(height / threads) + (((height % threads)==0) ? 0 : 1), threads, 0, cuda_stream>>>(
                dst, src, scale, height, width);
    }
}
template <>
void fix2float<NV>(float * dst,
               const float *sA, const float *sB,
               const float alpha, const float beta, int height, int width, Context<NV> ctx) {
    int threads = 256;
    cudaStream_t cuda_stream = ctx.get_compute_stream();
    calibrate_fix2float<<<height, threads, 0, cuda_stream>>>(dst, sA, sB, alpha, beta,
            height, width, threads);
}


}
}