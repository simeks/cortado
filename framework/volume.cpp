#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "volume.h"

#include <assert.h>

VolumeBase::VolumeBase(size_t elem_size) : 
    _element_size(elem_size)
{
    _gpu_vol = { 0 };
}
VolumeBase::~VolumeBase()
{
    release_gpu_volume();
}

void VolumeBase::allocate_volume(const Dims& size)
{
    assert(size.width != 0 && size.width != 0 && size.depth != 0);

    _size = size;
    _data.resize(_size.width * _size.height * _size.depth * _element_size);
}
void VolumeBase::allocate_gpu_volume(
    const struct cudaChannelFormatDesc& desc,
    const cudaExtent& size)
{
    assert(_gpu_vol.ptr == nullptr); // Memory should not already be allocated

    checkCudaErrors(cudaMalloc3DArray(&_gpu_vol.ptr, &desc, size, 0));
    _gpu_vol.size = { size.width, size.height, size.depth };
    _gpu_vol.format_desc = desc;
}
void VolumeBase::release_gpu_volume()
{
    if (_gpu_vol.ptr == nullptr) // not allocated
        return;

    checkCudaErrors(cudaFreeArray(_gpu_vol.ptr));
    _gpu_vol.ptr = nullptr;
    _gpu_vol.size = { 0, 0, 0 };
}
void VolumeBase::copy_to_gpu()
{
    assert(_gpu_vol.ptr != nullptr); // Requires gpu memory to be allocated

    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr(_data.data(), _size.width * _element_size, _size.width, _size.height);
    params.dstArray = _gpu_vol.ptr;
    params.extent = { _gpu_vol.size.width, _gpu_vol.size.height, _gpu_vol.size.depth };
    params.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&params));
}
void VolumeBase::copy_from_gpu()
{
    assert(_gpu_vol.ptr != nullptr); // Requires gpu memory to be allocated

    cudaMemcpy3DParms params = { 0 };
    params.srcArray = _gpu_vol.ptr;
    params.dstPtr = make_cudaPitchedPtr(_data.data(), _size.width * _element_size, _size.width, _size.height);
    params.extent = { _gpu_vol.size.width, _gpu_vol.size.height, _gpu_vol.size.depth };
    params.kind = cudaMemcpyDeviceToHost;
    checkCudaErrors(cudaMemcpy3D(&params));
}
void* VolumeBase::data()
{
    return _data.data();
}
void const* VolumeBase::data() const
{
    return _data.data();
}
