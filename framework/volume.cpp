#include <cuda_runtime.h>

#include "gpu_volume.h"
#include "helper_cuda.h"
#include "volume.h"

#include <assert.h>

size_t Volume::voxel_size(uint8_t type)
{
    switch (type)
    {
    case VoxelType_Float:
        return sizeof(float);
    case VoxelType_Float2:
        return sizeof(float)*2;
    case VoxelType_Float3:
        return sizeof(float)*3;
    case VoxelType_Float4:
        return sizeof(float)*4;
    case VoxelType_Double:
        return sizeof(double);
    case VoxelType_Double2:
        return sizeof(double) * 2;
    case VoxelType_Double3:
        return sizeof(double) * 3;
    case VoxelType_Double4:
        return sizeof(double) * 4;
    default:
        assert(false);
    };
    return 0;
}

VolumeData::VolumeData()
{
}
VolumeData::VolumeData(size_t size)
{
    data.resize(size);
}
VolumeData::~VolumeData()
{
}

Volume::Volume() : _ptr(nullptr)
{
}
Volume::Volume(const Dims& size, uint8_t voxel_type, uint8_t* data) :
    _size(size),
    _voxel_type(voxel_type)
{
    assert(voxel_type != VoxelType_Unknown);

    size_t num_bytes = _size.width * _size.height *
        _size.depth * voxel_size(_voxel_type);

    _data = std::make_shared<VolumeData>(num_bytes);
    _ptr = _data->data.data();

    if (data)
    {
        memcpy(_ptr, data, num_bytes);
    }
}
Volume::~Volume()
{
}
GpuVolume Volume::upload()
{
    GpuVolume vol = gpu::allocate_volume(_voxel_type, _size);
    upload(vol);
    return vol;
}
void Volume::upload(const GpuVolume& gpu_volume)
{
    assert(gpu_volume.ptr != nullptr); // Requires gpu memory to be allocated
    assert(valid()); // Requires cpu memory to be allocated as well

    // We also assume both volumes have same dimensions
    assert( gpu_volume.size.width == _size.width &&
            gpu_volume.size.height == _size.height &&
            gpu_volume.size.depth == _size.depth);

    // TODO: Validate format?

    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr(_ptr, _size.width * voxel_size(_voxel_type), _size.width, _size.height);
    params.dstArray = gpu_volume.ptr;
    params.extent = { gpu_volume.size.width, gpu_volume.size.height, gpu_volume.size.depth };
    params.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&params));
}
void Volume::download(const GpuVolume& gpu_volume)
{
    assert(gpu_volume.ptr != nullptr); // Requires gpu memory to be allocated
    assert(valid()); // Requires cpu memory to be allocated as well

    // We also assume both volumes have same dimensions
    assert( gpu_volume.size.width == _size.width &&
            gpu_volume.size.height == _size.height &&
            gpu_volume.size.depth == _size.depth);

    // TODO: Validate format?

    cudaMemcpy3DParms params = { 0 };
    params.srcArray = gpu_volume.ptr;
    params.dstPtr = make_cudaPitchedPtr(_ptr, _size.width * voxel_size(_voxel_type), _size.width, _size.height);
    params.extent = { gpu_volume.size.width, gpu_volume.size.height, gpu_volume.size.depth };
    params.kind = cudaMemcpyDeviceToHost;
    checkCudaErrors(cudaMemcpy3D(&params));
}

void Volume::release()
{
    _data = nullptr;
    _ptr = nullptr;
    _size = { 0, 0, 0 };
}
Volume Volume::clone() const
{
    Volume copy(_size, _voxel_type);

    size_t num_bytes = _size.width * _size.height * 
        _size.depth * voxel_size(_voxel_type);
    
    memcpy(copy._ptr, _ptr, num_bytes);

    return copy;
}
bool Volume::valid() const
{
    return _ptr != nullptr;
}
void* Volume::ptr()
{
    assert(_ptr);
    assert(_data);
    assert(_data->data.size());
    return _ptr;
}
void const* Volume::ptr() const
{
    assert(_ptr);
    assert(_data);
    assert(_data->data.size());
    return _ptr;
}
uint8_t Volume::voxel_type() const
{
    return _voxel_type;
}
const Dims& Volume::size() const
{
    return _size;
}
Volume::Volume(const Volume& other) :
    _data(other._data),
    _ptr(other._ptr),
    _size(other._size),
    _voxel_type(other._voxel_type)
{
}
Volume& Volume::operator=(const Volume& other)
{
    _data = other._data;
    _ptr = other._ptr;
    _size = other._size;
    _voxel_type = other._voxel_type;

    return *this;
}
