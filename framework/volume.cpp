#include <cuda_runtime.h>

#include "gpu_volume.h"
#include "helper_cuda.h"
#include "volume.h"

#include <assert.h>
#include <iostream>


namespace
{
    typedef void(*ConverterFn)(void*, void*, size_t num);

    template<typename TSrc, typename TDest>
    void convert_voxels(void* src, void* dest, size_t num)
    {
        // TODO:
        // Very rough conversion between voxel formats,
        // works fine for double <-> float, should suffice for now

        for (size_t i = 0; i < num; ++i)
        {
            ((TDest*)dest)[i] = TDest(((TSrc*)src)[i]);
        }
    }
}

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
    case VoxelType_UChar:
        return sizeof(uchar1);
    case VoxelType_UChar2:
        return sizeof(uchar1) * 2;
    case VoxelType_UChar3:
        return sizeof(uchar1) * 3;
    case VoxelType_UChar4:
        return sizeof(uchar1) * 4;
    default:
        assert(false);
    };
    return 0;
}
int Volume::voxel_num_components(uint8_t type)
{
    switch (type)
    {
    case VoxelType_Float:
    case VoxelType_Double:
    case VoxelType_UChar:
        return 1;
    case VoxelType_Float2:
    case VoxelType_Double2:
    case VoxelType_UChar2:
        return 2;
    case VoxelType_Float3:
    case VoxelType_Double3:
    case VoxelType_UChar3:
        return 3;
    case VoxelType_Float4:
    case VoxelType_Double4:
    case VoxelType_UChar4:
        return 4;
    default:
        assert(false);
    };
    return 0;
}
uint8_t Volume::voxel_base_type(uint8_t type)
{
    switch (type)
    {
    case VoxelType_Float:
    case VoxelType_Float2:
    case VoxelType_Float3:
    case VoxelType_Float4:
        return VoxelType_Float;
    case VoxelType_Double:
    case VoxelType_Double2:
    case VoxelType_Double3:
    case VoxelType_Double4:
        return VoxelType_Double;
    case VoxelType_UChar:
    case VoxelType_UChar2:
    case VoxelType_UChar3:
    case VoxelType_UChar4:
        return VoxelType_UChar;
    default:
        assert(false);
    };
    return VoxelType_Unknown;
}

VolumeData::VolumeData() : data(nullptr), size(0)
{
}
VolumeData::VolumeData(size_t size) : size(size)
{
    data = new uint8_t[size];
}
VolumeData::~VolumeData()
{
    if (data)
        delete[] data;
}

Volume::Volume() : _ptr(nullptr)
{
}
Volume::Volume(const Dims& size, uint8_t voxel_type, uint8_t* data) :
    _size(size),
    _voxel_type(voxel_type)
{
    allocate(size, voxel_type);
    if (data)
    {
        size_t num_bytes = _size.width * _size.height *
            _size.depth * voxel_size(_voxel_type);

        memcpy(_ptr, data, num_bytes);
    }
}
Volume::Volume(const GpuVolume& gpu_volume)
{
    allocate(gpu_volume.size, gpu::voxel_type(gpu_volume));
    download(gpu_volume);
}
Volume::~Volume()
{
}
GpuVolume Volume::upload() const
{
    GpuVolume vol = gpu::allocate_volume(_voxel_type, _size);
    upload(vol);
    return vol;
}
void Volume::upload(const GpuVolume& gpu_volume) const
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
Volume Volume::as_type(uint8_t type) const
{
    if (_voxel_type == type)
        return *this;

    assert(voxel_num_components(type) == voxel_num_components(_voxel_type));

    Volume dest(_size, type);
    
    uint8_t src_type = voxel_base_type(_voxel_type);
    uint8_t dest_type = voxel_base_type(type);

    size_t num = _size.width * _size.height * _size.depth * voxel_num_components(type);
    if (src_type == VoxelType_Float && dest_type == VoxelType_Double)
        convert_voxels<float, double>(_ptr, dest._ptr, num);
    if (src_type == VoxelType_Double && dest_type == VoxelType_Float)
        convert_voxels<double, float>(_ptr, dest._ptr, num);
    else
        assert(false);

    return dest;
}
bool Volume::valid() const
{
    return _ptr != nullptr;
}
void* Volume::ptr()
{
    assert(_ptr);
    assert(_data->data);
    assert(_data->size);
    return _ptr;
}
void const* Volume::ptr() const
{
    assert(_ptr);
    assert(_data->data);
    assert(_data->size);
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
void Volume::allocate(const Dims& size, uint8_t voxel_type)
{
    assert(voxel_type != VoxelType_Unknown);

    _size = size;
    _voxel_type = voxel_type;

    size_t num_bytes = _size.width * _size.height *
        _size.depth * voxel_size(_voxel_type);

    _data = std::make_shared<VolumeData>(num_bytes);
    _ptr = _data->data;
}
