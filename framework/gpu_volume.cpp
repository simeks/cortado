#include <cuda_runtime.h>

#include "gpu_volume.h"
#include "helper_cuda.h"
#include "volume.h"

#include <assert.h>

namespace
{
    cudaChannelFormatDesc create_format_desc(uint8_t voxel_type)
    {
        switch (voxel_type)
        {
        case Volume::VoxelType_Float:
            return cudaCreateChannelDesc<float>();
        case Volume::VoxelType_Float2:
            return cudaCreateChannelDesc<float2>();
        case Volume::VoxelType_Float3:
            return cudaCreateChannelDesc<float3>();
        case Volume::VoxelType_Float4:
            return cudaCreateChannelDesc<float4>();
        case Volume::VoxelType_Double:
            return cudaCreateChannelDesc<double>();
        case Volume::VoxelType_Double2:
            return cudaCreateChannelDesc<double2>();
        case Volume::VoxelType_Double3:
            return cudaCreateChannelDesc<double3>();
        case Volume::VoxelType_Double4:
            return cudaCreateChannelDesc<double4>();
        default:
            assert(false);
        };
        return{ 0 };
    }
}

namespace gpu
{
    GpuVolume allocate_volume(uint8_t voxel_type, const Dims& size)
    {
        GpuVolume vol = { 0 };
        vol.size = { size.width, size.height, size.depth };
        vol.format_desc = create_format_desc(voxel_type);

        checkCudaErrors(cudaMalloc3DArray(&vol.ptr, &vol.format_desc, 
            { size.width, size.height, size.depth }, 0));

        return vol;
    }
    void release_volume(GpuVolume& vol)
    {
        if (vol.ptr == nullptr) // not allocated
            return;

        checkCudaErrors(cudaFreeArray(vol.ptr));
        vol.ptr = nullptr;
        vol.size = { 0, 0, 0 };
    }
}