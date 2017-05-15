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
    uint8_t voxel_type(const GpuVolume& vol)
    {
        if (vol.format_desc.f == cudaChannelFormatKindFloat)
        {
            int num_comp = 0;
            if (vol.format_desc.x > 0) ++num_comp;
            if (vol.format_desc.y > 0) ++num_comp;
            if (vol.format_desc.z > 0) ++num_comp;
            if (vol.format_desc.w > 0) ++num_comp;
        
            if (vol.format_desc.x != 32)
                assert(false && "Unsupported format");

            uint8_t voxel_type = Volume::VoxelType_Unknown;
            if (num_comp == 1) voxel_type = Volume::VoxelType_Float;
            if (num_comp == 2) voxel_type = Volume::VoxelType_Float2;
            if (num_comp == 3) voxel_type = Volume::VoxelType_Float3;
            if (num_comp == 4) voxel_type = Volume::VoxelType_Float4;

            return voxel_type;
        }
        assert(false && "Unsupported format");
        return Volume::VoxelType_Unknown;
    }
}