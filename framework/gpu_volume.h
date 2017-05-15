#pragma once

#include "dims.h"

#include <channel_descriptor.h>
#include <stdint.h>

struct GpuVolume
{
    struct cudaArray* ptr;
    Dims size;
    cudaChannelFormatDesc format_desc;
};

namespace gpu
{
    /// Allocates a GPU volume of the specified type and size
    /// @param voxel_type : See Volume::VoxelType
    GpuVolume allocate_volume(uint8_t voxel_type, const Dims& size);

    /// Releases a volume allocated with allocate_volume
    void release_volume(GpuVolume& vol);
}
