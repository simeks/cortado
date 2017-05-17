#pragma once

#include "dims.h"

#include <memory>
#include <vector>

struct GpuVolume;

struct VolumeData
{
    VolumeData();
    VolumeData(size_t size);
    ~VolumeData();

    uint8_t* data;
    size_t size;
};
typedef std::shared_ptr<VolumeData> VolumeDataPtr;

class Volume
{
public:
    enum VoxelType : uint8_t
    {
        VoxelType_Unknown = 0,
        VoxelType_Float,
        VoxelType_Float2,
        VoxelType_Float3,
        VoxelType_Float4,
        VoxelType_Double,
        VoxelType_Double2,
        VoxelType_Double3,
        VoxelType_Double4,
        VoxelType_UChar,
        VoxelType_UChar2,
        VoxelType_UChar3,
        VoxelType_UChar4
    };

    Volume();
    Volume(const Dims& size, uint8_t voxel_type, uint8_t* data = nullptr);
    /// Creates a new volume on the CPU side and downloads the given volume from the gpu into it.
    Volume(const GpuVolume& gpu_volume);
    ~Volume();

    /// Uploads this volume to a newly allocated GPU volume
    /// @remark Requires both volumes to be of same size and type
    /// @return Handle to newly created GPU volume
    GpuVolume upload() const;

    /// Uploads this volume to given GPU volume
    /// @remark Requires both volumes to be of same size and type
    void upload(const GpuVolume& gpu_volume) const;

    /// Downloads the given volume into this volume
    /// @remark Requires both volumes to be of same size and type
    void download(const GpuVolume& gpu_volume);

    /// Release any allocated data the volume is holding
    /// This makes the volume object invalid
    void release();

    /// Clones this volume
    Volume clone() const;

    /// Attempts to convert this volume to the specified format,
    ///     If this volume already is of the specified format it will just return itself.
    ///     If not a copied converted version will be returned.
    /// Note:
    ///     You can only convert between volumes with the same number of components per voxel.
    ///     I.e. you cannot convert from Float3 to Int2.
    Volume as_type(uint8_t type) const;

    /// Returns true if the volume is allocated and ready for use
    bool valid() const;

    void* ptr();
    void const* ptr() const;

    uint8_t voxel_type() const;
    const Dims& size() const;

    /// @remark This does not copy the data, use clone if you want a separate copy.
    Volume(const Volume& other);
    Volume& operator=(const Volume& other);

    static size_t voxel_size(uint8_t type);
    static int voxel_num_components(uint8_t type);
    
    /// Returns the base type of a type, i.e. Float3 -> Float
    static uint8_t voxel_base_type(uint8_t type);

private:
    void allocate(const Dims& size, uint8_t voxel_type);

    VolumeDataPtr _data;
    void* _ptr; // Pointer to a location in _data

    Dims _size;
    uint8_t _voxel_type;
};
