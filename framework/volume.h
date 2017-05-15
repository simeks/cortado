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

    std::vector<uint8_t> data;
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
        VoxelType_Double4
    };

    Volume();
    Volume(const Dims& size, uint8_t voxel_type, uint8_t* data = nullptr);
    ~Volume();

    /// Uploads this volume to a newly allocated GPU volume
    /// @remark Requires both volumes to be of same size and type
    /// @return Handle to newly created GPU volume
    GpuVolume upload();

    /// Uploads this volume to given GPU volume
    /// @remark Requires both volumes to be of same size and type
    void upload(const GpuVolume& gpu_volume);

    /// Downloads the given volume into this volume
    /// @remark Requires both volumes to be of same size and type
    void download(const GpuVolume& gpu_volume);

    /// Release any allocated data the volume is holding
    /// This makes the volume object invalid
    void release();

    /// Clones this volume
    Volume clone() const;

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
private:
    VolumeDataPtr _data;
    void* _ptr; // Pointer to a location in _data

    Dims _size;
    uint8_t _voxel_type;
};

template<typename T>
class VolumeTpl
{
public:

};
//
//class VolumeBase
//{
//public:
//    enum Type
//    {
//        Type_Float,
//        Type_Float1,
//        Type_Float2,
//        Type_Float3,
//        Type_Float4
//    };
//
//    VolumeBase(size_t elem_size);
//    virtual ~VolumeBase();
//
//    void* data();
//    void const* data() const;
//
//protected:
//    void allocate_volume(const Dims& size);
//    
//    void allocate_gpu_volume(const struct cudaChannelFormatDesc& desc, const struct cudaExtent& size);
//    void release_gpu_volume();
//
//    CudaVolume _gpu_vol;
//
//    Dims _size;
//    
//    const size_t _element_size;
//    const int _element_count;
//
//    std::vector<uint8_t> _data;
//};
//
//template<typename T>
//class Volume : public VolumeBase
//{
//public:
//    Volume();
//    Volume(const Dims& size);
//    ~Volume();
//
//    T* data();
//    T const* data() const;
//
//    void allocate_gpu_volume();
//};
//
//template<typename T>
//Volume<T>::Volume() : VolumeBase(sizeof(T)) {}
//
//template<typename T>
//Volume<T>::Volume(const Dims& size) : VolumeBase(sizeof(T)) { allocate_volume(size); }
//
//template<typename T>
//Volume<T>::~Volume() {}
//
//template<typename T>
//T* Volume<T>::data()
//{
//    return (T*)VolumeBase::data();
//}
//template<typename T>
//T const* Volume<T>::data() const
//{
//    return (T const*)VolumeBase::data();
//}
//
//template<typename T>
//void Volume<T>::allocate_gpu_volume()
//{
//    VolumeBase::allocate_gpu_volume(cudaCreateChannelDesc<T>(), { _size.width, _size.height, _size.depth });
//}
