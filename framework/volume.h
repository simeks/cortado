#pragma once

#include <channel_descriptor.h>
#include <vector>

struct Dims
{
    size_t width;
    size_t height;
    size_t depth;
};

struct CudaVolume
{
    struct cudaArray* ptr;
    Dims size;
    cudaChannelFormatDesc format_desc;
};

class VolumeBase
{
public:
    VolumeBase(size_t elem_size);
    virtual ~VolumeBase();

    void copy_to_gpu();
    void copy_from_gpu();

    void* data();
    void const* data() const;

protected:
    void allocate_volume(const Dims& size);
    
    void allocate_gpu_volume(const struct cudaChannelFormatDesc& desc, const struct cudaExtent& size);
    void release_gpu_volume();

    CudaVolume _gpu_vol;

    Dims _size;
    const size_t _element_size;

    std::vector<uint8_t> _data;
};

template<typename T>
class Volume : public VolumeBase
{
public:
    Volume();
    Volume(const Dims& size);
    ~Volume();

    T* data();
    T const* data() const;

    void allocate_gpu_volume();
};

template<typename T>
Volume<T>::Volume() : VolumeBase(sizeof(T)) {}

template<typename T>
Volume<T>::Volume(const Dims& size) : VolumeBase(sizeof(T)) { allocate_volume(size); }

template<typename T>
Volume<T>::~Volume() {}

template<typename T>
T* Volume<T>::data()
{
    return (T*)VolumeBase::data();
}
template<typename T>
T const* Volume<T>::data() const
{
    return (T const*)VolumeBase::data();
}

template<typename T>
void Volume<T>::allocate_gpu_volume()
{
    VolumeBase::allocate_gpu_volume(cudaCreateChannelDesc<T>(), { _size.width, _size.height, _size.depth });
}

