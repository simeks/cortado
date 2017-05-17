#pragma once

#include "volume.h"

template<typename T>
class VolumeHelper : public Volume
{
public:
    VolumeHelper(Volume& other);
    ~VolumeHelper();


};


#include "volume_helper.inl"
