#include "stb.h"
#include "volume.h"

#pragma warning(push)
#pragma warning(disable: 4996) // 'fopen': This function or variable may be unsafe...

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#pragma warning(pop)

#include <assert.h>
#include <iostream>

namespace stb
{
    Volume read_image(const char* file)
    {
        int x, y, n;
        uint8_t* data = stbi_load(file, &x, &y, &n, 0);

        Dims size = { size_t(x), size_t(y), 1 };
        Volume::VoxelType voxel_type = Volume::VoxelType_Unknown;
        if (n == 1) voxel_type = Volume::VoxelType_UChar;
        if (n == 2) voxel_type = Volume::VoxelType_UChar2;
        if (n == 3) voxel_type = Volume::VoxelType_UChar3;
        if (n == 4) voxel_type = Volume::VoxelType_UChar4;

        Volume vol(size, voxel_type, data);

        stbi_image_free(data);
        return vol;
    }
    void write_image(const char* file, const Volume& volume)
    {
        assert(volume.voxel_type() == Volume::VoxelType_UChar ||
               volume.voxel_type() == Volume::VoxelType_UChar2 ||
               volume.voxel_type() == Volume::VoxelType_UChar3 ||
               volume.voxel_type() == Volume::VoxelType_UChar4);
        assert(volume.size().depth == 1);

        int num_comps = Volume::voxel_num_components(volume.voxel_type());
        Dims size = volume.size();

        /// Supported formats: .png, .bmp, .tga

        const char* ext = file + strlen(file) - 4;
        if (_stricmp(ext, ".png") == 0)
        {
            int ret = stbi_write_png(file, int(size.width), int(size.height), num_comps, volume.ptr(), int(size.width * num_comps));
            assert(ret == 0);
        }
        else if (_stricmp(ext, ".bmp") == 0)
        {
            int ret = stbi_write_bmp(file, int(size.width), int(size.height), num_comps, volume.ptr());
            assert(ret == 0);
        }
        else if (_stricmp(ext, ".tga") == 0)
        {
            int ret = stbi_write_tga(file, int(size.width), int(size.height), num_comps, volume.ptr());
            assert(ret == 0);
        }
        else
        {
            std::cout << "Unsupported image extension: " << ext << std::endl;
        }
    }
}