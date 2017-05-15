#include <framework/cuda.h>
#include <framework/gpu_volume.h>
#include <framework/volume.h>
#include <framework/vtk.h>

int main(int argc, char* argv[])
{
    cuda::init(0);

    Volume vol;
    vtk::read_volume(R"(C:\projects\azdb\test_img.vtk)", vol);
    Volume vol3 = vol.as_type(Volume::VoxelType_Float);
    vtk::write_volume(R"(C:\projects\azdb\test_img_out_pre_float.vtk)", vol);
    vtk::write_volume(R"(C:\projects\azdb\test_img_out_float.vtk)", vol3);

   // GpuVolume gpu_vol = vol.upload();

    //Volume vol2(gpu_vol);
    //vtk::write_volume(R"(C:\projects\azdb\test_img_out.vtk)", vol2);

    return 0;
}
