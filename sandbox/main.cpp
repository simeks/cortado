#include <framework/cuda.h>
#include <framework/gpu_volume.h>
#include <framework/volume.h>
#include <framework/vtk.h>

int main(int argc, char* argv[])
{
    cuda::init(0);

    Volume vol;
    vtk::read_volume(R"(C:\projects\azdb\test_img.vtk)", vol);
    vtk::write_volume(R"(C:\projects\azdb\test_img_out.vtk)", vol);

    return 0;
}
