#include <framework/cuda.h>
#include <framework/gpu_volume.h>
#include <framework/profiler.h>
#include <framework/stb.h>
#include <framework/volume.h>
#include <framework/vtk.h>

namespace sample
{
    void copy_volume(const Volume& in, Volume& out);
}

void run_registration_sandbox();


int main(int argc, char* argv[])
{
    cuda::init(0);
    profiler::register_current_thread("main");

    run_registration_sandbox();

    //Volume vol;
    //{
    //   PROFILE_SCOPE("read_vtk");

    //    vol = vtk::read_volume(R"(C:\projects\azdb\test_img.vtk)")
    //        .as_type(voxel::Type_Float);
    //}

    //Volume copy(vol.size(), vol.voxel_type());
    //sample::copy_volume(vol, copy);

    //{
    //    PROFILE_SCOPE("write_vtk");
    //    vtk::write_volume(R"(C:\projects\azdb\test_img_gpu_copy.vtk)", copy);
    //}


    return 0;
}
