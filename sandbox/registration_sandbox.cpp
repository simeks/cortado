#include <framework/stb.h>
#include <framework/volume.h>
#include <framework/volume_helper.h>

#include <algorithm>
#include <assert.h>
#include <iostream>

Volume downsample_2(const Volume& volume)
{
    Dims size = volume.size();
    Dims new_size = { size.width / 2, size.height / 2, std::max<uint32_t>(size.depth / 2, 1) };

    Volume new_volume = Volume(new_size, volume.voxel_type());

    size_t voxel_size = voxel::size(volume.voxel_type());
    uint8_t* src = (uint8_t*)volume.ptr();
    uint8_t* dest = (uint8_t*)new_volume.ptr();

    for (uint32_t z = 0; z < new_size.depth; ++z)
        for (uint32_t y = 0; y < new_size.height; ++y)
            for (uint32_t x = 0; x < new_size.width; ++x)
            {
                memcpy( dest + voxel_size * (z * new_size.width * new_size.height + y * new_size.width + x),
                        src + voxel_size * (2 * z * size.width * size.height + 2 * y * size.width + 2 * x),
                        voxel_size);
            }

    return new_volume;
}

float reg_weight = 0.1f;
float unary(const VolumeHelper<uchar4>& f, const VolumeHelper<uchar4>& m, int2 pos, int2 def_pos)
{
    // Unary
    float fixed_i = float(f(pos.x, pos.y, 0).x);
    float moving_i = 0.0f;
    if (def_pos.x >= 0 && def_pos.y >= 0 && def_pos.x < f.size().width && def_pos.y < f.size().height)
        moving_i = float(m(def_pos.x, def_pos.y, 0).x);


    return (1 - reg_weight) * powf(fixed_i - moving_i, 2);
}

float binary(const VolumeHelper<float2>& def, const VolumeHelper<uint8_t>& move_label, bool move, int2 delta, int2 pos)
{
    float2 my_def = def(pos.x, pos.y, 0);
    if (move) my_def = { my_def.x + delta.x, my_def.y + delta.y };

    float cost = 0;

    int2 neighbors[] = { { -1, 0 },{ 1, 0 },{ 0, -1 },{ 0, 1 } };
    for (int i = 0; i < 4; ++i)
    {
        int2 np = { pos.x + neighbors[i].x, pos.y + neighbors[i].y };
        if (np.x < 0 || np.y < 0)
            continue;
        if (np.x >= int(def.size().width) || np.y >= int(def.size().height))
            continue;

        float2 n_def = def(np.x, np.y, 0);
        if (move_label(np.x, np.y, 0) > 0) n_def = { n_def.x + delta.x, n_def.y + delta.y };

        float norm2 = (my_def.x - n_def.x)*(my_def.x - n_def.x) + (my_def.y - n_def.y)*(my_def.y - n_def.y);
        cost += norm2;
    }
    return reg_weight * cost;
}

size_t do_reg_move(const VolumeHelper<uchar4>& f, const VolumeHelper<uchar4>& m, int2 delta,
    VolumeHelper<float2>& def)
{
    int n = 0;
    bool done = false;
    size_t total_moves = 0;

    VolumeHelper<uint8_t> move_label(f.size(), 0);

    Dims size = f.size();
    while (!done)
    {
        size_t n_moves = 0;
        for (uint32_t z = 0; z < size.depth; ++z)
        {
            for (uint32_t y = 0; y < size.height; ++y)
            {
                for (uint32_t x = 0; x < size.width; ++x)
                {
                    int rx = int(x) + int(def(x, y, z).x);
                    int ry = int(y) + int(def(x, y, z).y);

                    float u0 = unary(f, m, { int(x), int(y) }, { rx, ry }); // Unary cost for no move
                    float u1 = unary(f, m, { int(x), int(y) }, { rx + delta.x, ry + delta.y }); // Unary cost to apply move

                    float b0 = binary(def, move_label, false, delta, { int(x), int(y) }); // binary cost for no move
                    float b1 = binary(def, move_label, true, delta, { int(x), int(y) }); // binary cost to apply move

                    uint8_t prev = move_label(x, y, z);
                    if (u0 + b0 <= u1 + b1)
                        move_label(x, y, z) = 0;
                    else
                        move_label(x, y, z) = 1;

                    if (move_label(x, y, z) != prev)
                        ++n_moves;
                }
            }
        }
        total_moves += n_moves;
        done = n_moves == 0;
    }

    double total_cost = 0;
    // Apply resulting moves
    for (uint32_t z = 0; z < size.depth; ++z)
    {
        for (uint32_t y = 0; y < size.height; ++y)
        {
            for (uint32_t x = 0; x < size.width; ++x)
            {
                int rx = int(x) + int(def(x, y, z).x);
                int ry = int(y) + int(def(x, y, z).y);

                float u0 = unary(f, m, { int(x), int(y) }, { rx, ry }); // Unary cost for no move
                float u1 = unary(f, m, { int(x), int(y) }, { rx + delta.x, ry + delta.y }); // Unary cost to apply move

                float b0 = binary(def, move_label, false, delta, { int(x), int(y) }); // binary cost for no move
                float b1 = binary(def, move_label, true, delta, { int(x), int(y) }); // binary cost to apply move

                if (move_label(x, y, z) > 0)
                {
                    total_cost += u1 + b1;

                    def(x, y, z).x += delta.x;
                    def(x, y, z).y += delta.y;
                }
                else
                    total_cost += u0 + b0;
            }
        }
    }
    std::cout << total_cost << ", #moves: " << total_moves << std::endl;

    return total_moves;
}

Volume do_reg(const VolumeHelper<uchar4>& f, const VolumeHelper<uchar4>& m)
{
    assert(f.size().width == m.size().width);
    assert(f.size().height == m.size().height);
    assert(f.size().depth == m.size().depth);

    VolumeHelper<float2> def(f.size(), { 0, 0 });

    uint8_t move_mask = 1;
    while (move_mask != 0)
    {
        move_mask = 0;

        int2 deltas[] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
        for (int i = 0; i < 4; ++i)
        {
            int2 d = deltas[i];

            if (do_reg_move(f, m, d, def) > 0)
            {
                move_mask |= 1 << i;
            }
        }
    }
    return def;
}

Volume transform(const VolumeHelper<uchar4>& vol, const VolumeHelper<float2>& def)
{
    VolumeHelper<uchar4> out = vol.clone();

    Dims size = vol.size();
    for (uint32_t z = 0; z < size.depth; ++z)
    {
        for (uint32_t y = 0; y < size.height; ++y)
        {
            for (uint32_t x = 0; x < size.width; ++x)
            {
                int rx = (int)roundf(x + def(x, y, z).x);
                int ry = (int)roundf(y + def(x, y, z).y);

                rx = std::min<int>((int)size.width, std::max(0, rx));
                ry = std::min<int>(size.height, std::max(0, ry));

                out(x, y, z) = vol(rx, ry, 0);
            }
        }
    }
    return out;
}

void run_registration_sandbox()
{
    auto fixed_img = stb::read_image(R"(C:\projects\non-parametric-test3\img_fixed.png)");
    auto moving_img = stb::read_image(R"(C:\projects\non-parametric-test3\img_moving.png)");

    auto scaled_fixed_img = downsample_2(fixed_img);
    scaled_fixed_img = downsample_2(scaled_fixed_img);
    scaled_fixed_img = downsample_2(scaled_fixed_img);

    auto scaled_moving_img = downsample_2(moving_img);
    scaled_moving_img = downsample_2(scaled_moving_img);
    scaled_moving_img = downsample_2(scaled_moving_img);


    auto def = do_reg(fixed_img, moving_img);
    auto out = transform(moving_img, def);
    stb::write_image(R"(C:\projects\non-parametric-test3\result.png)", out);
}