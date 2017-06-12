#include <framework/stb.h>
#include <framework/volume.h>
#include <framework/volume_helper.h>

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <sstream>

Volume transform(const VolumeHelper<uchar4>& vol, const VolumeHelper<float2>& def);
Volume upsample_deformation_field(const VolumeHelper<float2>& def);

inline uchar4 interp(const VolumeHelper<uchar4>& img, float2 coord)
{
    float fu = floorf(coord.x);
    float fv = floorf(coord.y);
    float cu = ceilf(coord.x);
    float cv = ceilf(coord.y);
    float du = coord.x - fu;
    float dv = coord.y - fv;

    if (fu < 0.0f || cu >= float(img.size().width) ||
        fv < 0.0f || cv >= float(img.size().height))
        return{ 0, 0, 0, 255 };

    uchar4 sff = img(int(fu), int(fv), 0);
    uchar4 sfc = img(int(fu), int(cv), 0);

    uchar4 scf = img(int(cu), int(fv), 0);
    uchar4 scc = img(int(cu), int(cv), 0);

    uchar4 s = {
        (1.0f - du) * (
            (1.0f - dv) * sff.x + dv * sfc.x
            ) +
        du * (
        (1.0f - dv) * scf.x + dv * scc.x
            ),

        (1.0f - du) * (
            (1.0f - dv) * sff.y + dv * sfc.y
                ) +
        du * (
        (1.0f - dv) * scf.y + dv * scc.y
            ),

        (1.0f - du) * (
            (1.0f - dv) * sff.z + dv * sfc.z
                ) +
        du * (
        (1.0f - dv) * scf.z + dv * scc.z
            ),
        255
    };

    return s;
}

template<typename T>
Volume downsample_2(const VolumeHelper<T>& volume)
{
    Dims size = volume.size();
    Dims new_size = { size.width / 2, size.height / 2, std::max<uint32_t>(size.depth / 2, 1) };

    VolumeHelper<T> new_volume = VolumeHelper<T>(new_size);

    for (uint32_t z = 0; z < new_size.depth; ++z)
        for (uint32_t y = 0; y < new_size.height; ++y)
            for (uint32_t x = 0; x < new_size.width; ++x)
            {
                new_volume(x, y, z) = volume(2 * x, 2 * y, 2 * z);
            }

    return new_volume;
}

float reg_weight = 0.05f;
float unary(const VolumeHelper<uchar4>& f, const VolumeHelper<uchar4>& m, int2 pos, float2 def_pos)
{
    // Unary
    uchar4 fixed_i = f(pos.x, pos.y, 0);
    uchar4 moving_i = interp(m, def_pos);

    uchar4 diff = { fixed_i.x*255.0f - moving_i.x*255.0f, fixed_i.y*255.0f - moving_i.y*255.0f, fixed_i.z*255.0f - moving_i.z*255.0f, 0 };

    return (1 - reg_weight) * powf(sqrt(diff.x*diff.x+ diff.y*diff.y+ diff.z*diff.z), 2);
}

float binary(const VolumeHelper<float2>& def, const VolumeHelper<uint8_t>& move_label, bool move, float2 delta, int2 pos, int level)
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

        float pixel_size = pow(2, level);
        float norm2 = pixel_size*pixel_size*(my_def.x - n_def.x)*(my_def.x - n_def.x) + pixel_size*pixel_size*(my_def.y - n_def.y)*(my_def.y - n_def.y);
        cost += norm2;
    }
    return reg_weight * cost;
}

size_t do_reg_move(const VolumeHelper<uchar4>& f, const VolumeHelper<uchar4>& m, float2 delta,
    VolumeHelper<float2>& def, int level)
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
                    float rx = x + def(x, y, z).x;
                    float ry = y + def(x, y, z).y;

                    float u0 = unary(f, m, { int(x), int(y) }, { rx, ry }); // Unary cost for no move
                    float u1 = unary(f, m, { int(x), int(y) }, { rx + delta.x, ry + delta.y }); // Unary cost to apply move

                    float b0 = binary(def, move_label, false, delta, { int(x), int(y) }, level); // binary cost for no move
                    float b1 = binary(def, move_label, true, delta, { int(x), int(y) }, level); // binary cost to apply move

                    uint8_t prev = move_label(x, y, z);
                    if (u0 + b0 <= u1 + b1)
                    //if (u0 <= u1)
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
                float rx = x + def(x, y, z).x;
                float ry = y + def(x, y, z).y;

                float u0 = unary(f, m, { int(x), int(y) }, { rx, ry }); // Unary cost for no move
                float u1 = unary(f, m, { int(x), int(y) }, { rx + delta.x, ry + delta.y }); // Unary cost to apply move

                float b0 = binary(def, move_label, false, delta, { int(x), int(y) }, level); // binary cost for no move
                float b1 = binary(def, move_label, true, delta, { int(x), int(y) }, level); // binary cost to apply move

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
    std::cout << total_cost << ", #moves: " << total_moves << " d: " << delta.x << ", " << delta.y << ", level: " << level << std::endl;

    static double last_cost = 1000000000000.0;
    static int cost_counter = 0;
    if (total_cost >= last_cost)
    {
        if (cost_counter == 10)
        {
            cost_counter = 0;
            last_cost = 1000000000000.0;
            return 0xffffffffffffffff;
        }
        ++cost_counter;
    }
    last_cost = total_cost;

    static int counter = 0;
    if (++counter % 10 == 0)
    {
        auto out = transform(m, def);
        stb::write_image(R"(C:\projects\non-parametric-test3\progress.png)", out);
    }

    return total_moves;
}

Volume do_reg(const VolumeHelper<uchar4>& f, const VolumeHelper<uchar4>& m, const VolumeHelper<float2>& start, int level)
{
    assert(f.size().width == m.size().width);
    assert(f.size().height == m.size().height);
    assert(f.size().depth == m.size().depth);

    VolumeHelper<float2> def(f.size(), { 0, 0 });
    if (start.valid())
        def = start.clone();

    uint8_t move_mask = 1;
    while (move_mask != 0)
    {
        move_mask = 0;

        int2 deltas[] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
        for (int i = 0; i < 4; ++i)
        {
            int2 d = deltas[i];

            size_t moves = do_reg_move(f, m, { d.x*0.5f, d.y*0.5f }, def, level);
            if (moves == 0xffffffffffffffff)
                return def;
            if (moves > 0)
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
    for (uint32_t y = 0; y < size.height; ++y)
    {
        for (uint32_t x = 0; x < size.width; ++x)
        {
            float u = x + def(x, y, 0).x;
            float v = y + def(x, y, 0).y;

            out(x, y, 0) = interp(vol, {u, v});
        }
    }
    return out;
}

Volume gaussian_filter_2d(const VolumeHelper<uchar4>& vol, float sigma)
{
    Dims size = vol.size();

    VolumeHelper<uchar4> tmp = vol.clone();

    float factor = -1.0f / (2 * sigma*sigma);
    int fsize = (int)ceil(2 * sigma);

    //X dimension
    for (uint32_t y = 0; y < size.height; ++y)
    {
        for (uint32_t x = 0; x < size.width; ++x)
        {
            float4 value = { 0.0f, 0.0f, 0.0f, 0.0f };
            float norm = 0;

            for (long t = -fsize; t <= fsize; t++)
            {
                if (x + t < 0 || x + t > tmp.size().width)
                    continue;

                float c = exp(factor*t*t);
                value.x += c*vol(x + t, y, 0).x;
                value.y += c*vol(x + t, y, 0).y;
                value.z += c*vol(x + t, y, 0).z;
                norm += c;
            }
            tmp(x, y, 0).x = uint8_t(value.x / norm);
            tmp(x, y, 0).y = uint8_t(value.y / norm);
            tmp(x, y, 0).z = uint8_t(value.z / norm);
            tmp(x, y, 0).w = 255;
        }
    }

    VolumeHelper<uchar4> tmp2 = tmp.clone();

    //Y dimension
    for (uint32_t y = 0; y < size.height; ++y)
    {
        for (uint32_t x = 0; x < size.width; ++x)
        {
            float4 value = { 0.0f, 0.0f, 0.0f, 0.0f };
            float norm = 0;

            for (long t = -fsize; t <= fsize; t++)
            {
                if (y + t < 0 || y + t > tmp2.size().height)
                    continue;

                float c = exp(factor*t*t);
                value.x += c*tmp2(x, y + t, 0).x;
                value.y += c*tmp2(x, y + t, 0).y;
                value.z += c*tmp2(x, y + t, 0).z;
                norm += c;
            }
            tmp(x, y, 0).x = uint8_t(value.x / norm);
            tmp(x, y, 0).y = uint8_t(value.y / norm);
            tmp(x, y, 0).z = uint8_t(value.z / norm);
            tmp(x, y, 0).w = 255;
        }
    }

    return tmp;
}

Volume upsample_deformation_field(const VolumeHelper<float2>& def)
{
    Dims dims = { def.size().width * 2, def.size().height * 2, 1 };
    
    VolumeHelper<float2> out = VolumeHelper<float2>({ dims.width, dims.height, 1 });

    for (int y = 0; y < dims.height; ++y)
    {
        for (int x = 0; x < dims.width; ++x)
        {
            //float2 coord = {0.5f * x, 0.5f * y};

            //float2 floor = def(floorf(coord.x), floorf(coord.y), 0);
            //float2 ceil = def(ceilf(coord.x), ceilf(coord.y), 0);

            //float dx = coord.x - floorf(coord.x);
            //float dy = coord.y - floorf(coord.y);

            //int2 icoord = { int((1 - dx)*floor.x + dx*ceil.x), int((1 - dy)*floor.y + dy*ceil.y) };
            //if (icoord.x < 0 || icoord.x >= def.size().width ||
            //    icoord.y < 0 || icoord.y >= def.size().height)
            //{
            //    out(x, y, 0) = { 0.0f, 0.0f };
            //    continue;
            //}

            float u = 0.5f*x;
            float v = 0.5f*y;
            float fu = floorf(u);
            float fv = floorf(v);
            float cu = std::min(ceilf(u), (float)def.size().width-1);
            float cv = std::min(ceilf(v), (float)def.size().height-1);
            float du = u - fu;
            float dv = v - fv;

            float2 sff = def(int(fu), int(fv), 0);
            float2 sfc = def(int(fu), int(cv), 0);

            float2 scf = def(int(cu), int(fv), 0);
            float2 scc = def(int(cu), int(cv), 0);

            float2 s = {
                (1.0f - du) * (
                    (1.0f - dv) * sff.x + dv * sfc.x
                    ) +
                du * (
                    (1.0f - dv) * scf.x + dv * scc.x
                    ),

                (1.0f - du) * (
                    (1.0f - dv) * sff.y + dv * sfc.y
                ) +
                du * (
                    (1.0f - dv) * scf.y + dv * scc.y
                )
            };

            out(x, y, 0) = { s.x, s.y };
        }
    }
    return out;
}

Volume deformation_magnitude(const VolumeHelper<float2>& def)
{
    VolumeHelper<uint8_t> out(def.size());
    for (int y = 0; y < def.size().height; ++y)
    {
        for (int x = 0; x < def.size().width; ++x)
        {
            float2 d = def(x, y, 0);
            out(x, y, 0) = uint8_t(sqrt(d.x * d.x + d.y * d.y)*0.5f);
        }
    }
    return out;
}


Volume deformation_color(const VolumeHelper<float2>& def)
{
    VolumeHelper<uchar4> out(def.size());
    for (int y = 0; y < def.size().height; ++y)
    {
        for (int x = 0; x < def.size().width; ++x)
        {
            float2 d = def(x, y, 0);
            out(x, y, 0) = {uint8_t(127+d.x*0.25f), uint8_t(127 + d.y*0.25f), 127, 255};
        }
    }
    return out;
}

void run_registration_sandbox()
{
    const int n_levels = 8;
    const int max_level = 1;

    Volume fixed_pyramid[n_levels];
    Volume moving_pyramid[n_levels];
    Volume def_pyramid[n_levels];

    fixed_pyramid[0] = stb::read_image(R"(C:\projects\non-parametric-test3\img_fixed_ez.png)");
    moving_pyramid[0] = stb::read_image(R"(C:\projects\non-parametric-test3\img_moving.png)");

    //VolumeHelper<float2> def(moving_pyramid[0].size(), { 20.0f, 20.0f });
    //stb::write_image(R"(C:\projects\non-parametric-test3\test.png)", transform(moving_pyramid[0], def));


    //return;
    for (int i = 1; i < n_levels; ++i)
    {
        fixed_pyramid[i] = downsample_2<uchar4>(gaussian_filter_2d(fixed_pyramid[i-1], 8.0f));
        moving_pyramid[i] = downsample_2<uchar4>(gaussian_filter_2d(moving_pyramid[i-1], 8.0f));
        //fixed_pyramid[i] = downsample_2<uchar4>(fixed_pyramid[i - 1]);
        //moving_pyramid[i] = downsample_2<uchar4>(moving_pyramid[i - 1]);

        std::stringstream ss;
        ss << R"(C:\projects\non-parametric-test3\fixed_)" << i << ".png";
        stb::write_image(ss.str().c_str(), fixed_pyramid[i]);

        ss.str("");
        ss << R"(C:\projects\non-parametric-test3\moving_)" << i << ".png";
        stb::write_image(ss.str().c_str(), moving_pyramid[i]);
    }

    def_pyramid[n_levels];
    for (int i = n_levels - 1; i >= 0; --i)
    {
        Volume prev_def;
        if (i < n_levels - 1)
            prev_def = upsample_deformation_field(def_pyramid[i + 1]);
        else
            prev_def = VolumeHelper<float2>(fixed_pyramid[i].size(), { 0.0f, 0.0f });

        if (i >= max_level)
        {
            def_pyramid[i] = do_reg(fixed_pyramid[i], moving_pyramid[i], prev_def, i);
        }
        else
        {
            def_pyramid[i] = prev_def;
        }

        auto out = transform(moving_pyramid[i], def_pyramid[i]);

        std::stringstream ss;
        ss << R"(C:\projects\non-parametric-test3\result_)" << i << ".png";
        stb::write_image(ss.str().c_str(), out);
    }
    
    stb::write_image(R"(C:\projects\non-parametric-test3\def.png)", deformation_magnitude(def_pyramid[0]));
    stb::write_image(R"(C:\projects\non-parametric-test3\def_color.png)", deformation_color(def_pyramid[0]));
    auto out = transform(moving_pyramid[0], def_pyramid[0]);
    stb::write_image(R"(C:\projects\non-parametric-test3\result.png)", out);
}