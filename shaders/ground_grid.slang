struct VSInput {
    float3 position : LOCATION0;
    float4 color : LOCATION1;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 world : TEXCOORD0;
};

struct Push {
    column_major float4x4 mvp;
    float4 grid;
    float4 toggles;
};

[[vk::push_constant]]
cbuffer PushConstants {
    Push pc;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput o;
    o.position = mul(pc.mvp, float4(input.position, 1.0));
    o.world    = input.position + (input.color.xyz * 1e-8);
    return o;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    const float step = max(pc.grid.x, 0.0001);
    const float major_step = max(pc.grid.y, step);
    const float extent = max(pc.grid.z, step);
    const float axis_length = max(pc.grid.w, 0.001);

    const float origin_scale = max(pc.toggles.x, 0.001);
    const float show_grid = pc.toggles.y;
    const float show_axes = pc.toggles.z;
    const float show_origin = pc.toggles.w;

    const float2 p = float2(input.world.x, input.world.z);
    const float2 uv_minor = p / step;
    const float2 uv_major = p / major_step;

    const float2 minor_grid = abs(frac(uv_minor - 0.5) - 0.5) / fwidth(uv_minor);
    const float2 major_grid = abs(frac(uv_major - 0.5) - 0.5) / fwidth(uv_major);

    float minor = 1.0 - min(min(minor_grid.x, minor_grid.y), 1.0);
    float major = 1.0 - min(min(major_grid.x, major_grid.y), 1.0);

    if (max(abs(p.x), abs(p.y)) > extent) discard;
    minor *= show_grid;
    major *= show_grid;

    float axis_x = 1.0 - smoothstep(0.0, fwidth(p.y), abs(p.y));
    float axis_z = 1.0 - smoothstep(0.0, fwidth(p.x), abs(p.x));
    axis_x *= saturate(1.0 - abs(p.x) / axis_length) * show_axes;
    axis_z *= saturate(1.0 - abs(p.y) / axis_length) * show_axes;

    float origin_x = axis_x * saturate(1.0 - abs(p.x) / origin_scale);
    float origin_z = axis_z * saturate(1.0 - abs(p.y) / origin_scale);
    float origin = max(origin_x, origin_z) * show_origin;

    const float minor_alpha = minor * 0.6;
    const float major_alpha = major * 0.9;

    float3 color = float3(0.0, 0.0, 0.0);
    color += float3(0.18, 0.18, 0.19) * minor_alpha;
    color += float3(0.32, 0.32, 0.34) * major_alpha;
    color += float3(0.90, 0.15, 0.15) * axis_x;
    color += float3(0.20, 0.40, 0.95) * axis_z;
    color += float3(0.9, 0.9, 0.9) * origin;

    float alpha = max(max(minor_alpha, major_alpha), max(axis_x, axis_z));
    alpha = max(alpha, origin);
    if (alpha < 0.001) discard;

    return float4(saturate(color), saturate(alpha));
}
