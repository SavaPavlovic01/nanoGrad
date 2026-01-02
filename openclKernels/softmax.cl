__kernel void softmax_ndim(__global const float* src, __global float* dest, __global const uint* shape, __global const uint* stride, uint numel) {

}

__kernel void softmax_naive(__global const float* src, __global float* dest, uint w, uint h) {
    int row = get_global_id(0);
    if(row >= h) return;
    int base = row * w;

    float m = src[base];
    for(int i = 1; i < w; i++) {
        m = fmax(m, src[base + i]);
    }

    float sum = 0.0f;

    for(int i = 0; i < w; i++) {
        sum += exp(src[base + i] - m);
    }

    for(int i = 0; i < w; i++) {
        dest[base + i] = exp(src[base + i] - m) / sum;
    }
}