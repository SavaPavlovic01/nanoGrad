void atomic_add_float_global_cmpxchg(volatile __global float *addr, float val) {
    union { unsigned int u32; float f32; } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}
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


__kernel void softmax_cross_entropy_naive(
    __global const float* logits,   
    __global const uint* targets,   
    __global float* loss,           
    uint V,
    uint B
) {
    uint b = get_global_id(0);
    if (b >= B) return;

    uint base = b * V;
    uint y = targets[b];

    float m = logits[base];
    for (uint i = 1; i < V; i++) {
        m = fmax(m, logits[base + i]);
    }

    float sum = 0.0f;
    for (uint i = 0; i < V; i++) {
        sum += exp(logits[base + i] - m);
    }

    float log_sum_exp = log(sum);

    loss[b] = -(logits[base + y] - m - log_sum_exp);
}

inline void cross_entropy_backprop_from_probs(__global const float* probs, 
    __global const uint* targets,
    __global float* dest,
    uint V,
    uint B
) {
    uint r = get_global_id(0); 

    if( r >= B) return;

    uint base = r * V;
    uint target = targets[r];

    for(int i = 0; i < V; i++) {
        dest[base + i] = probs[base + i];
        if(i == target) dest[base + i] -= 1;
    }
}

__kernel void cross_entropy_backprop(__global const float* logits, __global const uint* targets, __global float* dest, uint w, uint h) {
    int row = get_global_id(0);
    if(row >= h) return;
    int base = row * w;

    float m = logits[base];
    for(int i = 1; i < w; i++) {
        m = fmax(m, logits[base + i]);
    }

    float sum = 0.0f;

    for(int i = 0; i < w; i++) {
        sum += exp(logits[base + i] - m);
    }

    for(int i = 0; i < w; i++) {
        dest[base + i] = exp(logits[base + i] - m) / sum;
    }   

    uint target = targets[row];

    dest[base + target] -= 1;
}

__kernel void embedding_backward(
    __global const float* X_grad,
    __global float* C_grad,  
    __global const int* indices,
    const int embed_dim      
) {
    int row = get_global_id(0);  // batch index

    int idx = indices[row];  
    for(int d = 0; d < embed_dim; d++) {
        atomic_add_float_global_cmpxchg(&C_grad[idx * embed_dim + d], X_grad[row * embed_dim + d]);
    }
}
