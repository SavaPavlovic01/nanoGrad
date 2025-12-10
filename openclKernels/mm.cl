int getIndex(int i, int j, int strideM, int strideN) {
    return j * strideN + i * strideM;
}

__kernel void matrixMult_simple(
    __global const float* a, uint M, uint N, uint strideM, uint strideN, 
    __global const float* b, uint Mb, uint Nb, uint strideMB, uint strideNB, 
    __global float* dest) {

        int idx = get_global_id(0);
        int idy = get_global_id(1);

        if(idx >= M || idy >= Nb) return;

        float val = 0;
        for(int i = 0; i < N; i++) {
            val += a[getIndex(idx, i, strideM, strideN)] * b[getIndex(i, idy, strideMB, strideNB)];
        }
        dest[idx * Nb + idy] = val;
        
        return;
}

