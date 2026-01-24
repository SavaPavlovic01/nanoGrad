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

#define TILE_WIDTH 16

__kernel void matrixMult_tile(
    __global const float* a, uint M, uint N,  
    __global const float* b, uint Mb, uint Nb,
    __global float* dest) {

        int global_row = get_global_id(0);
        int global_col = get_global_id(1);

        int local_row = get_local_id(0);
        int local_col = get_local_id(1);
        
        const int num_tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

        __local float shared_A[TILE_WIDTH][TILE_WIDTH];
        __local float shared_B[TILE_WIDTH][TILE_WIDTH];

        float acc = 0.0f;

        for(int i = 0; i < num_tiles; i++) {
            int b_row = i * TILE_WIDTH + local_row;
            int a_col = i * TILE_WIDTH + local_col;

            if(global_row >= M || a_col >= N) shared_A[local_row][local_col] = 0.0f;
            else shared_A[local_row][local_col] = a[global_row * N + a_col];

            if(global_col >= Nb || b_row >= Mb) shared_B[local_row][local_col] = 0.0f;
            else shared_B[local_row][local_col] = b[b_row * Nb + global_col];

            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for(int index = 0; index < TILE_WIDTH; index++){
                acc += shared_A[local_row][index] * shared_B[index][local_col];
            }


            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(global_row < M && global_col < Nb) dest[global_row * Nb + global_col] = acc;
}