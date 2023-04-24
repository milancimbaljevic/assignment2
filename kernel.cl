__kernel void matrix_multiplication(__global const float *A, __global const float* B, __global float* C, int m, int n, int k) {
	
	const int tx = get_global_id(0);
	const int ty = get_global_id(1);

	// k = width of A and height of B
	// n = width of B

	float acc = 0.0f;
	for (int cnt = 0; cnt < k; cnt++) {
		acc += A[ty * k + cnt] * B[cnt*n + tx];
	}

	C[ty * n + tx] = acc;
}

// with memory tiling

//__kernel void matrix_multiplication(__global const float* A, __global const float* B, __global float* C, int m, int n, int k) {
//
//	const int tx = get_global_id(0);
//	const int ty = get_global_id(1);
//	const int lx = get_local_id(0);
//	const int ly = get_local_id(1);
//
//	__local float A_sub[32][32];
//	__local float B_sub[32][32];
//
//	float acc = 0.0f;
//	for (int currentTile = 0; currentTile < k / 32; currentTile++) {
//		A_sub[ly][lx] = A[ty*k + currentTile * 32 + lx];
//		B_sub[ly][lx] = B[(currentTile * 32 + ly) * n + tx];
//	
//		barrier(CLK_LOCAL_MEM_FENCE);
//
//		for (int cnt = 0; cnt < 32; cnt++) {
//			acc += A_sub[ly][cnt] * B_sub[cnt][lx];
//		}
//
//		barrier(CLK_LOCAL_MEM_FENCE);
//	}
//	
//	C[ty * n + tx] = acc;
//}

