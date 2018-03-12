#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cudasha256.h"
#include "cuda_helper.h"

__constant__ static unsigned int __align__(8) c_midstate76[8];
__constant__ static unsigned int __align__(8) c_dataEnd80[4];

const __constant__  unsigned int __align__(8) c_H256[8] = {
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};
__constant__ static unsigned int __align__(8) c_K[64];
__constant__ static unsigned int __align__(8) c_target[2];

static unsigned int* d_resNonces[MAX_GPUS] = { 0 };
volatile unsigned int *pFinished = NULL;

unsigned int uiBlocksPerGrid = 0;
unsigned int uiThreadsPerBlock = 0;
// ------------------------------------------------------------------------------------------------

static const unsigned int cpu_H256[8] = {
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};

static const unsigned int cpu_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

#define ROTR ROTR32

__host__
static void sha256_step1_host(unsigned int a, unsigned int b, unsigned int c, unsigned int &d,
	unsigned int e, unsigned int f, unsigned int g, unsigned int &h,
	unsigned int in, const unsigned int Kshared)
{
	unsigned int t1,t2;
	unsigned int vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	unsigned int bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	unsigned int bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	unsigned int andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

__host__
static void sha256_step2_host(unsigned int a, unsigned int b, unsigned int c, unsigned int &d,
	unsigned int e, unsigned int f, unsigned int g, unsigned int &h,
	unsigned int* in, unsigned int pc, const unsigned int Kshared)
{
	unsigned int t1,t2;

	int pcidx1 = (pc-2)  & 0xF;
	int pcidx2 = (pc-7)  & 0xF;
	int pcidx3 = (pc-15) & 0xF;

	unsigned int inx0 = in[pc];
	unsigned int inx1 = in[pcidx1];
	unsigned int inx2 = in[pcidx2];
	unsigned int inx3 = in[pcidx3];

	unsigned int ssg21 = ROTR(inx1, 17) ^ ROTR(inx1, 19) ^ SPH_T32((inx1) >> 10); //ssg2_1(inx1);
	unsigned int ssg20 = ROTR(inx3, 7) ^ ROTR(inx3, 18) ^ SPH_T32((inx3) >> 3); //ssg2_0(inx3);
	unsigned int vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	unsigned int bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	unsigned int bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	unsigned int andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d =  d + t1;
	h = t1 + t2;
}

__host__
static void sha256_round_body_host(unsigned int* in, unsigned int* state, const unsigned int* Kshared)
{
	unsigned int a = state[0];
	unsigned int b = state[1];
	unsigned int c = state[2];
	unsigned int d = state[3];
	unsigned int e = state[4];
	unsigned int f = state[5];
	unsigned int g = state[6];
	unsigned int h = state[7];

	sha256_step1_host(a,b,c,d,e,f,g,h,in[ 0], Kshared[ 0]);
	sha256_step1_host(h,a,b,c,d,e,f,g,in[ 1], Kshared[ 1]);
	sha256_step1_host(g,h,a,b,c,d,e,f,in[ 2], Kshared[ 2]);
	sha256_step1_host(f,g,h,a,b,c,d,e,in[ 3], Kshared[ 3]);
	sha256_step1_host(e,f,g,h,a,b,c,d,in[ 4], Kshared[ 4]);
	sha256_step1_host(d,e,f,g,h,a,b,c,in[ 5], Kshared[ 5]);
	sha256_step1_host(c,d,e,f,g,h,a,b,in[ 6], Kshared[ 6]);
	sha256_step1_host(b,c,d,e,f,g,h,a,in[ 7], Kshared[ 7]);
	sha256_step1_host(a,b,c,d,e,f,g,h,in[ 8], Kshared[ 8]);
	sha256_step1_host(h,a,b,c,d,e,f,g,in[ 9], Kshared[ 9]);
	sha256_step1_host(g,h,a,b,c,d,e,f,in[10], Kshared[10]);
	sha256_step1_host(f,g,h,a,b,c,d,e,in[11], Kshared[11]);
	sha256_step1_host(e,f,g,h,a,b,c,d,in[12], Kshared[12]);
	sha256_step1_host(d,e,f,g,h,a,b,c,in[13], Kshared[13]);
	sha256_step1_host(c,d,e,f,g,h,a,b,in[14], Kshared[14]);
	sha256_step1_host(b,c,d,e,f,g,h,a,in[15], Kshared[15]);

	for (int i=0; i<3; i++)
	{
		sha256_step2_host(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
		sha256_step2_host(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
		sha256_step2_host(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
		sha256_step2_host(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
		sha256_step2_host(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
		sha256_step2_host(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
		sha256_step2_host(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
		sha256_step2_host(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
		sha256_step2_host(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
		sha256_step2_host(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
		sha256_step2_host(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha256_step2_host(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha256_step2_host(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha256_step2_host(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha256_step2_host(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha256_step2_host(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

#define xor3b(a,b,c) (a ^ b ^ c)

__device__ __forceinline__ unsigned int bsg2_0(const unsigned int x)
{
	return xor3b(ROTR32(x,2),ROTR32(x,13),ROTR32(x,22));
}

__device__ __forceinline__ unsigned int bsg2_1(const unsigned int x)
{
	return xor3b(ROTR32(x,6),ROTR32(x,11),ROTR32(x,25));
}

__device__ __forceinline__ unsigned int ssg2_0(const unsigned int x)
{
	return xor3b(ROTR32(x,7),ROTR32(x,18),(x>>3));
}

__device__ __forceinline__ unsigned int ssg2_1(const unsigned int x)
{
	return xor3b(ROTR32(x,17),ROTR32(x,19),(x>>10));
}

__device__ __forceinline__ unsigned int andor32(const unsigned int a, const unsigned int b, const unsigned int c)
{
	unsigned int result;
	asm("{\n\t"
		".reg .u32 m,n,o;\n\t"
		"and.b32 m,  %1, %2;\n\t"
		" or.b32 n,  %1, %2;\n\t"
		"and.b32 o,   n, %3;\n\t"
		" or.b32 %0,  m, o ;\n\t"
		"}\n\t" : "=r"(result) : "r"(a), "r"(b), "r"(c)
	);
	return result;
}

__device__ __forceinline__ uint2 vectorizeswap(uint64_t v) {
	uint2 result;
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(result.y), "=r"(result.x) : "l"(v));
	return result;
}

__device__
static void sha2_step1(unsigned int a, unsigned int b, unsigned int c, unsigned int &d, unsigned int e, unsigned int f, unsigned int g, unsigned int &h,
	unsigned int in, const unsigned int Kshared)
{
	unsigned int t1,t2;
	unsigned int vxandx = xandx(e, f, g);
	unsigned int bsg21 = bsg2_1(e);
	unsigned int bsg20 = bsg2_0(a);
	unsigned int andorv = andor32(a,b,c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

__device__
static void sha2_step2(unsigned int a, unsigned int b, unsigned int c, unsigned int &d, unsigned int e, unsigned int f, unsigned int g, unsigned int &h,
	unsigned int* in, unsigned int pc, const unsigned int Kshared)
{
	unsigned int t1,t2;

	int pcidx1 = (pc-2) & 0xF;
	int pcidx2 = (pc-7) & 0xF;
	int pcidx3 = (pc-15) & 0xF;

	unsigned int inx0 = in[pc];
	unsigned int inx1 = in[pcidx1];
	unsigned int inx2 = in[pcidx2];
	unsigned int inx3 = in[pcidx3];

	unsigned int ssg21 = ssg2_1(inx1);
	unsigned int ssg20 = ssg2_0(inx3);
	unsigned int vxandx = xandx(e, f, g);
	unsigned int bsg21 = bsg2_1(e);
	unsigned int bsg20 = bsg2_0(a);
	unsigned int andorv = andor32(a,b,c);

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d =  d + t1;
	h = t1 + t2;
}

__device__
static void sha256_round_body(unsigned int* in, unsigned int* state, unsigned int* const Kshared)
{
    
	unsigned int a = state[0];
	unsigned int b = state[1];
	unsigned int c = state[2];
	unsigned int d = state[3];
	unsigned int e = state[4];
	unsigned int f = state[5];
	unsigned int g = state[6];
	unsigned int h = state[7];

	sha2_step1(a,b,c,d,e,f,g,h,in[ 0], Kshared[ 0]);
	sha2_step1(h,a,b,c,d,e,f,g,in[ 1], Kshared[ 1]);
	sha2_step1(g,h,a,b,c,d,e,f,in[ 2], Kshared[ 2]);
	sha2_step1(f,g,h,a,b,c,d,e,in[ 3], Kshared[ 3]);
	sha2_step1(e,f,g,h,a,b,c,d,in[ 4], Kshared[ 4]);
	sha2_step1(d,e,f,g,h,a,b,c,in[ 5], Kshared[ 5]);
	sha2_step1(c,d,e,f,g,h,a,b,in[ 6], Kshared[ 6]);
	sha2_step1(b,c,d,e,f,g,h,a,in[ 7], Kshared[ 7]);
	sha2_step1(a,b,c,d,e,f,g,h,in[ 8], Kshared[ 8]);
	sha2_step1(h,a,b,c,d,e,f,g,in[ 9], Kshared[ 9]);
	sha2_step1(g,h,a,b,c,d,e,f,in[10], Kshared[10]);
	sha2_step1(f,g,h,a,b,c,d,e,in[11], Kshared[11]);
	sha2_step1(e,f,g,h,a,b,c,d,in[12], Kshared[12]);
	sha2_step1(d,e,f,g,h,a,b,c,in[13], Kshared[13]);
	sha2_step1(c,d,e,f,g,h,a,b,in[14], Kshared[14]);
	sha2_step1(b,c,d,e,f,g,h,a,in[15], Kshared[15]);

	#pragma unroll
	for (int i=0; i<3; i++)
	{
		sha2_step2(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
		sha2_step2(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}


__device__
static void sha256_round_last(unsigned int* in, unsigned int* state, unsigned int* const Kshared)
{
	unsigned int a = state[0];
	unsigned int b = state[1];
	unsigned int c = state[2];
	unsigned int d = state[3];
	unsigned int e = state[4];
	unsigned int f = state[5];
	unsigned int g = state[6];
	unsigned int h = state[7];

	sha2_step1(a,b,c,d, e,f,g,h, in[ 0], Kshared[ 0]);
	sha2_step1(h,a,b,c, d,e,f,g, in[ 1], Kshared[ 1]);
	sha2_step1(g,h,a,b, c,d,e,f, in[ 2], Kshared[ 2]);
	sha2_step1(f,g,h,a, b,c,d,e, in[ 3], Kshared[ 3]);
	sha2_step1(e,f,g,h, a,b,c,d, in[ 4], Kshared[ 4]);
	sha2_step1(d,e,f,g, h,a,b,c, in[ 5], Kshared[ 5]);
	sha2_step1(c,d,e,f, g,h,a,b, in[ 6], Kshared[ 6]);
	sha2_step1(b,c,d,e, f,g,h,a, in[ 7], Kshared[ 7]);
	sha2_step1(a,b,c,d, e,f,g,h, in[ 8], Kshared[ 8]);
	sha2_step1(h,a,b,c, d,e,f,g, in[ 9], Kshared[ 9]);
	sha2_step1(g,h,a,b, c,d,e,f, in[10], Kshared[10]);
	sha2_step1(f,g,h,a, b,c,d,e, in[11], Kshared[11]);
	sha2_step1(e,f,g,h, a,b,c,d, in[12], Kshared[12]);
	sha2_step1(d,e,f,g, h,a,b,c, in[13], Kshared[13]);
	sha2_step1(c,d,e,f, g,h,a,b, in[14], Kshared[14]);
	sha2_step1(b,c,d,e, f,g,h,a, in[15], Kshared[15]);

	#pragma unroll
	for (int i=0; i<2; i++)
	{
		sha2_step2(a,b,c,d, e,f,g,h, in, 0, Kshared[16+16*i]);
		sha2_step2(h,a,b,c, d,e,f,g, in, 1, Kshared[17+16*i]);
		sha2_step2(g,h,a,b, c,d,e,f, in, 2, Kshared[18+16*i]);
		sha2_step2(f,g,h,a, b,c,d,e, in, 3, Kshared[19+16*i]);
		sha2_step2(e,f,g,h, a,b,c,d, in, 4, Kshared[20+16*i]);
		sha2_step2(d,e,f,g, h,a,b,c, in, 5, Kshared[21+16*i]);
		sha2_step2(c,d,e,f, g,h,a,b, in, 6, Kshared[22+16*i]);
		sha2_step2(b,c,d,e, f,g,h,a, in, 7, Kshared[23+16*i]);
		sha2_step2(a,b,c,d, e,f,g,h, in, 8, Kshared[24+16*i]);
		sha2_step2(h,a,b,c, d,e,f,g, in, 9, Kshared[25+16*i]);
		sha2_step2(g,h,a,b, c,d,e,f, in,10, Kshared[26+16*i]);
		sha2_step2(f,g,h,a, b,c,d,e, in,11, Kshared[27+16*i]);
		sha2_step2(e,f,g,h, a,b,c,d, in,12, Kshared[28+16*i]);
		sha2_step2(d,e,f,g, h,a,b,c, in,13, Kshared[29+16*i]);
		sha2_step2(c,d,e,f, g,h,a,b, in,14, Kshared[30+16*i]);
		sha2_step2(b,c,d,e, f,g,h,a, in,15, Kshared[31+16*i]);
	}

	sha2_step2(a,b,c,d, e,f,g,h, in, 0, Kshared[16+16*2]);
	sha2_step2(h,a,b,c, d,e,f,g, in, 1, Kshared[17+16*2]);
	sha2_step2(g,h,a,b, c,d,e,f, in, 2, Kshared[18+16*2]);
	sha2_step2(f,g,h,a, b,c,d,e, in, 3, Kshared[19+16*2]);
	sha2_step2(e,f,g,h, a,b,c,d, in, 4, Kshared[20+16*2]);
	sha2_step2(d,e,f,g, h,a,b,c, in, 5, Kshared[21+16*2]);
	sha2_step2(c,d,e,f, g,h,a,b, in, 6, Kshared[22+16*2]);
	sha2_step2(b,c,d,e, f,g,h,a, in, 7, Kshared[23+16*2]);
	sha2_step2(a,b,c,d, e,f,g,h, in, 8, Kshared[24+16*2]);
	sha2_step2(h,a,b,c, d,e,f,g, in, 9, Kshared[25+16*2]);
	sha2_step2(g,h,a,b, c,d,e,f, in,10, Kshared[26+16*2]);
	sha2_step2(f,g,h,a, b,c,d,e, in,11, Kshared[27+16*2]);
	sha2_step2(e,f,g,h, a,b,c,d, in,12, Kshared[28+16*2]);
	sha2_step2(d,e,f,g, h,a,b,c, in,13, Kshared[29+16*2]);

	state[6] += g;
	state[7] += h;
}

__device__ __forceinline__
uint64_t cuda_swab32ll(uint64_t x) {
	return MAKE_ULONGLONG(cuda_swab32(_LODWORD(x)), cuda_swab32(_HIDWORD(x)));
}

// 最后一个块的双hash CUDA程序
__global__
/*__launch_bounds__(256,3)*/
void sha256d_gpu_hash_shared(const unsigned int threads, volatile unsigned int *pFinished, const unsigned int startNonce, const unsigned int endNonce, unsigned int *resNonces)
{
	const unsigned int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	unsigned int nonce = startNonce + thread;
	int iLoopTime = 0;
	volatile __shared__ bool someoneFoundIt;

	if(thread >= threads)
	{
		return;
	}

	if (threadIdx.x == 0) someoneFoundIt = *pFinished;
	__syncthreads();

	// 可能多做一个循环
	int iLoopCount = (endNonce - startNonce + 1) / threads + 1;
	while((!someoneFoundIt) && (iLoopTime < iLoopCount) )
	{
		unsigned int dat[16];
		AS_UINT2(dat) = AS_UINT2(c_dataEnd80);
		dat[ 2] = c_dataEnd80[2];
		dat[ 3] = cuda_swab32(nonce);
		dat[ 4] = 0x80000000;
		dat[15] = 0x280;
		#pragma unroll
		for (int i=5; i<15; i++) dat[i] = 0;

		unsigned int buf[8];
		#pragma unroll
		for (int i=0; i<8; i+=2) AS_UINT2(&buf[i]) = AS_UINT2(&c_midstate76[i]);

		sha256_round_body(dat, buf, c_K);

		// second sha256
		#pragma unroll
		for (int i=0; i<8; i++) dat[i] = buf[i];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		#pragma unroll
		for (int i=0; i<8; i++) buf[i] = c_H256[i];

		sha256_round_last(dat, buf, c_K);
//		sha256_round_body(dat, buf, c_K);

		// valid nonces
		uint64_t high = cuda_swab32ll(((uint64_t*)buf)[3]);
		uint64_t *target = (uint64_t *)c_target;
		if (high <= target[0]) {
			unsigned int *pResNonces = (unsigned int *)resNonces;
			resNonces[1] = atomicExch(pResNonces, nonce);
			someoneFoundIt = true;
			*pFinished = true;
		}
		if (threadIdx.x == 0 && *pFinished) someoneFoundIt = true;
		__syncthreads();

		iLoopTime++;
		nonce = startNonce + thread + threads * iLoopTime;
	}
}

__host__
void initSha256d(int deviceId, unsigned int uiBlockNum, unsigned int uiThreadNum)
{
//	cuda_get_arch(deviceId);
	uiBlocksPerGrid = uiBlockNum;
	uiThreadsPerBlock = uiThreadNum;
	cudaSetDevice(deviceId);
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	cudaMemcpyToSymbol(c_K, cpu_K, sizeof(cpu_K), 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[deviceId], 2*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMalloc(&pFinished, sizeof(unsigned int)));
}

__host__
void freeSha256d(int deviceId)
{
	if (d_resNonces[deviceId]) cudaFree(d_resNonces[deviceId]);
	d_resNonces[deviceId] = NULL;
	
	if(pFinished) cudaFree((void *)pFinished);
	pFinished = NULL;
}

//  算比特币第一个区块的hash，在CPU上计算，计算结果拷贝到GPU的全局内存（global或constant）
__host__
void hashFirstBlock(unsigned int *pdata, unsigned int *ptarget)
{
	unsigned int __align__(64) in[16], buf[8], end[4];
	for (int i=0;i<16;i++) in[i] = cuda_swab32(pdata[i]);
	for (int i=0;i<8;i++) buf[i] = cpu_H256[i];
	for (int i=0;i<4;i++) end[i] = cuda_swab32(pdata[16+i]);
	sha256_round_body_host(in, buf, cpu_K);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_midstate76, buf, 32, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_dataEnd80,  end, sizeof(end), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_target, &ptarget[6], 8, 0, cudaMemcpyHostToDevice));
}

// 计算比特币第二个区块（结束区块）的hash，之后对最终结果再做hash，如果小于目标值，则记录到特定地址（resNonces[1]）
__host__
void hashLastBlock(int deviceId, unsigned int threadNumber, unsigned int startNonce, unsigned int endNonce, unsigned int *resNonces)
{
	CUDA_SAFE_CALL(cudaMemset(d_resNonces[deviceId], 0xFF, 2 * sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset((void *)pFinished, 0, sizeof(unsigned int)));
	cudaDeviceSynchronize();
	sha256d_gpu_hash_shared <<<uiBlocksPerGrid, uiThreadsPerBlock>>> (threadNumber, pFinished, startNonce, endNonce, d_resNonces[deviceId]);

	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_resNonces[deviceId], 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	if (resNonces[0] == resNonces[1]) {
		resNonces[1] = UINT32_MAX;
	}
}

// 计算单个hash值,无法真正并行
__global__ void sha256SingleRound(unsigned int threads, unsigned int* in, unsigned int* hashValue)
{
    sha256_round_body(in, hashValue, c_K);
    for(int i = 0; i < 8; i++){
		hashValue[i] = cuda_swab32(hashValue[i]);
	}
}

// 单块hash操作
__host__ void hashSingleBlock(unsigned int threads, unsigned int* in, unsigned int* hashValue)
{
	unsigned int *puiDeviceHashValue;
	unsigned int *puiDeviceIn;

	cudaMalloc( (void **)&puiDeviceHashValue, 8 * sizeof(int) );
	cudaMalloc( (void **)&puiDeviceIn, 16 * sizeof(int) );
	cudaMemcpy(puiDeviceHashValue, hashValue, 8 * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy(puiDeviceIn, in, 16 * sizeof(int), cudaMemcpyHostToDevice );

    sha256SingleRound <<<1, 1>>> (threads, puiDeviceIn, puiDeviceHashValue);
    
    cudaMemcpy(hashValue, puiDeviceHashValue, 8 * sizeof(int), cudaMemcpyDeviceToHost );
	cudaFree(puiDeviceHashValue);
}

__global__ void vecAdd(int *piArray1, int *piArray2, int *piArrayDest, int iLength){
	int tid = threadIdx.x;
	if (tid < iLength){
		piArrayDest[tid] = piArray1[tid] + piArray2[tid];
	}
}

__host__
void vecAddTest(int *piArray1, int *piArray2, int *piArrayDest, int iLength)
{

	int *piArrayDevice1, *piArrayDevice2, *piArrayDeviceDest;
	cudaMalloc( (void **)&piArrayDevice1, iLength * sizeof(int) );
    cudaMalloc( (void **)&piArrayDevice2, iLength * sizeof(int) );
    cudaMalloc( (void **)&piArrayDeviceDest, iLength * sizeof(int) );
	cudaMemset( piArrayDeviceDest, 0, iLength * sizeof(int) );
	cudaMemcpy(piArrayDevice1, piArray1, iLength * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy(piArrayDevice2, piArray2, iLength * sizeof(int), cudaMemcpyHostToDevice );

    vecAdd<<<1, iLength>>>(piArrayDevice1, piArrayDevice2, piArrayDeviceDest, iLength);
    cudaMemcpy(piArrayDest, piArrayDeviceDest, iLength * sizeof(int), cudaMemcpyDeviceToHost );
	cudaThreadSynchronize();
	
	cudaFree(piArrayDevice1);
    cudaFree(piArrayDevice2);
    cudaFree(piArrayDeviceDest);
}
