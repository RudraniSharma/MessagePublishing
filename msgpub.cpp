
#include <iostream>
#include <cuda.h>


/**
  Allocate this class in CPU/GPU unified memory.  
  Inherit to always be unified.
*/
class Unified {
public:
/** Allocate instances in CPU/GPU unified memory */
  void *operator new(size_t len) {
	void *ptr;
	cudaMallocManaged(&ptr, len);
	return ptr;
  }
  void operator delete(void *ptr) {
	cudaFree(ptr);
  }

/** Allocate all arrays in CPU/GPU unified memory */
  void* operator new[] (std::size_t size) {
	void *ptr; 
	cudaMallocManaged(&ptr,size);
	return ptr;
  }
  void operator delete[] (void* ptr) {
	cudaFree(ptr);
  }
};


// The application would be built with Unified classes,
//   which are accessible from either CPU or GPU.
class widget : public Unified 
{
public:
	float value;

	/*
	This method is meant to run on the GPU (__device__)
	By default methods run on the CPU (__host__)
	*/
	__device__ void setValue(float v) { value=v; }
};

/* GPU kernel: set an array of widgets to a value */
__global__ void set_array(widget *w,float param) {
	int i=threadIdx.x + blockIdx.x*blockDim.x; // <- my thread index
	w[i].setValue(i+param);
}

int main(int argc,char *argv[]) 
{
// Allocate space shared between CPU and GPU
	int n=16; // total number of floats
	widget *w=new widget[n]; // shared array of n values (overloaded new[])

// Run "GPU kernel" on shared space
	int nBlocks=1; // GPU thread blocks to run
	int blockDim=n; // threads/block, should be 256 for best performance
	set_array<<<nBlocks,blockDim>>>(w,0.1234); /* run kernel on GPU */ 

	cudaDeviceSynchronize(); /* Wait for kernel to finish filling vals array */

// Show results
	int i=7;
	std::cout<<"widget["<<i<<"] = "<<w[i].value<<"\n";
        return 0;

}
