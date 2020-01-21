#include <iostream>
#include <cuda.h>
#include <cstdlib>



class Unified {
public:
  void *operator new(size_t len) {
	void *ptr;
	cudaMallocManaged(&ptr, len);
	return ptr;
  }
  void operator delete(void *ptr) {
	cudaFree(ptr);
  }


  void *operator new[] (std::size_t size) {
	void *ptr; 
	cudaMallocManaged(&ptr,size);
	return ptr;
  }
  void operator delete[] (void* ptr) {
	cudaFree(ptr);
  }
};


class publisher : public Unified 
{
public:
	float value;
	__device__ void setValue(float v) { value=v; }
};

__global__ void publish_msg(publisher *topic,float num) {
	int i=threadIdx.x + blockIdx.x*blockDim.x; 
	topic[i].setValue(i+num);
}



/* GPU kernel: set an array of topic to a value */
__host__ void sub_msg(publisher *topic,int i, int s) {
      
      std::cout<<"subscriber "<< s <<": Topic["<<i<<"] = "<<topic[i].value<<"\n";
}


int main(int argc,char *argv[]) 
{

        int t=0,n=20;
	int s=0;//subscriber number
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
	
        publisher *topic = new publisher[n];
	
        publish_msg<<<1,n>>>(topic,0.1543); //n=20 is size of topic array
       	
	cudaDeviceSynchronize();

	s=1,t=0; //subscriber s and topic number t
	
	sub_msg(topic,t,s);
	cudaEventRecord(stop);  
        cudaEventSynchronize(stop);
        float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout<<"Elapsed time = "<<milliseconds<<" milliseconds\n";
	return 0;

}
