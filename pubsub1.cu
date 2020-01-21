#include <iostream>
#include <cuda.h>


/* Publish topic GPU code: set an topic array to a value */

__global__ void pub_topic(float *topic,float param) {
	int i=threadIdx.x;                                                  /* find my index */
	topic[i]=i+param;
}

/* Subscribe topic CPU code: get antopic value */
void sub_topic(float *topic, int i) {
 
        /* Copy elements back to CPU for subscriber */
        int s=1;
	i=0;
	float f=0.0; /* CPU copy of value */
        cudaMemcpy(&f,&topic[i],sizeof(float),cudaMemcpyDeviceToHost);
        std::cout<<"subscriber "<< s <<":topic["<<i<<"] = "<<f<<"\n";


}

/* CPU code: memory movement and kernel calls */

int main(int argc,char *argv[]) {
	int i=0;
	int n=20;                                      /* total number of floats */
	float *topic;                                  /* device array of n values */

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	cudaMalloc( (void**) &topic, n*sizeof(float) );     //Allocate GPU space
	
        pub_topic<<<1,n>>>(topic,0.1543);                /* Initialize the space on the GPU */
        sub_topic(topic,i);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
      
        std::cout<<"Elapsed time = "<<milliseconds<<" milliseconds\n";
        return 0;
}
