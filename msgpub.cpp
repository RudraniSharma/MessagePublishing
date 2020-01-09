
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


  void* operator new[] (std::size_t size) {
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
    topic[i].setValue(num);
}

class subscriber : public Unified
{
public:
    float value;
    __device__ void getValue(float v) { value=v; }
};

/* GPU kernel: set an array of topic to a value */
__host__ void sub_msg(publisher *topic,int i) {
      std::cout<<"Topic["<<i<<"] = "<<topic[i].value<<"\n";
}


int main(int argc,char *argv[])
{

        int n=1;
        int i=0;
        publisher *topic=new publisher[n];
    worker<<<1,1>>>(topic);
    //subscribe(topic_out);
    publish_msg(topic, 6.9);  // gpu kernel recv msg, replies to topic
    m = recv(topic);  // blocking
    // m== reply from gpu
    
        publish_msg<<<1,1>>>(topic,6.9); /* GPU */
    //std::cout<<"Topic["<<i<<"] = "<<topic[i].value<<"\n";
    cudaDeviceSynchronize();
    sub_msg(topic,i);

       

    i++;
    publisher *topic1=new publisher[n];
    publish_msg<<<1,2>>>(topic1,7.7);
    //std::cout<<"Topic["<<i<<"] = "<<topic1[i].value<<"\n";
    cudaDeviceSynchronize();
           sub_msg(topic1,i);

    return 0;

}
