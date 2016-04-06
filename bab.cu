// SM 3.0 or greater GPUS only!
// compile with: nvcc bab.cu -o bab -arch=sm_35 -rdc=true
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <curand_kernel.h>
#include <vector>
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
        exit(code);
    }
}

__global__
void compute_new_intervals(float * stack, uint * elems_on_stack, 
		uint * num_solutions_on_stack, uint stack_capacity, uint NUM_DIMS, uint batch_size) {
	uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	// One thread per elem on stack
	if(thread_index >= batch_size)
		return;	
	// Initialize random number generation
	curandState s;
	curand_init (thread_index , 0, 0, &s);
	float prn = curand_uniform(&s);
	//float interval1 = 


}

// SM 3.0 > devices only
__global__
void branch_and_bound(float * stack, uint * num_candidates_on_stack, uint * num_solutions_on_stack,
	 uint stack_capacity, uint NUM_DIMS, uint num_SMs)
{
	uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	// Branch and bound controlled by one thread
	if(thread_index != 0)
		return;
	// Get device ID for debugging purposes
	int device_id;
	cudaGetDevice(&device_id);
	// Set launch bounds to maximize hardware occupancy
	uint MAX_BLOCK_SIZE = 1024;
	uint BLOCKS_PER_SM = 2;
	uint MAX_THREADS_PER_SM = MAX_BLOCK_SIZE * BLOCKS_PER_SM;
	uint MAX_BATCH_SIZE = num_SMs * MAX_THREADS_PER_SM;
	uint NUM_BLOCKS = BLOCKS_PER_SM * num_SMs;
	// Init stack params
	*num_solutions_on_stack = 0;
	*num_candidates_on_stack = 1;
	// Continue until a solution is found
	while(*num_solutions_on_stack == 0) {
		uint batch_size = *num_candidates_on_stack < MAX_BATCH_SIZE ? *num_candidates_on_stack : MAX_BATCH_SIZE;
		printf("Device ID: %d, num_candidates_on_stack: %d\n", device_id, *num_candidates_on_stack);
		compute_new_intervals<<<NUM_BLOCKS, MAX_BLOCK_SIZE>>>
			(stack, num_candidates_on_stack, num_solutions_on_stack, stack_capacity, NUM_DIMS, batch_size);
	}

}

int main(int argc, char **argv) {
	uint NUM_DIMS;
	if(argc != 2) {
		printf("./bab <number_of_dims>\n");
		return -1;
	}
	NUM_DIMS = (uint) atoi(argv[1]);
	printf("Num dims: %d\n", NUM_DIMS);
	time_t initial, final;
	int num_devices;
    cudaDeviceProp prop;
	gpuErrchk( cudaGetDeviceCount(&num_devices) );
	printf("Num devices: %d\n", num_devices);


	// Host-Side allocations
	// Make up a search space
	float ** search_space = new float * [NUM_DIMS];
	for(int i = 0; i < NUM_DIMS; ++i) {
		search_space[i] = new float[2]; //min, max
	}
	// Create array of handles to device stacks
	float ** dev_stacks = new float * [num_devices];
	// Create array of handles to device num_candidates
	uint ** dev_num_candidates = new uint * [num_devices];
	// Create array of handles to device num_solutions
	uint ** dev_num_solutions = new uint * [num_devices];
	// Divy up the workspace. Encode search space dim1_min, dim1_max ... dimn_min, dimn_max
	float ** initial_search_spaces = new float * [num_devices];
	for(int i = 0; i < num_devices; ++i) {
		initial_search_spaces[i] = new float[NUM_DIMS * 2];
	}
	uint size_of_an_interval = 2 * NUM_DIMS * sizeof(float);

	// Populate search space
	for(int i = 0; i < NUM_DIMS; i++) {
		search_space[i][0] = -10.0; // min
		search_space[i][1] = 10.0; // max
	}

	// First N - 1 dimensions are the same as the initial searchspace dims
	for(int i = 0; i < num_devices; ++i) {
		for(int j = 0; j < NUM_DIMS - 1; ++j) {
			initial_search_spaces[i][2 * j] = search_space[j][0];
			initial_search_spaces[i][2 * j + 1] = search_space[j][1];
		}
		// Split the last dimension evenly
		float last_dim_min = search_space[NUM_DIMS - 1][0];
		float last_dim_max = search_space[NUM_DIMS - 1][1];
		float stride = (last_dim_max - last_dim_min) / num_devices;
		float this_lower_bound = i * stride + last_dim_min;
		initial_search_spaces[i][2 * NUM_DIMS - 2] = this_lower_bound;
		initial_search_spaces[i][2 * NUM_DIMS - 1] = std::min(this_lower_bound + stride, last_dim_max); 
	}

	// Store the stack capacities/sizes of every GPU
	std::vector<uint> stack_capacities(num_devices);
	std::vector<uint> stack_sizes(num_devices);

	// Launch kernels on each GPU
	initial = clock();
	for(int i = 0; i < num_devices; ++i) {
		// Select GPU
		cudaSetDevice(i);
		// Read in available memory on this GPU
		size_t free_memory, total_memory;
		gpuErrchk( cudaMemGetInfo(&free_memory, &total_memory) );
		// Determine how big we can make the stack
		uint stack_capacity = (uint) floor(free_memory / (2 * sizeof(float) * NUM_DIMS) - size_of_an_interval);
		stack_capacity *= .99;
		stack_capacities[i] = stack_capacity;
		stack_sizes[i] = stack_capacity * 2 * NUM_DIMS * sizeof(float);
		// Malloc and copy over stack and ints representing num candidates and num solutions
		gpuErrchk( cudaMalloc((void **) &dev_stacks[i], stack_sizes[i]) );
		gpuErrchk( cudaMemcpy(dev_stacks[i], initial_search_spaces[i], size_of_an_interval, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMalloc((void **) &dev_num_candidates[i], sizeof(uint)) );
		gpuErrchk( cudaMalloc((void **) &dev_num_solutions[i], sizeof(uint)) );
		
		cudaGetDeviceProperties(&prop, i);
		branch_and_bound<<<1,1>>>(dev_stacks[i], dev_num_candidates[i], dev_num_solutions[i],
		 	stack_capacities[i], NUM_DIMS, prop.multiProcessorCount);

        // Check for errors on kernel call
        cudaError err = cudaGetLastError();
        if (cudaSuccess != err)
            printf("Error %s\n",cudaGetErrorString(err));
	}
	// Copy back
	float * host_arr = new float[ *std::max_element(stack_sizes.begin(), stack_sizes.end()) ];
	for(int i = 0; i < num_devices; ++i) {
 		gpuErrchk( cudaMemcpy(host_arr, dev_stacks[i], stack_sizes[i], cudaMemcpyDeviceToHost) );
 		// Read back
  		for(int j = 0; j < 3; ++j) {
 			printf("device: %d - host arr at index %d: %f\n", i, j, host_arr[j]);
 		}
 		// Cleanup device
		cudaFree( dev_stacks[i] );
	}
	final = clock();
	printf("Total time: %f (s) \n", (final - initial) / 1000000.0 );
	// cleanup host
	delete [] host_arr;
	delete [] dev_stacks;
	delete [] dev_num_candidates;
	delete [] dev_num_solutions;

	for(int i = 0; i < num_devices; ++i) {
		delete [] initial_search_spaces[i];
	}
	for(int i = 0; i < NUM_DIMS; ++i) {
		delete [] search_space[i];
	}
	delete initial_search_spaces;
	delete search_space;
}
