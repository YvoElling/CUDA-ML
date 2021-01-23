#define ERROR_DIMENSIONS 3

__global__ void simple_linear_regression(float* d_x, float* d_y, float* d_bias, float* d_intercept, float* d_results, uint8_t in_size) {
	// Define shared memory in thread block where we maintian the collective sum of squared errors
	__shared__ float errors[ERROR_DIMENSIONS];

	// Retrieve thread index to use a array index
	int index = threadIdx.x;

	// Calculate y_predicted based on current bias and intercept
	float y_pred = *d_bias + *d_intercept * d_x[index];

	// Calculate J for this specific index and store in errors index 0
	float j = 0.5f * pow((d_y[index] - y_pred), 2);
	errors[0] += atomicAdd(&errors[0], j);

	// Calculate bias error for this index and store in errors index
	errors[1] += atomicAdd(&errors[1], - (d_y[index] - y_pred));

	// Calculate intercept error for this index
	float intercept_err = -(d_y[index] - y_pred)*d_x[index];
	errors[2] += atomicAdd(&errors[2], intercept_err);

	// Wait until threads are synchronized before returning the shared memory data to host memory.
	__syncthreads();

	// Update the output values
	if (threadIdx.x == in_size - 1) {
		// Write to host memory
		d_results[0] = errors[0];
		d_results[1] = errors[1];
		d_results[2] = errors[2];

		// Reset shared memory for the next iteration of the algorithm
		errors[0] = 0;
		errors[1] = 0;
		errors[2] = 0;
	}
}
