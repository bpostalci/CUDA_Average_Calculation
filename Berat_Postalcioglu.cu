// Berat Postalcioglu
/* OUTPUT
	Enter number of children: 5
	15
	3
	14
	13
	12
	cpu average: 11.4
	gpu average: 11.4
	Average is calculated correctly for 5 children as 11.4
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>   
#include <ctime>  
#include <iostream>
#include <cmath>

using namespace std;

void init_arr(int** arr, int size)
{

	for (int i = 0; i < size; i++)
	{
		*(*arr + i) = rand() % 16 + 1;
		cout << *(*arr + i) << endl;
	}
}

double calculate_average_oncpu(int** arr, int size)
{
	int total = 0;
	for (int i = 0; i < size; i++)
	{
		total += *(*arr + i);
	}
	return (double)total / size;
}

__global__ void calculate_average_ongpu(int *arr, int *size, double *res)
{
	int total = 0;
	for (int i = 0; i < *size; i++)
	{
		total += arr[i];
	}
	*res = ((double)total / *size);
}

int main()
{
	srand(time(NULL));

	int *arr;
	int size;
	cout << "Enter number of children: ";
	cin >> size;
	arr = new int[size];
	init_arr(&arr, size);
	double cpu_av = calculate_average_oncpu(&arr, size);
	cout << "cpu average: " << cpu_av << endl;

	// cuda
	int *gpu_arr, *gpu_size;
	double *gpu_av;
	cudaMalloc((void**)&gpu_arr, size * sizeof(int));
	cudaMemcpy((void*)gpu_arr, (const void*)arr, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&gpu_size, sizeof(int));
	cudaMemcpy((void*)gpu_size, (const void*)&size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&gpu_av, sizeof(double));
	calculate_average_ongpu<<<1, 1>>> (gpu_arr, gpu_size, gpu_av);
	
	double *_gpu_av = new double;
	cudaMemcpy((void*)_gpu_av, (const void*)gpu_av, sizeof(double), cudaMemcpyDeviceToHost);
	cout << "gpu average: " << *_gpu_av << endl;

	if (*_gpu_av == cpu_av)
	{
		cout << "Average is calculated correctly for " << size << " children as " << cpu_av << endl;
	}
}
