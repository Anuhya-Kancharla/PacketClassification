/*
 ============================================================================
 Name        : PacketClassification.cu
 Author      : MRP
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#include "common.h"
#include "args.h"
#include "init.h"
#include "LinearSearch_GPU.h"

int main(int argc, char **argv) {

	/*
	 *	parse user's input arguments
	 */
	parse_args(argc, argv);

	/*
	 * Initialization
	 */
	init();

	/*
	 * Linear search on GPU
	 */
	linearSearch_GPU();

	/*
	 * free allocated memories
	 */
	free_mem();

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

