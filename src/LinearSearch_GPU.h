/*
 * LinearSearch_GPU.h
 *
 *  Created on: Jan 5, 2017
 *      Author: mrp
 */

#ifndef LINEARSEARCH_GPU_H_
#define LINEARSEARCH_GPU_H_

#include "common.h"

__device__ uint8_t full_match(tuple_t *tup, rule_t *rule) {

	/*
	 * Note: In exact match fields if mask == 1 then we should check its value.
	 */

	int i;

	/*
	 * ingress port --> exact match
	 */
	if (rule->mask.ingress_ports_mask)
		if (tup->value.ingress_port != rule->value.ingress_port)
			return 0;

	/*
	 * metadata --> exact match
	 */
	if (rule->mask.metadata_mask)
		if (tup->value.metadata != rule->value.metadata)
			return 0;

	/*
	 * src mac --> exact match
	 */
	if (rule->mask.eth_src_mask)
		for (i = 0; i < 6; i++) {
			if (tup->value.eth_src.bytes[i] != rule->value.eth_src.bytes[i])
				return 0;
		}

	/*
	 * dst mac --> exact match
	 */
	if (rule->mask.eth_dst_mask)
		for (i = 0; i < 6; i++) {
			if (tup->value.eth_dst.bytes[i] != rule->value.eth_dst.bytes[i])
				return 0;
		}

	/*
	 * ethertype --> exact match
	 */
	if (rule->mask.ether_type_mask)
		if (tup->value.ether_type != rule->value.ether_type)
			return 0;

	/*
	 * vlan id --> exact match
	 */
	if (rule->mask.vid_mask)
		if (tup->value.vid != rule->value.vid)
			return 0;

	/*
	 * vlan priority --> exact match
	 */
	if (rule->mask.vprty_mask)
		if (tup->value.vprty != rule->value.vprty)
			return 0;

	/*
	 * tos --> exact match
	 */
	if (rule->mask.tos_mask)
		if (tup->value.tos != rule->value.tos)
			return 0;

	/*
	 * mpls lable --> exact match
	 */
	if (rule->mask.mpls_lbl_mask)
		if (tup->value.mpls_lbl != rule->value.mpls_lbl)
			return 0;

	/*
	 * mpls traffic class --> exact match
	 */
	if (rule->mask.mpls_tfc_mask)
		if (tup->value.mpls_tfc != rule->value.mpls_tfc)
			return 0;

	/*
	 * src ip --> prefix match
	 */
	if ((tup->value.ip_src & rule->mask.ip_src_mask) != rule->value.ip_src)
		return 0;

	/*
	 * dst ip --> prefix match
	 */
	if ((tup->value.ip_dst & rule->mask.ip_dst_mask) != rule->value.ip_dst)
		return 0;

	/*
	 * protocol  --> exact match
	 */
	if (rule->mask.proto_mask)
		if (tup->value.proto != rule->value.proto)
			return 0;

	/*
	 * src port --> range match
	 */
	if (tup->value.port_src < rule->value.port_src.lower_bound
			|| tup->value.port_src > rule->value.port_src.upper_bound)
		return 0;

	/*
	 * dst port --> range match
	 */
	if (tup->value.port_dst < rule->value.port_dst.lower_bound
			|| tup->value.port_dst > rule->value.port_dst.upper_bound)
		return 0;

	/*
	 * full match occurred :)
	 */
	return 1;
}

__global__ void LinearSearch_Kernel(rule_t *rules, tuple_t *tuples,
		int *results, int nb_rules, int nb_tup) {

	/*
	 * Auxiliary Variables
	 */
	int r, t, rIdx, rPri;
	uint8_t match;

	/*
	 * thread_tuple --> tuple for this thread
	 * block_rules	--> rules that should be considered in this block and are shared between threads
	 */
	tuple_t thread_tuple;
	__shared__ rule_t block_rules[RULES_PER_BLOCK];

	int rule_offset = blockIdx.x * RULES_PER_BLOCK;
	while (rule_offset < nb_rules) {

		/*
		 * for the last iteration
		 */
		int nb_rules_per_block = min(RULES_PER_BLOCK, nb_rules - rule_offset);

		/*
		 * Threads copy nb_rules_per_block rule to block shared memory
		 */
		t = threadIdx.x;
		while (t < nb_rules_per_block) {
			block_rules[t] = rules[rule_offset + t];
			t += blockDim.x;
		}
		__syncthreads();

		/*
		 * Each thread will match some tuples with rules according to its threadIdx.x
		 */
		t = threadIdx.x;
		while (t < nb_tup) {

			/*
			 * get tuple t
			 */
			thread_tuple = tuples[t];

			rIdx = -1;
			rPri = -1;
			for (r = 0; r < nb_rules_per_block; r++) {
				match = full_match(&thread_tuple, &block_rules[r]);
				if (match == 1) {
					if (rPri < block_rules[r].pri) {
						rPri = block_rules[r].pri;
						rIdx = block_rules[r].id;
					}
				}
			}
			if (rIdx >= 0) {
				atomicMax(&results[t], (rPri << 24) | rIdx);
			}
			t += blockDim.x;
		}

		__syncthreads();
		rule_offset += (gridDim.x * RULES_PER_BLOCK);
	}

}

int linearSearch_GPU(void) {

	printf("\n\nSearching %d Tuple(s) in %d Rule(s) on GPU ...\n", nb_tuples,
			nb_rules);
	sleep(1);

	int i;
	int *results_host;
	tuple_t *tuples_host;

	rule_t *rules_dev;
	tuple_t *tuples_dev;
	int *results_dev;

	cudaEvent_t start, stop;
	cudaStream_t stream;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaStreamCreate(&stream);

	/*
	 * Allocate memory on host
	 * This type of memory allocation is used to support asynchronous memory copy
	 */
	CUDA_CHECK_RETURN(
			cudaMallocHost((void **) &results_host, nb_results * sizeof(int),
					cudaHostAllocDefault));
	for (i = 0; i < nb_results; i++) {
		results_host[i] = -1; // LOOKUP_MISS
	}

	CUDA_CHECK_RETURN(
			cudaMallocHost((void **) &tuples_host, nb_tuples * sizeof(tuple_t),
					cudaHostAllocDefault));
	for (i = 0; i < nb_tuples; i++) {
//		tuples_host[i].id = tuples[i].id;
//		tuples_host[i].rid = tuples[i].rid;
//		tuples_host[i].value = tuples[i].value;
		tuples_host[i] = tuples[i];
	}

	/*
	 * allocate memory on device
	 */
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &rules_dev, nb_rules * sizeof(rule_t)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &tuples_dev, BATCH_SIZE * sizeof(tuple_t)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &results_dev, nb_results * sizeof(int)));

	/*
	 * copy rules to device and reset results
	 */
	CUDA_CHECK_RETURN(
			cudaMemcpy(rules_dev, &rules[0], nb_rules * sizeof(rule_t),
					cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(
			cudaMemcpy(results_dev, results_host, nb_results * sizeof(int),
					cudaMemcpyHostToDevice));

	sleep(1);

	/*
	 * set time stamp t1
	 */
	cudaEventRecord(start, stream);

	for (i = 0; i < NUM_ITER; i++) {

		int batch_size = min(BATCH_SIZE, nb_tuples - (BATCH_SIZE * i));

		/*
		 *  copy tuples to device
		 */
		CUDA_CHECK_RETURN(
				cudaMemcpyAsync(tuples_dev, &tuples_host[i * BATCH_SIZE],
						batch_size * sizeof(tuple_t), cudaMemcpyHostToDevice,
						stream));

		/*
		 * Launch Kernel Function
		 */
		LinearSearch_Kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, 0, stream>>>(
				rules_dev, tuples_dev, &results_dev[i * BATCH_SIZE], nb_rules,
				batch_size);

		/*
		 * copy results to host
		 */
		CUDA_CHECK_RETURN(
				cudaMemcpyAsync(&results_host[i * BATCH_SIZE],
						&results_dev[i * BATCH_SIZE], batch_size * sizeof(int),
						cudaMemcpyDeviceToHost, stream));
	}

	cudaStreamSynchronize(stream);

	/*
	 * set time stamp t2
	 */
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	/*
	 * show search results
	 */
#ifdef SHOW_SEARCH_RESULTS
	for (i = 0; i < nb_tuples; i++) {
		printf("tuple %d with rid = %d matches with rule %d\n", tuples[i].id,
				tuples[i].rid, results_host[i] & 0xFFFFFF);
		if (tuples[i].rid != (results_host[i] & 0xFFFFFF)) {
			printf("error\n");
			getchar();
		}
	}
#endif

	/*
	 * free host & device memory
	 */
	CUDA_CHECK_RETURN(cudaFreeHost(results_host));
	CUDA_CHECK_RETURN(cudaFreeHost(tuples_host));
	CUDA_CHECK_RETURN(cudaFree(rules_dev));
	CUDA_CHECK_RETURN(cudaFree(tuples_dev));
	CUDA_CHECK_RETURN(cudaFree(results_dev));

	printf("Done. Linear Search takes %f ms on GPU, per Batch = %f ms\n\n",
			elapsedTime, elapsedTime / NUM_ITER);

	return 0;

}

#endif /* LINEARSEARCH_GPU_H_ */
