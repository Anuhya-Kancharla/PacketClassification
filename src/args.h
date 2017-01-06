/*
 * args.h
 *
 *  Created on: Jan 5, 2017
 *      Author: mrp
 */

#ifndef ARGS_H_
#define ARGS_H_

#include <getopt.h>
#include "common.h"

void usage(char *filename) {
	printf("Usage: %s -f ruleset -n nb_traces [-h]\n", filename);
	exit (EXIT_FAILURE);
}

void parse_args(int argc, char **argv) {
	int c;
	opterr = 0;
	while ((c = getopt(argc, argv, "f:n:h")) != -1) {
		switch (c) {
		case 'f':
			ruleset_file = strdup(optarg);
			break;
		case 'n':
			nb_tuples = atoi(strdup(optarg));
			break;
		case '?':
		case 'h':
			usage(argv[0]);
			break;
		default:
			break;
		}
	}
	if (ruleset_file == NULL) {
		printf("please specify ruleset file name\n");
		usage(argv[0]);
	}
	if (access(ruleset_file, F_OK) != 0) {
		printf("%s does not exist\n", ruleset_file);
		exit (EXIT_FAILURE);
	}
	if (access(ruleset_file, R_OK) != 0) {
		printf("can not read from %s\n", ruleset_file);
		exit (EXIT_FAILURE);
	}
	if (nb_tuples == 0) {
		printf("please specify nb_traces\n");
		usage(argv[0]);
	}

}

#endif /* ARGS_H_ */
