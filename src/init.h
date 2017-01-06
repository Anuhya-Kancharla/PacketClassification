/*
 * init.h
 *
 *  Created on: Jan 5, 2017
 *      Author: mrp
 */

#ifndef INIT_H_
#define INIT_H_

#include "common.h"

void dump_rule(rule_t rule) {
	char *format = "rule%d:\n"
			"0x%x/%d\t" // ingress port
			"0x%lx/%d\t"// meta data
			"%hhx:%hhx:%hhx:%hhx:%hhx:%hhx/%d\t"// src mac
			"%hhx:%hhx:%hhx:%hhx:%hhx:%hhx/%d\t"// dst mac
			"0x%hx/%d\t"// ether type
			"0x%hx/%d\t"// vid
			"0x%hhx/%d\t"// vprty
			"0x%hhx/%d\t"// tos
			"0x%x/%d\t"// mpls lbl
			"0x%hhx/%d\t"// mpls tfc
			"0x%x/%x\t"// src ip
			"0x%x/%x\t"// dst ip
			"%hhx/%d\t"// proto
			"%hu:%hu\t"// src port
			"%hu:%hu\n";// dst port

	printf(format,
			rule.id, //
			rule.value.ingress_port, rule.mask.ingress_ports_mask,
			rule.value.metadata, rule.mask.metadata_mask, /***/
			rule.value.eth_src.bytes[0], rule.value.eth_src.bytes[1],
			rule.value.eth_src.bytes[2], rule.value.eth_src.bytes[3],
			rule.value.eth_src.bytes[4], rule.value.eth_src.bytes[5],
			rule.mask.eth_src_mask, /***/
			rule.value.eth_dst.bytes[0], rule.value.eth_dst.bytes[1],
			rule.value.eth_dst.bytes[2], rule.value.eth_dst.bytes[3],
			rule.value.eth_dst.bytes[4], rule.value.eth_dst.bytes[5],
			rule.mask.eth_dst_mask, /***/
			rule.value.ether_type, rule.mask.ether_type_mask, /***/
			rule.value.vid, rule.mask.vid_mask, /***/
			rule.value.vprty, rule.mask.vprty_mask, /***/
			rule.value.tos, rule.mask.tos_mask, /***/
			rule.value.mpls_lbl, rule.mask.mpls_lbl_mask, /***/
			rule.value.mpls_tfc, rule.mask.mpls_tfc_mask, /***/
			rule.value.ip_src, rule.mask.ip_src_mask, /***/
			rule.value.ip_dst, rule.mask.ip_dst_mask, /***/
			rule.value.proto, rule.mask.proto_mask, /***/
			rule.value.port_src.lower_bound, rule.value.port_src.upper_bound, /***/
			rule.value.port_dst.lower_bound, rule.value.port_dst.upper_bound); /***/
	printf("------------------------------------------------------------\n");
}

void load_rule(rule_t *rule, char *input_rule) {
	char *format = ""
			"0x%08x/%d\t" // ingress port
			"0x%lx/%d\t"// meta data
			"%hhx:%hhx:%hhx:%hhx:%hhx:%hhx/%d\t"// src mac
			"%hhx:%hhx:%hhx:%hhx:%hhx:%hhx/%d\t"// dst mac
			"0x%hx/%d\t"// ether type
			"0x%hx/%d\t"// vid
			"0x%hhx/%d\t"// vprty
			"0x%hhx/%d\t"// tos
			"0x%x/%d\t"// mpls lbl
			"0x%hhx/%d\t"// mpls tfc
			"%hhu.%hhu.%hhu.%hhu/%x\t"// src ip
			"%hhu.%hhu.%hhu.%hhu/%x\t"// dst ip
			"%hhx/%d\t"// proto
			"%hu:%hu\t"// src port
			"%hu:%hu\n";// dst port
	ip_t ip_src, ip_dst;
	sscanf(input_rule,
			format, //
			&rule->value.ingress_port, &rule->mask.ingress_ports_mask, /***/
			&rule->value.metadata, &rule->mask.metadata_mask, /***/
			&rule->value.eth_src.bytes[0], &rule->value.eth_src.bytes[1],
			&rule->value.eth_src.bytes[2], &rule->value.eth_src.bytes[3],
			&rule->value.eth_src.bytes[4], &rule->value.eth_src.bytes[5],
			&rule->mask.eth_src_mask, /***/
			&rule->value.eth_dst.bytes[0], &rule->value.eth_dst.bytes[1],
			&rule->value.eth_dst.bytes[2], &rule->value.eth_dst.bytes[3],
			&rule->value.eth_dst.bytes[4], &rule->value.eth_dst.bytes[5],
			&rule->mask.eth_dst_mask, /***/
			&rule->value.ether_type, &rule->mask.ether_type_mask, /***/
			&rule->value.vid, &rule->mask.vid_mask, /***/
			&rule->value.vprty, &rule->mask.vprty_mask, /***/
			&rule->value.tos, &rule->mask.tos_mask, /***/
			&rule->value.mpls_lbl, &rule->mask.mpls_lbl_mask, /***/
			&rule->value.mpls_tfc, &rule->mask.mpls_tfc_mask, /***/
			&ip_src.bytes[0], &ip_src.bytes[1], &ip_src.bytes[2],
			&ip_src.bytes[3], &rule->mask.ip_src_mask, /***/
			&ip_dst.bytes[0], &ip_dst.bytes[1], &ip_dst.bytes[2],
			&ip_dst.bytes[3], &rule->mask.ip_dst_mask, /***/
			&rule->value.proto, &rule->mask.proto_mask, /***/
			&rule->value.port_src.lower_bound,
			&rule->value.port_src.upper_bound, /***/
			&rule->value.port_dst.lower_bound,
			&rule->value.port_dst.upper_bound); /***/

	rule->value.ip_src = (ip_src.bytes[0] << 24) | (ip_src.bytes[1] << 16)
			| (ip_src.bytes[2] << 8) | ip_src.bytes[3];

	rule->value.ip_dst = (ip_dst.bytes[0] << 24) | (ip_dst.bytes[1] << 16)
			| (ip_dst.bytes[2] << 8) | ip_dst.bytes[3];

}

void dump_tuple(tuple_t tup) {
	char *format = "tup%d --> rule%d:\n"
			"0x%x\t" // ingress port
			"0x%lx\t"// meta data
			"%hhx:%hhx:%hhx:%hhx:%hhx:%hhx\t"// src mac
			"%hhx:%hhx:%hhx:%hhx:%hhx:%hhx\t"// dst mac
			"0x%hx\t"// ether type
			"0x%hx\t"// vid
			"0x%hhx\t"// vprty
			"0x%hhx\t"// tos
			"0x%x\t"// mpls lbl
			"0x%hhx\t"// mpls tfc
			"0x%x\t"// src ip
			"0x%x\t"// dst ip
			"%hhx\t"// proto
			"%hu\t"// src port
			"%hu\n";// dst port

	printf(format, tup.id, tup.rid, /***/
	tup.value.ingress_port, /***/
	tup.value.metadata, /***/
	tup.value.eth_src.bytes[0], tup.value.eth_src.bytes[1], /***/
	tup.value.eth_src.bytes[2], tup.value.eth_src.bytes[3], /***/
	tup.value.eth_src.bytes[4], tup.value.eth_src.bytes[5], /***/
	tup.value.eth_dst.bytes[0], tup.value.eth_dst.bytes[1], /***/
	tup.value.eth_dst.bytes[2], tup.value.eth_dst.bytes[3], /***/
	tup.value.eth_dst.bytes[4], tup.value.eth_dst.bytes[5], /***/
	tup.value.ether_type, /***/
	tup.value.vid, /***/
	tup.value.vprty, /***/
	tup.value.tos, /***/
	tup.value.mpls_lbl, /***/
	tup.value.mpls_tfc, /***/
	tup.value.ip_src, /***/
	tup.value.ip_dst, /***/
	tup.value.proto, /***/
	tup.value.port_src, /***/
	tup.value.port_dst); /***/
	printf("------------------------------------------------------------\n");
}

void generate_tuple(tuple_t *tup, int id) {
	int i;
	int random_rid;
	random_rid = rand() % nb_rules;

	tup->id = id;
	tup->rid = random_rid;
	tup->value.ingress_port = rules[random_rid].value.ingress_port;
	tup->value.metadata = rules[random_rid].value.metadata;
	for (i = 0; i < 6; i++) {
		tup->value.eth_src.bytes[i] = rules[random_rid].value.eth_src.bytes[i];
		tup->value.eth_dst.bytes[i] = rules[random_rid].value.eth_dst.bytes[i];
	}
	tup->value.ether_type = rules[random_rid].value.ether_type;
	tup->value.vid = rules[random_rid].value.vid;
	tup->value.vprty = rules[random_rid].value.vprty;
	tup->value.tos = rules[random_rid].value.tos;
	tup->value.mpls_lbl = rules[random_rid].value.mpls_lbl;
	tup->value.mpls_tfc = rules[random_rid].value.mpls_tfc;
	tup->value.ip_src = rules[random_rid].value.ip_src;
	tup->value.ip_dst = rules[random_rid].value.ip_dst;
	tup->value.proto = rules[random_rid].value.proto;
	tup->value.port_src = rules[random_rid].value.port_src.lower_bound;	// This could be better
	tup->value.port_dst = rules[random_rid].value.port_dst.upper_bound; // This could be better
}

void init(void) {
	FILE * fp;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;
	char ch;
	int nb_lines = 0;
	int i = 0, j = 0;

	/*
	 * Initialize rand()
	 */
	srand (time(NULL));

	/*
	 * get the number of rules in rule set file
	 */
fp	= fopen(ruleset_file, "r");
	nb_lines = 0;
	while (!feof(fp)) {
		ch = fgetc(fp);
		if (ch == '\n') {
			nb_lines++;
		}
	}
	nb_rules = nb_lines;
	fclose(fp);
	printf("NB_RULES : %d\n", nb_rules);

	/*
	 * load rules from file
	 */
	fp = fopen(ruleset_file, "r");
	rules = (rule_t *) malloc(nb_rules * sizeof(rule_t));
	i = 0;
	while ((read = getline(&line, &len, fp)) != -1) {
		rules[i].id = i;
		rules[i].pri = 10;
		load_rule(&rules[i], line);
#ifdef DUMP_RULES
		dump_rule(rules[i]);
#endif
		i++;
	}
	if (line != NULL)
		free(line);

	/*
	 * Create MC-SBC data structures
	 */

	/*
	 * Generate 15-field traces
	 */
	tuples = (tuple_t *) malloc(nb_tuples * sizeof(tuple_t));
	for (i = 0; i < nb_tuples; i++) {
		generate_tuple(&tuples[i], i);
#ifdef DUMP_TUPES
		dump_tuple(tuples[i]);
#endif
	}

	/*
	 * Initialize results
	 */
	results = (int *) malloc(nb_tuples * sizeof(int));
	for (i = 0; i < nb_results; i++) {
		results[i] = -1; // lookup miss
	}
}

void free_mem(void) {
	free(rules);
	free(tuples);
	free(results);
}

#endif /* INIT_H_ */
