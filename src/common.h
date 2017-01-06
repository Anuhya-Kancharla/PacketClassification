/*
 * common.h
 *
 *  Created on: Jan 5, 2017
 *      Author: mrp
 */

#ifndef COMMON_H_
#define COMMON_H_

/*
 * Debug Options
 */
//#define DUMP_RULES
//#define DUMP_TUPES
//#define SHOW_SEARCH_RESULTS

/*
 * GPU Parameters
 */
#define BATCH_SIZE	32
#define NUM_ITER	(((nb_tuples - 1)/BATCH_SIZE) + 1)

#define BLOCKS_PER_GRID		32
#define THREADS_PER_BLOCK	32
#define RULES_PER_BLOCK		32

/*
 * Rule Parameters
 */
#define RULE_LEN			356
#define INGRESS_PORT_LEN    32
#define METADATA_LEN        64
#define ETH_SRC_LEN         48
#define ETH_DST_LEN         48
#define ETHER_TYPE_LEN      16
#define VID_LEN             12
#define VPRTY_LEN           3
#define TOS_LEN             6
#define MPLS_LBL_LEN        20
#define MPLS_TFC_LEN        3
#define IP_SRC_LEN          32
#define IP_DST_LEN          32
#define PROTO_LEN           8
#define PORT_SRC_LEN        16
#define PORT_DST_LEN        16

/*
 * Definition of Structures
 */
typedef struct ip {
	uint8_t bytes[4];
} ip_t;

typedef struct mac {
	uint8_t bytes[6];
} mac_t;

typedef struct port {
	uint16_t lower_bound;
	uint16_t upper_bound;
} port_t;

typedef struct rule_value {
	uint32_t ingress_port;
	uint64_t metadata;
	mac_t eth_src;
	mac_t eth_dst;
	uint16_t ether_type;
	uint16_t vid;
	uint8_t vprty;
	uint8_t tos;
	uint32_t mpls_lbl;
	uint8_t mpls_tfc;
	uint32_t ip_src;
	uint32_t ip_dst;
	uint8_t proto;
	port_t port_src;
	port_t port_dst;
} rvalue_t;

typedef struct rule_mask {
	uint8_t ingress_ports_mask;
	uint8_t metadata_mask;
	uint8_t eth_src_mask;
	uint8_t eth_dst_mask;
	uint8_t ether_type_mask;
	uint8_t vid_mask;
	uint8_t vprty_mask;
	uint8_t tos_mask;
	uint8_t mpls_lbl_mask;
	uint8_t mpls_tfc_mask;
	uint32_t ip_src_mask;
	uint32_t ip_dst_mask;
	uint8_t proto_mask;
	uint16_t port_src_mask;
	uint16_t port_dst_mask;
} mask_t;

typedef struct rule {
	int id;
	int pri;
	rvalue_t value;
	mask_t mask;
	void *action;
} rule_t;

typedef struct tuple_value {
	uint32_t ingress_port;
	uint64_t metadata;
	mac_t eth_src;
	mac_t eth_dst;
	uint16_t ether_type;
	uint16_t vid;
	uint8_t vprty;
	uint8_t tos;
	uint32_t mpls_lbl;
	uint8_t mpls_tfc;
	uint32_t ip_src;
	uint32_t ip_dst;
	uint8_t proto;
	uint16_t port_src;
	uint16_t port_dst;
} tvalue_t;

typedef struct tuple {
	int id;
	int rid;
	tvalue_t value;
} tuple_t;

/*
 * Global Variables
 */
char *ruleset_file = NULL;
int nb_rules = 0;
rule_t *rules;

int nb_tuples = 0;
tuple_t *tuples;

int *results;
#define nb_results nb_tuples

#endif /* COMMON_H_ */
