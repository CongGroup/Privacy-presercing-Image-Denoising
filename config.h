#ifndef GLOBAL_DEFS_H
#define GLOBAL_DEFS_H

#define CONFIG_DIR "./config/"
#define TESTDATA_DIR "./testdata/"

#define NUM_LAYER 5
#define STRIDE_SIZE 3

#define CONFIG_L 64
#define CONFIG_S 18
#define CONFIG_K 40
#define CONFIG_L_S 46 //CONFIG_L - CONFIG_S;

const int CONFIG_SCALING = 1 << CONFIG_S;

#define PATCH_IN_W 39
#define PATCH_IN_H 39
#define PATCH_IN_SIZE 1521
#define PATCH_OUT_W 17
#define PATCH_OUT_H 17
#define PATCH_OUT_SIZE 289

#define PAD_SIZE 22

#define CONFIG_NN_SIGMA 25.0
extern double CONFIG_IMG_SIGMA;

extern int CONFIG_NUM_WORKER;

#endif