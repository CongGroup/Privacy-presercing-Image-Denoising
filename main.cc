#include "types.h"
#include "evaluation.h"
#include "test_driver.h"
#include "utils.h"
#include "denoising.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <iostream>

extern double CONFIG_IMG_SIGMA;

int mt = 0;

void print_config()
{
    std::cout << "=============Config=============" << std::endl
        << "l : " << CONFIG_L << std::endl
        << "s : " << CONFIG_S << std::endl
        << "k : " << CONFIG_K << std::endl
        << "l-s : " << CONFIG_L_S << std::endl
        << "image noisy level : " << CONFIG_IMG_SIGMA << std::endl;
    if (mt == 1)
        std::cout << "number of worker threads : " << CONFIG_NUM_WORKER << std::endl;
    std::cout << "===============================" << std::endl;

}

extern mpz_class MOD_RING, MOD_HALFRING;
int main(int argc, char **argv)
{
	int c;
	opterr = 0;

    char in_path[200], out_path[200];

    int img_width = 0, img_height = 0;

	while((c = getopt(argc, argv, "s:n:i:o:w:h:et")) != -1) {
		switch(c) {
            case 's':
                CONFIG_IMG_SIGMA = atof(optarg);
                break;
            case 'n':
                mt = 1;
                CONFIG_NUM_WORKER = atoi(optarg);
                break;
            case 'i':
                strcpy(in_path, optarg);
                break;
            case 'o':
                strcpy(out_path, optarg);
                break;
            case 'w':
                img_width = atoi(optarg);
                break;
            case 'h':
                img_height = atoi(optarg);
                break;
			case 't':
				test_suit();
				exit(0);
            case 'e':
                eval_suit();
                exit(0);
			default:
				;
		}
	}

    if (img_width <= 0 || img_height <= 0) {
        std::cout << "Bad image size!\n";
        exit(1);
    }

    print_config();

    matrix_d img(img_width+PAD_SIZE, img_height+PAD_SIZE), denoised(img_width, img_height);
    load_table(img, in_path, img_width + PAD_SIZE, img_height + PAD_SIZE);

    denoise_init();
    if(mt == 1)    
        denoise_image_mt(img, denoised);
    else
        denoise_image(img, denoised);

    write_table(denoised, out_path);
}