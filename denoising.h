/* main secure protocols used in the denoising */

#ifndef DENOISING_H
#define DENOISING_H

#include "types.h"

// denoising APIs
void tanh_init();

void denoise_init();

//void denoise_reset();

void denoise_patch(const matrix_z patch_share[2], matrix_z denoised_share[2],
                   const nn_layer_t nn_layer[5], nn_buffer_t nn_buf[5]);

void denoise_image(const matrix_d &img, matrix_d &denoised);

// multi-threading version
void denoise_image_mt(const matrix_d &img, matrix_d &denoised);

// secure computation protocols
void ss_encrypt(const matrix_z &plain, matrix_z &share0, matrix_z &share1);

void ss_decrypt(matrix_z &plain, const matrix_z &share0, const matrix_z &share1);

void secure_muliplication(const matrix_z shareA[2], const matrix_z shareB[2], 
						  const triplet_z shareTri[2],
						  // intermediate buffers
						  ss_tuple_z &U, ss_tuple_z &V, 
						  // output
						  matrix_z shareAB[2],
						  // whether to conduct piecewise multiplication
						  int piecewise);

void gc_simulate(ss_tuple_z &X, 
				 ss_tuple_z T[4],
				 ss_tuple_z &O);

void tanh_polynomials(const matrix_z shareX[2], const matrix_z shareXsqr[2],
					  // must be zeroized
					  matrix_z shareT1[2], matrix_z shareT2[2], 
					  matrix_z shareT3[2], matrix_z shareT4[2]);

void secure_rescale(mpz_class &v0, mpz_class &v1);

void secure_rescale(matrix_z &v0, matrix_z &v1);

#endif