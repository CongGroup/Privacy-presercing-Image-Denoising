#ifndef SECRET_SHARING_H
#define SECRET_SHARING_H

#include "types.h"

void ss_encrypt(const mpz_class &plain, mpz_class &share0, mpz_class &share1);

void ss_encrypt(const matrix_z &plain, matrix_z &share0, matrix_z &share1);

void ss_decrypt(mpz_class &plain, const mpz_class &share0, const mpz_class &share1);

void ss_decrypt(matrix_z &plain, const matrix_z &share0, const matrix_z &share1);

#endif