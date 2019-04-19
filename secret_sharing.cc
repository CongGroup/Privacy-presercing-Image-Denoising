#include "secret_sharing.h"

#include "utils.h"

extern gmp_randclass gmp_prn;

void ss_encrypt(const mpz_class & plain, mpz_class & share0, mpz_class & share1)
{
	share1 = gmp_prn.get_z_bits(CONFIG_L);
	share0 = plain - share1;
    mod_2exp(share0, CONFIG_L);
}

void ss_encrypt(const matrix_z &plain, matrix_z &share0, matrix_z &share1)
{
	matrix_rand_2exp(share1, CONFIG_L);
	share0 = plain - share1;
    mod_2exp(share0, CONFIG_L);
}

void ss_decrypt(mpz_class & plain, const mpz_class & share0, const mpz_class & share1)
{
	plain = share0 + share1;
    mod_2exp(plain, CONFIG_L);
}

void ss_decrypt(matrix_z &plain, const matrix_z &share0, const matrix_z &share1)
{
	plain = share0 + share1;
    mod_2exp(plain, CONFIG_L);
}