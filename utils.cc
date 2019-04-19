#include "utils.h"

#include <assert.h>
#include <stdio.h>

gmp_randclass gmp_prn(gmp_randinit_default);

void load_table(matrix_d &tab, 
				const char *filepath, int nrow, int ncol)
{
	assert(tab.rows() == nrow && tab.cols() == ncol);

	FILE *infile;
	infile = fopen(filepath, "r");
	if(infile == NULL) {
		printf("Fail to open %s: %s\n", filepath, strerror(errno));
		exit(1);
	}

	int i=0, j=0;
	// TODO: error checking
	for(i=0; i<nrow; ++i) {
		for(j=0; j<ncol; ++j) {
			fscanf(infile, "%lf", &tab(i, j));
		}
	}

	fclose(infile);
}

void load_table(matrix_z &tab, 
				const char *filepath, int nrow, int ncol)
{
	matrix_d raw_tab(nrow, ncol);
	load_table(raw_tab, filepath, nrow, ncol);
	tab = (raw_tab*CONFIG_SCALING).cast<mpz_class>();
}

void write_table(const matrix_z &tab, const char *filepath)
{
	FILE *outFile;
	outFile = fopen(filepath, "w");
	for(int i=0; i<tab.rows(); ++i) {
		for(int j=0; j<tab.cols(); ++j)
			fprintf(outFile, "%f\t", tab(i, j).get_d()/CONFIG_SCALING);
		fprintf(outFile, "\n");
	}
	fclose(outFile);
}

void write_table(const matrix_d & tab, const char * filepath)
{
    printf("Writing to %s\n", filepath);

    FILE *outFile;
    outFile = fopen(filepath, "w");
    for (int i = 0; i<tab.rows(); ++i) {
        for (int j = 0; j<tab.cols(); ++j)
            fprintf(outFile, "%f\t", tab(i, j));
        fprintf(outFile, "\n");
    }
    fclose(outFile);
}

void matrix_rand_2exp(matrix_z &mat, int l)
{	 
	mpz_class *data = mat.data();
	for (int i = 0, size = mat.size(); i < size; ++i)
		*(data + i) = gmp_prn.get_z_bits(l);				
}

void mod_2exp(mpz_class &x, int n)
{
	mpz_fdiv_r_2exp(x.get_mpz_t(), x.get_mpz_t(), n);
}

void mod_2exp(matrix_z &mat, int n)
{
	mpz_class *data = mat.data();
	for (int i = 0, size = mat.size(); i<size; ++i) {
		mod_2exp(*(data + i), n);
	}
}

void mod_double(matrix_d & mat, int n)
{
    double *data = mat.data();
    for (int i = 0, size = mat.size(); i<size; ++i) {
        *(data + i) = (int)*(data + i) % n;
    }
}

//std::ofstream den_log("log.txt");