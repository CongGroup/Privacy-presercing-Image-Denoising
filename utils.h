#ifndef UTILS_H
#define UTILS_H

#include "types.h"

void load_table(matrix_d &tab, 
				const char *filepath, int nrow, int ncol);

void load_table(matrix_z &tab, 
                const char *filepath, int nrow, int ncol);

void write_table(const matrix_z &tab, const char *filepath);

void write_table(const matrix_d &tab, const char *filepath);

void matrix_rand_2exp(matrix_z &mat, int l);

void mod_2exp(mpz_class &x, int n);

void mod_2exp(matrix_z &mat, int n);

void mod_double(matrix_d &mat, int n);

typedef enum{flat_column, flat_row} flatten_order_t;

template <typename MatrixT>
inline void matrix_flatten(const MatrixT &in, MatrixT &out, flatten_order_t order)
{
    assert(in.size() <= out.size());

    if (order == flat_column) {
        for (int i = 0, ncols = in.cols(); i < ncols; ++i)
            for (int j = 0, nrows = in.rows(); j < nrows; ++j)
                out(i*nrows + j) = in(j, i);
    }
    else if (order == flat_row) {
        for (int i = 0, nrows = in.rows(); i < nrows; ++i)
            for (int j = 0, ncols = in.cols(); j < ncols; ++j)
                out(i*ncols + j) = in(i, j);
    }
    else {
        assert(false);
    }
}

template<typename MatrixT>
inline void matrix_deflatten(const MatrixT & in, MatrixT & out, flatten_order_t order)
{
    assert(in.size() == out.size());

    if (order == flat_column) {
        for (int i = 0, ncols = out.cols(); i < ncols; ++i)
            for (int j = 0, nrows = out.rows(); j < nrows; ++j)
                out(j, i) = in(i*nrows + j);
    }
    else if (order == flat_row) {
        for (int i = 0, nrows = out.rows(); i < nrows; ++i)
            for (int j = 0, ncols = out.cols(); j < ncols; ++j)
                out(i, j) = in(i*ncols + j);
    }
    else {
        assert(false);
    }
}

inline matrix_d& matrix_z2d(const matrix_z &in, matrix_d &out)
{
    assert(in.size() == out.size());

    const mpz_class *z_data = in.data();
    double *d_data = out.data();
    for (int i = 0, size = in.size(); i < size; ++i)
        *(d_data + i) = (z_data + i)->get_d();   

    return out;
}

//#include <fstream>
extern std::ofstream den_log;
extern mpz_class MOD_RING, MOD_HALFRING;

//#include <iostream>
inline void ring_neg_recover(mpz_class &v)
{
    if (v >= MOD_HALFRING) {
        //std::cout << "neg recover before " << v << std::endl;
        v -= MOD_RING;
        //std::cout << "neg recover after " << v << std::endl;
    }
}

inline void matrix_neg_recover (matrix_z& mat)
{
    mpz_class *data = mat.data();
    for (int i = 0, size = mat.size(); i < size; ++i)
        //if (*(data + i) >= MOD_HALFRING) {
        //    *(data + i) -= MOD_RING;
        //    //std::cout << "neg value found: " << *data << std::endl;
        //    //den_log << "neg value found: " << *(data + i) << std::endl;
        //}
        ring_neg_recover(*(data + i));
}

#endif