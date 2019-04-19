#ifndef TYPES_H
#define TYPES_H

#include "config.h"

#include <gmpxx.h>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace Eigen {
  template<> struct NumTraits<mpz_class> : GenericNumTraits<mpz_class>
  {
    typedef mpz_class Real;
    typedef mpz_class NonInteger;
    typedef mpz_class Nested;
    static inline Real epsilon() { return 0; }
    static inline Real dummy_precision() { return 0; }
    //static inline Real digits10() { return 0; }
    enum {
      IsInteger = 1,
      IsSigned = 1,
      IsComplex = 0,
      RequireInitialization = 1,
      ReadCost = 1,
      AddCost = 1,
      MulCost = 1
    };
  };
}

typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> matrix_i;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_d;
typedef Eigen::Matrix<mpz_class, Eigen::Dynamic, Eigen::Dynamic> matrix_z;

typedef struct tanh_param {
	mpz_class n1;
	mpz_class n2;
	mpz_class c1;
	mpz_class c2;
	mpz_class d1;
	mpz_class d1_i;
	mpz_class d2;
	mpz_class d2_i;
	mpz_class a;
	mpz_class b;
	mpz_class neg_a;
	mpz_class neg_b;
} tanh_param_z;

// Triplet for beaver's secure multiplication protocol
typedef struct triplet {
	triplet();

	// matrix version
	triplet(int X_row, int X_col, int Y_col);

	// piecewise version
	explicit triplet(int n);

	matrix_z X;
	matrix_z Y;
	matrix_z Z;
} triplet_z;

// Secret-sharing tuples used for convenient prototyping
typedef struct ss_tuple {
	// big three
	ss_tuple();
    ss_tuple(const ss_tuple& s);
	ss_tuple& operator= (const ss_tuple& rhs);

	ss_tuple(int nrow, int ncol);

	void encrypt();

	void decrypt();

    void reset();

	matrix_z plain;
	matrix_z share[2];
} ss_tuple_z;

typedef struct tri_tuple {
	// big three
	tri_tuple();
	tri_tuple(const tri_tuple& t);
	tri_tuple& operator= (const tri_tuple& rhs);

	// matrix version
	tri_tuple(int X_row, int X_col, int Y_col);

	// piecewise version
	explicit tri_tuple(int n);

	void encrypt();

	void decrypt();

	triplet_z plain;
	triplet_z share[2];
} tri_tuple_z;

// Neural network layer for encrypted image denoising
typedef struct nn_layer {
	nn_layer();
	nn_layer(int nrow, int ncol);

	int _nrow, _ncol;

	// input weights
	ss_tuple_z G; 

	// triplet for secure multiplication
	tri_tuple_z tri_mul; 

	// triplet for tanh approximation
	tri_tuple_z tri_tanh;

} nn_layer_t;

// Intermediate buffers used during denoising
typedef struct nn_buffer {
    nn_buffer();

    nn_buffer(int nrow, int ncol);

    void reset();

    int _nrow, _ncol;

    // for computing Gpb
    ss_tuple_z U, V, Gpb;
    // for approximating tanh
    ss_tuple_z GpbSqr, U_tanh, V_tanh, T[4];

    // output buffer
    ss_tuple_z O;
} nn_buffer_t;

#endif