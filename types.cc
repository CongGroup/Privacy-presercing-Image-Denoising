#include "types.h"

#include "utils.h"
#include "secret_sharing.h"

triplet::triplet()
{
}

triplet::triplet(int X_row, int X_col, int Y_col)
: X(X_row, X_col),
  Y(X_col, Y_col),
  Z(X_row, Y_col) 
{
	// no need for shares
	matrix_rand_2exp(X, CONFIG_L);
	matrix_rand_2exp(Y, CONFIG_L);
	Z = X*Y;
    //mod_2exp(Z, CONFIG_L);
}

triplet::triplet(int n)
: X(n, 1),
  Y(n, 1),
  Z(n, 1)
{
	// no need for shares
	matrix_rand_2exp(X, CONFIG_L);
	matrix_rand_2exp(X, CONFIG_L);
	Z = X.array()*Y.array();
    //mod_2exp(Z, CONFIG_L);
}

ss_tuple::ss_tuple(int nrow, int ncol)
: plain(nrow, ncol),
  share{ matrix_z(nrow, ncol), matrix_z(nrow, ncol) }
{
}

ss_tuple::ss_tuple()
{
}

ss_tuple::ss_tuple(const ss_tuple & s)
{
	*this = s;
}

ss_tuple & ss_tuple::operator=(const ss_tuple & rhs)
{
	plain = rhs.plain;
	share[0] = rhs.share[0];
	share[1] = rhs.share[1];

	return *this;
}

void ss_tuple::encrypt() 
{
	ss_encrypt(plain, share[0], share[1]);
}

void ss_tuple::decrypt() 
{
	ss_decrypt(plain, share[0], share[1]);
}

void ss_tuple::reset()
{
    plain.setZero();
    share[0].setZero();
    share[1].setZero();
}


tri_tuple::tri_tuple(int X_row, int X_col, int Y_col)
: plain(X_row, X_col, Y_col),
  share{triplet_z(X_row, X_col, Y_col) , triplet_z(X_row, X_col, Y_col)}
{
}

tri_tuple::tri_tuple(int n)
: plain(n),
  share({ triplet_z(n), triplet_z(n) })
{
}

tri_tuple::tri_tuple()
{
}

tri_tuple::tri_tuple(const tri_tuple &t)
{
	*this = t;
}

tri_tuple & tri_tuple::operator=(const tri_tuple &rhs)
{
	plain = rhs.plain;
	share[0] = rhs.share[0];
	share[1] = rhs.share[1];

	return *this;
}

void tri_tuple::encrypt() 
{
	ss_encrypt(plain.X, share[0].X, share[1].X);
	ss_encrypt(plain.Y, share[0].Y, share[1].Y);
	ss_encrypt(plain.Z, share[0].Z, share[1].Z);
}

void tri_tuple::decrypt() 
{
	ss_decrypt(plain.X, share[0].X, share[1].X);
	ss_decrypt(plain.Y, share[0].Y, share[1].Y);
	ss_decrypt(plain.Z, share[0].Z, share[1].Z);
}

nn_layer::nn_layer()
{
}

nn_layer::nn_layer(int nrow, int ncol)
: _nrow(nrow), _ncol(ncol),
  G(nrow, ncol),
  tri_mul(nrow, ncol, 1),
  tri_tanh(nrow) 
{
}

nn_buffer::nn_buffer()
{
}

nn_buffer::nn_buffer(int nrow, int ncol)
: _nrow(nrow), _ncol(ncol),
  U(nrow, ncol), V(ncol, 1), Gpb(nrow, 1),
  GpbSqr(nrow, 1), U_tanh(nrow, 1), V_tanh(nrow, 1),
  //T{ ss_tuple_z(nrow, 1), ss_tuple_z(nrow, 1), ss_tuple_z(nrow, 1), ss_tuple_z(nrow, 1) },
  O(nrow + 1, 1) // output buffer will be padded by one more element
{
    for (int i = 0; i < 4; ++i)
        T[i] = ss_tuple_z(nrow, 1);
}

void nn_buffer::reset()
{
    U.reset();
    V.reset();
    Gpb.reset();
    GpbSqr.reset();
    U_tanh.reset();
    V_tanh.reset();
    for (int i = 0; i<4; ++i)
        T[i].reset();
    O.reset();
}
