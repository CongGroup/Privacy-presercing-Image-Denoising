#include "test_driver.h"

#include "denoising.h"
#include "secret_sharing.h"
#include "utils.h"

#include <iostream>
#include <fstream>

extern gmp_randclass gmp_prn;

// Test cases
void test_vector()
{
	printf("========%s=========\n", __func__);
	matrix_d A(3,2), B(2, 1);
	load_table(A, "test_A.txt", 3, 2);
	load_table(B, "test_B.txt", 2, 1);
	std::cout << "A\n" << A << std::endl << std::endl;
	std::cout << "B\n" << B << std::endl << std::endl;

	std::cout << "A*B\n" << A * B << std::endl << std::endl;

	matrix_d A1 = matrix_d::Constant(3, 1, 10);
	std::cout << "(10, 10, 10)+A*B\n"<< A1+A*B << std::endl << std::endl;
	std::cout << "(10, 10, 10)*A*B\n" << A1.array()*((A*B).array()) << std::endl << std::endl;

	matrix_z C=(A*CONFIG_SCALING).cast<mpz_class>(), D=(B*CONFIG_SCALING).cast<mpz_class>();
	std::cout << "C\n" << C << std::endl << std::endl;
	std::cout << "C/1000\n" << C/1000 << std::endl << std::endl;
	std::cout << "D\n" << D << std::endl << std::endl;

	std::cout << "(10^6, 10^6, 10^6)+C*D\n" << matrix_z::Constant(3, 1, 1000000)+C*D << std::endl << std::endl;
}

 void test_secure_mul_single()
 {
 	printf("========%s=========\n", __func__);
 	double rawA=0.7, rawB=2.5;
 	mpz_class a=rawA*CONFIG_SCALING, b= rawB*CONFIG_SCALING;

	mpz_class x = gmp_prn.get_z_bits(CONFIG_L), y = gmp_prn.get_z_bits(CONFIG_L);
	mpz_class z = x * y;

 	mpz_class shareA[2], shareB[2], shareX[2], shareY[2], shareZ[2], shareU[2], shareV[2];
	
 	ss_encrypt(a, shareA[0], shareA[1]);
 	ss_encrypt(b, shareB[0], shareB[1]);
 	ss_encrypt(x, shareX[0], shareX[1]);
 	ss_encrypt(y, shareY[0], shareY[1]);
 	ss_encrypt(z, shareZ[0], shareZ[1]);
	
 	mpz_class u, v;

 	for(int i=0; i<2; ++i) {
 		shareU[i] = shareA[i] - shareX[i];
 		shareV[i] = shareB[i] - shareY[i];
 	}
	
 	u = shareU[0] + shareU[1];
 	v = shareV[0] + shareV[1];

 	mpz_class ab, shareAB[2];
	
 	for(int i=0; i<2; ++i) {
 		if(i==1)
 			shareAB[i] += u * v;
 		shareAB[i] += u * shareY[i];
 		shareAB[i] += v * shareX[i];
 		shareAB[i] +=  shareZ[i];
 	}

 	ab = shareAB[0] + shareAB[1];
    mod_2exp(ab, CONFIG_L);

 	printf("%f\n%f\n\n", rawA*rawB, ab.get_d()/CONFIG_SCALING/CONFIG_SCALING);
 }

 void test_secure_mul()
 {
 	printf("========%s=========\n", __func__);

 	// preparation
 	ss_tuple_z A(3,2), B(2,1), U(3,2), V(2,1), AB(3,1);

 	load_table(A.plain, "test_A.txt", 3, 2);
	load_table(B.plain, "test_B.txt", 2, 1);

	A.encrypt();
	B.encrypt();

 	tri_tuple_z tri(3,2,1);
	tri.encrypt();

 	// run protocol
    secure_muliplication(A.share, B.share, tri.share,
     					 U, V, AB.share, 0);

	AB.decrypt();

    mod_2exp(AB.plain, CONFIG_L);
    matrix_neg_recover(AB.plain);

	matrix_d rawA(3, 2), rawB(2,1);
	load_table(rawA, "test_A.txt", 3, 2);
	load_table(rawB, "test_B.txt", 2, 1);
	matrix_d rawC = rawA * rawB;

	printf("plaintext:\n");
	std::cout << rawC << std::endl;

 	printf("\nSecured:\n");
	for(int i=0; i<AB.plain.size(); ++i)
		std::cout << AB.plain(i).get_d() / CONFIG_SCALING / CONFIG_SCALING << std::endl;

 	printf("\n");
}

void test_secure_mul_pw()
{
	printf("========%s=========\n", __func__);

	// preparation
	ss_tuple_z A(2,1), B(2,1), U(2,1), V(2,1), AB(2,1);

	load_table(A.plain, "test_C.txt", 2, 1);
	load_table(B.plain, "test_B.txt", 2, 1);

	A.encrypt();
	B.encrypt();

	tri_tuple_z tri(2);
	tri.encrypt();

	// run protocol
	secure_muliplication(A.share, B.share, tri.share,
     						U, V, AB.share, 1);
    secure_rescale(AB.share[0], AB.share[1]);
	AB.decrypt();

	// report results
	printf("plaintext:\n");
	matrix_d rawA(2, 1), rawB(2, 1);
	load_table(rawA, "test_C.txt", 2, 1);
	load_table(rawB, "test_B.txt", 2, 1);
	matrix_d rawC = rawA.array() * rawB.array();
	std::cout << rawC << std::endl;

	printf("\nSecured:\n");
	for (int i = 0; i<AB.plain.size(); ++i)
		std::cout << AB.plain(i).get_d() / CONFIG_SCALING << std::endl;

	printf("\n");
}

 void test_tanh()
 {
 	printf("========%s=========\n", __func__);
 	double n=2.6;
 	printf("n : %f\n", n);

 	if(n >= 0) {
 		if(n <= 1.52)
 			printf("plaintext: t1 %f\n", -0.2716*n*n+1*n+0.016);
 		else if(n <= 2.57)
 			printf("plaintext: t2 %f\n", -0.0848*n*n+0.42654*n+0.4519);
 		else
 			printf("plaintext: 1 %f\n", 1.0);
 	}
 	else {
 		if(-1.52 <= n)
 			printf("plaintext: t3 %f\n", 0.2716*n*n+1*n-0.016);
 		else if(-2.57 <= n)
 			printf("plaintext: t4 %f\n", 0.0848*n*n+0.42654*n-0.4519);
 		else
 			printf("plaintext: -1 %f\n", -1.0);
 	}

	ss_tuple_z O(2, 1), N(1, 1), N2(1, 1), U(1, 1), V(1, 1);
	ss_tuple_z T[4];
	for (int i = 0; i < 4; ++i)
		T[i] = ss_tuple_z(1, 1);
 	tri_tuple_z tri(1,1,1);
	tri.encrypt();

 	N.plain(0) = n*CONFIG_SCALING;
	N.encrypt();

 	// start
 	secure_muliplication(N.share, N.share, tri.share, 
 						 U, V, 
 						 N2.share, 1);
    secure_rescale(N2.share[0], N2.share[1]);
	N2.decrypt();
 	//std::cout << "x: " << N.plain(0).get_d() / CONFIG_SCALING << " x^2: " << N2.plain(0).get_d() / CONFIG_SCALING << std::endl;

 	// generating four candidate polynomials
 	tanh_polynomials(N.share, N2.share, 
 					 T[0].share, T[1].share, 
 					 T[2].share, T[3].share);
    for (int i = 0; i < 4; ++i)
        secure_rescale(T[i].share[0], T[i].share[1]);

 	// simulate the computation undertaken by GC
 	gc_simulate(N,
 				T,
 				O);

	O.decrypt();
    matrix_neg_recover(O.plain);

 	printf("secured: %f\n", O.plain(0).get_d()/CONFIG_SCALING);
 	printf("\n");
 }

 void test_rescale()
 {
 	printf("========%s=========\n", __func__);

 	double aa=0.17, bb=5.45;
	mpz_class a = aa * CONFIG_SCALING, b = bb * CONFIG_SCALING;
	mpz_class c = a * b, c0, c1;

	ss_encrypt(c, c0, c1);

	gmp_printf("before rescale: %f (true %f)\n", c.get_d()/CONFIG_SCALING/CONFIG_SCALING, aa*bb);

 	secure_rescale(c0, c1);
 	mpz_class cc;
	ss_decrypt(cc, c0, c1);
 	mod_2exp(cc, CONFIG_L);

 	gmp_printf("after rescale: %f\n\n", cc.get_d()/CONFIG_SCALING);
}

//void test_denoise_patch()
//{
//    printf("========%s=========\n", __func__);
//
//    matrix_z img(576, 768);
//    ss_tuple_z patch(1522, 1), denoised(289, 1);
//    load_table(img, CONFIG_DIR"noisyImage.txt", 576, 768);
//    matrix_flatten<matrix_z>(img.block(0, 0, PATCH_IN_H, PATCH_IN_W), patch.plain, flat_row);
//    write_table(patch.plain, "resized.txt");
//    patch.plain(1521) = 1 << CONFIG_S;
//    patch.encrypt();
//
//    tanh_init();
//    denoise_patch(patch.share, denoised.share);
//    denoised.decrypt();
//
//    write_table(denoised.plain, "denoised.txt");
//}

void test_denoise_image()
{
    printf("========%s=========\n", __func__);

    matrix_d img(534, 534), denoised(512, 512);
    load_table(img, TESTDATA_DIR"image/""25/""testHealthPad001.25.txt", 534, 534);

    denoise_init();
    denoise_image(img, denoised);

    write_table(denoised, "denoised.txt");
}

void test_denoise_image_mt()
{
    printf("========%s=========\n", __func__);

    matrix_d img(534, 534), denoised(512, 512);
    load_table(img, TESTDATA_DIR"image/""25/""testHealthPad001.25.txt", 534, 534);

    denoise_init();
    denoise_image_mt(img, denoised);

    write_table(denoised, "denoised.txt");
}

void test_suit()
{
	std::cout << "test_suit\n";
	tanh_init();

	//test_vector();
	test_secure_mul_single();
	test_secure_mul();
	test_secure_mul_pw();
	test_tanh();
	test_rescale();
    //test_denoise_patch();
    //test_denoise_image();
}