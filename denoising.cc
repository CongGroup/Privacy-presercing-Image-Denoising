#include "denoising.h"

#include "utils.h"

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

extern gmp_randclass gmp_prn;
	
// some useful constants
mpz_class MOD_HALFRING(1ul << (CONFIG_L-1)), MOD_RING(MOD_HALFRING*2);
mpz_class MPZ_SCALED_ONE(CONFIG_SCALING), MPZ_SCALED_NEG_ONE(-CONFIG_SCALING);

// main model						
const char *weight_file[NUM_LAYER] = { CONFIG_DIR"w1.txt",
									   CONFIG_DIR"w2.txt",
									   CONFIG_DIR"w3.txt",
									   CONFIG_DIR"w4.txt",
									   CONFIG_DIR"w5.txt" };
const int weight_size[NUM_LAYER][2] = { {3072, 1522},
                                        {3072, 3073},
                                        {2559, 3073},
                                        {2047, 2560},
                                        {289,  2048} };

nn_layer_t layers[NUM_LAYER];
nn_buffer_t buffers[NUM_LAYER];
matrix_d pixel_weight(PATCH_OUT_H, PATCH_OUT_W);

// approximation parameters for tanh
tanh_param_z TANH_PARAM;

void nn_init()
{
	for (int i = 0; i<NUM_LAYER; ++i) {
		std::cout << "Set NN layer with " << weight_file[i] << std::endl;

        layers[i] = nn_layer_t(weight_size[i][0], weight_size[i][1]);
        
		load_table(layers[i].G.plain, weight_file[i], layers[i]._nrow, layers[i]._ncol);
		layers[i].G.encrypt();

		layers[i].tri_mul.encrypt();
		layers[i].tri_tanh.encrypt();
	}

    matrix_d raw_pixel_weight(PATCH_OUT_H, PATCH_OUT_W);
    load_table(raw_pixel_weight, CONFIG_DIR"pixel_weights.txt", PATCH_OUT_H, PATCH_OUT_W);
    pixel_weight = raw_pixel_weight.transpose();
}

void buf_init(nn_buffer_t *buf)
{
    for (int i = 0; i < NUM_LAYER; ++i) {
        buf[i] = nn_buffer_t(weight_size[i][0], weight_size[i][1]);
    }
}

void tanh_init()
{
	TANH_PARAM.n1 = -0.2716*CONFIG_SCALING;
    mod_2exp(TANH_PARAM.n1, CONFIG_L);
	TANH_PARAM.n2 = -0.0848*CONFIG_SCALING;
    mod_2exp(TANH_PARAM.n2, CONFIG_L);
	TANH_PARAM.c1 = 1 * CONFIG_SCALING;
	TANH_PARAM.c2 = 0.42654*CONFIG_SCALING;
	TANH_PARAM.d1 = 0.016*CONFIG_SCALING;
	TANH_PARAM.d1_i = 0.016*CONFIG_SCALING*CONFIG_SCALING; // for scaling correctness
	TANH_PARAM.d2 = 0.4519*CONFIG_SCALING;
	TANH_PARAM.d2_i = 0.4519*CONFIG_SCALING*CONFIG_SCALING; // for scaling correctness
	TANH_PARAM.a = 1.52*CONFIG_SCALING;
	TANH_PARAM.b = 2.57*CONFIG_SCALING;
	TANH_PARAM.neg_a = -1.52*CONFIG_SCALING;
	mod_2exp(TANH_PARAM.neg_a, CONFIG_L);
	TANH_PARAM.neg_b = -2.57*CONFIG_SCALING;
	mod_2exp(TANH_PARAM.neg_b, CONFIG_L);
}

void denoise_init()
{
    std::cout << "Initialize denoising parameters ...\n";
	nn_init();
    buf_init(buffers);
	tanh_init();

    mod_2exp(MPZ_SCALED_NEG_ONE, CONFIG_L);
}

//void denoise_reset()
//{
//	for(int i=0; i<NUM_LAYER; ++i)
//		nn_buf[i].reset();
//}

// simulate server-side encrypted image denoising
void denoise_patch(const matrix_z patch_share[2], matrix_z denoised_share[2],
                   const nn_layer_t nn_layer[5], nn_buffer_t nn_buf[5])
{
    for (int i = 0; i<NUM_LAYER; ++i)
        nn_buf[i].reset();

	for(int i=0; i<NUM_LAYER; ++i){
		//printf("Layer %d ...\n", i+1);
		const nn_layer_t &crt_layer = nn_layer[i];
        nn_buffer_t &crt_buf = nn_buf[i];
		/* Gp + b */
		secure_muliplication(crt_layer.G.share, (i==0)?(patch_share):(nn_buf[i-1].O.share),
		 					 crt_layer.tri_mul.share,
                             crt_buf.U, crt_buf.V,
                             crt_buf.Gpb.share,
		 					 0);
		// rescale
		secure_rescale(crt_buf.Gpb.share[0], crt_buf.Gpb.share[1]);

		/* approximating tanh(Gp+b) */
		// no need to do tanh for the last layer 
		if(i<(NUM_LAYER-1)) {
			// squaring and rescale (O = Gpb^2)
			secure_muliplication(crt_buf.Gpb.share, crt_buf.Gpb.share,
								 crt_layer.tri_tanh.share, 
                                 crt_buf.U_tanh, crt_buf.V_tanh,
                                 crt_buf.GpbSqr.share,
								 1);
			secure_rescale(crt_buf.GpbSqr.share[0], crt_buf.GpbSqr.share[1]);

			// generating four candidate polynomials
			tanh_polynomials(crt_buf.Gpb.share,  crt_buf.GpbSqr.share, 
							 crt_buf.T[0].share, crt_buf.T[1].share, 
							 crt_buf.T[2].share, crt_buf.T[3].share);
            for(int j=0; j<4; ++j)
                secure_rescale(crt_buf.T[j].share[0], crt_buf.T[j].share[1]);

			// simulate the computation undertaken by GC
			gc_simulate(crt_buf.Gpb,
						crt_buf.T,
						crt_buf.O);
		}
		else{
            denoised_share[0] = crt_buf.Gpb.share[0];
            denoised_share[1] = crt_buf.Gpb.share[1];
		}
	}
}

void denoise_image(const matrix_d &img, matrix_d &denoised)
{
    assert((img.rows() - PATCH_IN_W) % STRIDE_SIZE == 0 &&
        (img.cols() - PATCH_IN_H) % STRIDE_SIZE == 0);

    // buffers
    matrix_d in_patch(PATCH_IN_H, PATCH_IN_W), out_patch(PATCH_OUT_H, PATCH_OUT_W), out_flatten_buffer(PATCH_OUT_SIZE, 1);
    ss_tuple_z in_patch_scaled(PATCH_IN_SIZE + 1, 1), out_patch_scaled(PATCH_OUT_SIZE, 1);
    matrix_d pixel_weight_image(denoised.rows(), denoised.cols());

    double m = 0;
    for (int i = 0; i <= (img.rows() - PATCH_IN_W); i += STRIDE_SIZE) {
        for (int j = 0; j <= (img.cols() - PATCH_IN_H); j += STRIDE_SIZE) {
    //int indice[][3] = { {0,0}, {3, 252}, {144, 78} };
    //for (int k = 0; k < 3; ++k) {
            int i = indice[k][0];
            int j = indice[k][1];
            std::cout << "patch at " << i+1 << ", " << j+1 << std::endl;

            in_patch = img.block<PATCH_IN_H, PATCH_IN_W>(i, j);
            
            if (CONFIG_IMG_SIGMA != CONFIG_NN_SIGMA) {
                m = in_patch.mean() + 0.5;
                in_patch = in_patch.array() - m;
            }

            matrix_flatten<matrix_z>((in_patch*CONFIG_SCALING).cast<mpz_class>(), in_patch_scaled.plain, flat_row);
            in_patch_scaled.plain(PATCH_IN_SIZE) = MPZ_SCALED_ONE; // pad the last element

            in_patch_scaled.encrypt();

            denoise_patch(in_patch_scaled.share, out_patch_scaled.share,
                          layers, buffers);

            out_patch_scaled.decrypt();

            // recover negative values
            //mod_2exp(out_patch_scaled.plain, CONFIG_L);
            //matrix_neg_recover(out_patch_scaled.plain);
            
            // scale down and deflatten
            matrix_deflatten<matrix_d>(matrix_z2d(out_patch_scaled.plain, out_flatten_buffer)/CONFIG_SCALING, 
                                       out_patch, flat_row);

            static int cc = 0;
            write_table(out_patch, ("patch_"+std::to_string(i+1)+"_"+std::to_string(j+1)+".txt").c_str());

            if (CONFIG_IMG_SIGMA != CONFIG_NN_SIGMA)
                out_patch = out_patch.array() + m;

            out_patch = out_patch.array() * pixel_weight.array();

            denoised.block<PATCH_OUT_H, PATCH_OUT_W>(i, j) += out_patch;
            pixel_weight_image.block<PATCH_OUT_H, PATCH_OUT_H>(i, j) += pixel_weight; // this could be preprocessed
        }
    }
    exit(0);

    denoised = denoised.array() / pixel_weight_image.array();

    denoised = (denoised.array()*0.2 + 0.5) * (255/(CONFIG_NN_SIGMA/CONFIG_IMG_SIGMA));
}

std::mutex idx_mtx, img_mtx;
void denoise_worker(int id, int nrow, int ncol, int *next_row, int *next_col, 
                    const matrix_d *img, matrix_d* denoised, matrix_d* pixel)
{
    std::cout << "worker " << id << " starting!" << std::endl;

    nn_buffer_t local_buf[NUM_LAYER];
    buf_init(local_buf);

    matrix_d in_patch(PATCH_IN_H, PATCH_IN_W), out_patch(PATCH_OUT_H, PATCH_OUT_W), out_flatten_buffer(PATCH_OUT_SIZE, 1);
    ss_tuple_z in_patch_scaled(PATCH_IN_SIZE + 1, 1), out_patch_scaled(PATCH_OUT_SIZE, 1);
    double m = 0;

    int local_row=0, local_col=0;

    std::cout << "worker " << id << " started!" << std::endl;
    while (true) {
        idx_mtx.lock();

        if ((*next_col + PATCH_IN_W) <= ncol) {
            local_col = *next_col;
            local_row = *next_row;
            *next_col += STRIDE_SIZE;
        }
        else {
            *next_row += STRIDE_SIZE;
            if ((*next_row + PATCH_IN_H) <= nrow) {
                local_col = 0;
                local_row = *next_row;
                *next_col = STRIDE_SIZE;
            }
            else {
                std::cout << "worker " << id << " finished!" << std::endl;
                idx_mtx.unlock();
                return;
            }
        }
        in_patch = img->block<PATCH_IN_H, PATCH_IN_W>(local_row, local_col);
        std::cout << id << " : " << local_row << ", " << local_col << std::endl;
        idx_mtx.unlock();

        if (CONFIG_IMG_SIGMA != CONFIG_NN_SIGMA) {
            m = in_patch.mean() + 0.5;
            in_patch = in_patch.array() - m;
        }

        matrix_flatten<matrix_z>((in_patch*CONFIG_SCALING).cast<mpz_class>(), in_patch_scaled.plain, flat_row);
        mod_2exp(in_patch_scaled.plain, CONFIG_L); // convert to ring element
        in_patch_scaled.plain(PATCH_IN_SIZE) = MPZ_SCALED_ONE; // pad the last element

        in_patch_scaled.encrypt();

        denoise_patch(in_patch_scaled.share, out_patch_scaled.share,
                      layers, local_buf);

        out_patch_scaled.decrypt();

        // recover negative values
        mod_2exp(out_patch_scaled.plain, CONFIG_L);
        matrix_neg_recover(out_patch_scaled.plain);

        // scale down and deflatten
        matrix_deflatten<matrix_d>(matrix_z2d(out_patch_scaled.plain, out_flatten_buffer) / CONFIG_SCALING,
            out_patch, flat_row);

        if (CONFIG_IMG_SIGMA != CONFIG_NN_SIGMA)
            out_patch = out_patch.array() + m;

        out_patch = out_patch.array() * pixel_weight.array();

        img_mtx.lock();
        denoised->block<PATCH_OUT_H, PATCH_OUT_W>(local_row, local_col) += out_patch;
        pixel->block<PATCH_OUT_H, PATCH_OUT_H>(local_row, local_col) += pixel_weight;
        img_mtx.unlock();
    }
}

// multi-threading version
void denoise_image_mt(const matrix_d &img, matrix_d &denoised)
{
    assert((img.rows() - PATCH_IN_W) % STRIDE_SIZE == 0 &&
           (img.cols() - PATCH_IN_H) % STRIDE_SIZE == 0);

    std::cout << "Denoising started!\n";

    // this could be preprocessed for all images with the same size
    matrix_d pixel_weight_image(denoised.rows(), denoised.cols());

    std::vector<std::thread> workers;
 
    int next_row = 0, next_col = 0;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < CONFIG_NUM_WORKER; ++i)
        workers.push_back(std::thread(denoise_worker, i+1, 
                                      img.rows(), img.cols(), &next_row, &next_col,
                                      &img, &denoised, &pixel_weight_image));
    
    for (auto& w : workers)
        w.join();

    denoised = denoised.array() / pixel_weight_image.array();

    denoised = (denoised.array()*0.2 + 0.5) * (255 / (CONFIG_NN_SIGMA / CONFIG_IMG_SIGMA));
    auto end = std::chrono::steady_clock::now();

    std::cout << "Denoising finished in " << std::chrono::duration<double>(end - start).count() << " seconds.\n;";
}

// secure computation protocols
void secure_muliplication(const matrix_z shareA[2], const matrix_z shareB[2], 
						  const triplet_z shareTri[2],
						  // intermediate buffers
						  ss_tuple_z &U, ss_tuple_z &V, 
						  // output
						  matrix_z shareAB[2],
						  // whether to conduct piecewise multiplication
						  int piecewise)
{
	// 1)
	for(int i=0; i<2; ++i) {
		U.share[i] = shareA[i] - shareTri[i].X;
		V.share[i] = shareB[i] - shareTri[i].Y;
	}

	// 2)
	U.decrypt();
	V.decrypt();

	// 3)
	for(int i=0; i<2; ++i) {
		if(piecewise) {
			if(i==1)
				shareAB[i].array() += U.plain.array()*V.plain.array();
			shareAB[i].array() += U.plain.array()*shareTri[i].Y.array();
			shareAB[i].array() += shareTri[i].X.array()*V.plain.array();
		}
		else {
			if(i==1)
				shareAB[i] += U.plain*V.plain;
			shareAB[i] += U.plain*shareTri[i].Y;
			shareAB[i] += shareTri[i].X*V.plain;
		}
		shareAB[i] += shareTri[i].Z;
	}

    mod_2exp(shareAB[0], CONFIG_L);
    mod_2exp(shareAB[1], CONFIG_L);
}

void gc_simulate(ss_tuple_z &X, 
				 ss_tuple_z T[4],
				 ss_tuple_z &O)
{
	X.decrypt();
    for (int i = 0; i < 4; ++i) {
        T[i].decrypt();
    }
	
	//// for comparison			
	//mod_2exp(X.plain, CONFIG_L);

	for(int i=0, size=X.plain.size(); i<size; ++i) {
		mpz_class &x = X.plain(i);
		/* x >= 0 */
		if(x <= MOD_HALFRING) {
			// x <= a
			if(x <= TANH_PARAM.a) {
				O.plain(i) = T[0].plain(i);
			}
			// a < x <= b
			else if(x <= TANH_PARAM.b){
				O.plain(i) = T[1].plain(i);
			}
			// b < x
            else {
                O.plain(i) = MPZ_SCALED_ONE;
            }
		}
		/* x < 0 */
		else {
			// -a <= x
			if(TANH_PARAM.neg_a <= x){
				O.plain(i) = T[2].plain(i);
			}
			// -b <= x < -a
			else if(TANH_PARAM.neg_b <= x) {
				O.plain(i) = T[3].plain(i);
			}
			// x < -b
            else {
                O.plain(i) = MPZ_SCALED_NEG_ONE;
            }
		}
	}

	// pad output buffer for next iteration
	O.plain(X.plain.size()) = MPZ_SCALED_ONE;

	// encrypt tanh
	O.encrypt();
}

void tanh_polynomials(const matrix_z shareX[2], const matrix_z shareXsqr[2],
					  matrix_z shareT1[2], matrix_z shareT2[2], 
					  matrix_z shareT3[2], matrix_z shareT4[2])
{
	for(int i=0; i<2; ++i) {        
		// T1
		shareT1[i] = shareXsqr[i]*TANH_PARAM.n1;
		shareT1[i] += shareX[i]*TANH_PARAM.c1;
		if(i==1)
			shareT1[i].array() += TANH_PARAM.d1_i;
        mod_2exp(shareT1[i], CONFIG_L);

		// T2
		shareT2[i] = shareXsqr[i]*TANH_PARAM.n2;
		shareT2[i] += shareX[i]*TANH_PARAM.c2;
		if(i==1)
			shareT2[i].array() += TANH_PARAM.d2_i;
        mod_2exp(shareT2[i], CONFIG_L);

		// T3
		shareT3[i] = -shareXsqr[i]*TANH_PARAM.n1;
		shareT3[i] += shareX[i]*TANH_PARAM.c1;
		if(i==1)
			shareT3[i].array() -= TANH_PARAM.d1_i;
        mod_2exp(shareT3[i], CONFIG_L);

		// T4
		shareT4[i] = -shareXsqr[i]*TANH_PARAM.n2;
		shareT4[i] += shareX[i]*TANH_PARAM.c2;
		if(i==1)
			shareT4[i].array() -= TANH_PARAM.d2_i;
        mod_2exp(shareT4[i], CONFIG_L);
	}
}

void secure_rescale(mpz_class &v0, mpz_class &v1)
{
	mpz_class rescale_r;

    mod_2exp(v0, CONFIG_L);
    mod_2exp(v1, CONFIG_L);

	// 1)
	rescale_r = gmp_prn.get_z_bits(CONFIG_L+CONFIG_K);
	v0 += rescale_r;

	// 2)
	v1 += v0;
	v1 /= CONFIG_SCALING;
	mod_2exp(v1, CONFIG_L_S);

	// 3)
	v0 = rescale_r / CONFIG_SCALING;
	mod_2exp(v0, CONFIG_L_S);
	v0 = -v0;

    mod_2exp(v0, CONFIG_L);
    mod_2exp(v1, CONFIG_L);
}

void secure_rescale(matrix_z &v0, matrix_z &v1)
{
	for(int i=0, size=v0.size(); i<size; ++i) {
		secure_rescale(v0(i), v1(i));
	}
}