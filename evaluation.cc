#include "evaluation.h"

#include "denoising.h"
#include "types.h"
#include "secret_sharing.h"
#include "utils.h"

#include <chrono>
#include <iostream>

#define EVAL_TRIALS 1

void eval_local()
{
    printf("\n========%s=========\n", __func__);

    matrix_d patch = matrix_d::Random(PATCH_IN_H, PATCH_IN_W);
    mod_double(patch, 256);
    matrix_d image = matrix_d::Random(534, 534);
    mod_double(image, 256);
    matrix_z scaled(PATCH_IN_H, PATCH_IN_W), s0(PATCH_IN_H, PATCH_IN_W), s1(PATCH_IN_H, PATCH_IN_W);

    const double sigma_nn = 25.0;
    const double sigma_img = 15.0;
    const double sigma_prod = sigma_nn * sigma_img;
    double m = 0;

    auto start = std::chrono::steady_clock::now(), end = std::chrono::steady_clock::now();
    double time_preproc = 0, time_enc = 0;
    
    /* per-patch time */
    for (int t = 0; t < EVAL_TRIALS; ++t) {
        // pre-processing
        start = std::chrono::steady_clock::now();

        patch *= sigma_prod;
        patch = (patch.array() / 255 - 0.5) * 5;
        m = patch.mean() + 0.5;
        patch = patch.array() - m;

        end = std::chrono::steady_clock::now();
        time_preproc += std::chrono::duration<double, std::nano>(end - start).count();

        // encryption
        start = std::chrono::steady_clock::now();
        scaled = (patch*CONFIG_SCALING).cast<mpz_class>();
        ss_encrypt(scaled, s0, s1);
        end = std::chrono::steady_clock::now();
        time_enc += std::chrono::duration<double, std::nano>(end - start).count();
    }
    std::cout << "local computation time for one patch : " << std::endl
        << "\t- preprocessing: " << time_preproc / EVAL_TRIALS << " ns " << std::endl
        << "\t- encryption: " << time_enc / EVAL_TRIALS << " ns " << std::endl;

    /* per-image (512 x 512) time */
    double time_img = 0;
    for (int t = 0; t < EVAL_TRIALS; ++t) {
        start = std::chrono::steady_clock::now();

        for (int i = 0; (i + PATCH_IN_H) <= 534; i += 3) {
            for (int j = 0; (j + PATCH_IN_W) <= 534; j += 3) {
                matrix_d p = image.block<PATCH_IN_H, PATCH_IN_W>(i, j);
                p *= sigma_prod;
                p = (p.array() / 255 - 0.5) * 5;
                m = p.mean() + 0.5;
                p = p.array() - m;

                scaled = (p*CONFIG_SCALING).cast<mpz_class>();
                ss_encrypt(scaled, s0, s1);
            }
        }

        end = std::chrono::steady_clock::now();
        time_img += std::chrono::duration<double, std::nano>(end - start).count();
    }
    std::cout << "local computation time for one image (512x512) : \n\t"
        << time_img / EVAL_TRIALS << "ns" << std::endl;

    /* local per-patch */
    double time_post = 0, time_dec = 0;
    matrix_d dec_pat(PATCH_IN_H, PATCH_IN_W);
    matrix_z dec_scaled_pat(PATCH_IN_H, PATCH_IN_W), dec_img(534, 534);
    for (int t = 0; t < EVAL_TRIALS; ++t) {
        // decryption
        start = std::chrono::steady_clock::now();

        ss_decrypt(dec_scaled_pat, s0, s1);
        matrix_z2d(dec_scaled_pat, dec_pat);

        end = std::chrono::steady_clock::now();
        time_dec += std::chrono::duration<double, std::nano>(end - start).count();

        // post-processing
        start = std::chrono::steady_clock::now();
        
        dec_pat = dec_pat.array() + m;
        dec_pat = (dec_pat.array() / 5 + 0.5) * 255 / sigma_prod;

        end = std::chrono::steady_clock::now();
        time_post += std::chrono::duration<double, std::nano>(end - start).count();
    }
    std::cout << "local computation time for one patch : " << std::endl
                << "\t- decryption: " << time_dec / EVAL_TRIALS << " ns " << std::endl
                << "\t- postprocessing: " << time_post / EVAL_TRIALS << " ns " << std::endl;

    /* per-image (512 x 512) time */
    time_img = 0;
    for (int t = 0; t < EVAL_TRIALS; ++t) {
        start = std::chrono::steady_clock::now();

        for (int i = 0; (i + PATCH_IN_H) <= 534; i += 3) {
            for (int j = 0; (j + PATCH_IN_W) <= 534; j += 3) {
                ss_decrypt(dec_scaled_pat, s0, s1);
                matrix_z2d(dec_scaled_pat, dec_pat);
                dec_pat = dec_pat.array() + m;
                dec_pat = (dec_pat.array() / 5 + 0.5) * 255 / sigma_prod;
                image.block<PATCH_IN_H, PATCH_IN_W>(i, j) += dec_pat;
            }
        }

        end = std::chrono::steady_clock::now();
        time_img += std::chrono::duration<double, std::nano>(end - start).count();
    }
    std::cout << "local computation time for one image (512x512) : \n\t"
                  << time_img / EVAL_TRIALS << "ns" << std::endl;
}

void eval_cloud()
{
    printf("\n========%s=========\n", __func__);
    
    // use the real image data for accurate estimation, as the cost may depend on the GMP internal 
    matrix_d img(534, 534);
    load_table(img, "images/txtdata/""25/""testHealthPad001.25.txt", 534, 534);
    
    /* atomic operations */
    const int atomic_eval_size = 128;
    matrix_z rlt(atomic_eval_size, 1);
    ss_tuple_z va(atomic_eval_size, 1), vb(atomic_eval_size, 1);
    va.plain = (img.block<atomic_eval_size, 1>(0, 0)*CONFIG_SCALING).cast<mpz_class>();
    vb.plain = (img.block<atomic_eval_size, 1>(0, 1)*CONFIG_SCALING).cast<mpz_class>();
    va.encrypt();
    vb.encrypt();

    // secure addition
    auto start = std::chrono::steady_clock::now();
    rlt = va.share[0] + vb.share[0];
    rlt = va.share[1] + vb.share[1];
    auto end = std::chrono::steady_clock::now();
    std::cout << "cloud computation - secure addition :\n\t " 
              << std::chrono::duration<double, std::nano>(end - start).count() / atomic_eval_size / 2 
              << "ns" << std::endl;

    // secure multiplication
    ss_tuple_z U(atomic_eval_size, 1), V(atomic_eval_size, 1), ab(atomic_eval_size, 1);
    tri_tuple_z tri(atomic_eval_size);
    tri.encrypt();

    start = std::chrono::steady_clock::now();
    secure_muliplication(va.share, vb.share,
                         tri.share,
                         U, V,
                         ab.share,
                         1);
    end = std::chrono::steady_clock::now();
    std::cout << "cloud computation - secure multiplication : \n\t"
              << std::chrono::duration<double, std::nano>(end - start).count() / atomic_eval_size / 2 
              << "ns" << std::endl;

    // secure rescale
    start = std::chrono::steady_clock::now();
    secure_rescale(ab.share[0], ab.share[1]);
    end = std::chrono::steady_clock::now();
    std::cout << "cloud computation - secure rescaling : \n\t"
              << std::chrono::duration<double, std::nano>(end - start).count() / atomic_eval_size / 2 
              << "ns" << std::endl;

    // secure approximation of tanh without GC
    ss_tuple_z absqr(atomic_eval_size, 1);
    ss_tuple_z T[4];
    for(int i=0; i<4; ++i)
        T[i] = ss_tuple_z(atomic_eval_size, 1);
    secure_muliplication(ab.share, ab.share, tri.share, U, V, absqr.share, 1);
    secure_rescale(absqr.share[0], absqr.share[1]);
    tanh_init();

    start = std::chrono::steady_clock::now();
    tanh_polynomials(ab.share, absqr.share, T[0].share, T[1].share, T[2].share, T[3].share);
    end = std::chrono::steady_clock::now();
    std::cout << "cloud computation - secure approximation of tanh : \n\t"
        << std::chrono::duration<double, std::nano>(end - start).count() / atomic_eval_size / 2
        << "ns" << std::endl;
}

extern nn_layer_t layers[NUM_LAYER];
extern nn_buffer_t buffers[NUM_LAYER];

double eval_patch_unit(matrix_z patch_share[2], matrix_z denoised_share[2])
{
    double total_time = 0;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i<NUM_LAYER; ++i)
        buffers[i].reset();
    auto end = std::chrono::steady_clock::now();

    total_time += std::chrono::duration<double, std::nano>(end - start).count();

    for (int i = 0; i<NUM_LAYER; ++i) {
        start = std::chrono::steady_clock::now();

        const nn_layer_t &crt_layer = layers[i];
        nn_buffer_t &crt_buf = buffers[i];
        /* Gp + b */
        secure_muliplication(crt_layer.G.share, (i == 0) ? (patch_share) : (buffers[i - 1].O.share),
            crt_layer.tri_mul.share,
            crt_buf.U, crt_buf.V,
            crt_buf.Gpb.share,
            0);
        // rescale
        secure_rescale(crt_buf.Gpb.share[0], crt_buf.Gpb.share[1]);

        /* approximating tanh(Gp+b) */
        // no need to do tanh for the last layer 
        if (i<(NUM_LAYER - 1)) {
            // squaring and rescale (O = Gpb^2)
            secure_muliplication(crt_buf.Gpb.share, crt_buf.Gpb.share,
                crt_layer.tri_tanh.share,
                crt_buf.U_tanh, crt_buf.V_tanh,
                crt_buf.GpbSqr.share,
                1);
            secure_rescale(crt_buf.GpbSqr.share[0], crt_buf.GpbSqr.share[1]);

            // generating four candidate polynomials
            tanh_polynomials(crt_buf.Gpb.share, crt_buf.GpbSqr.share,
                crt_buf.T[0].share, crt_buf.T[1].share,
                crt_buf.T[2].share, crt_buf.T[3].share);

            end = std::chrono::steady_clock::now();
            // exclude the time for gc simulation, the true estimation of which will be added later

            // simulate the computation undertaken by GC
            gc_simulate(crt_buf.Gpb,
                crt_buf.T,
                crt_buf.O);
        }
        else {
            denoised_share[0] = crt_buf.Gpb.share[0];
            denoised_share[1] = crt_buf.Gpb.share[1];

            end = std::chrono::steady_clock::now();
        }

        total_time += std::chrono::duration<double, std::nano>(end - start).count();
    }

    return total_time;
}

void eval_patch()
{
    printf("\n========%s=========\n", __func__);
    
    matrix_z img(576, 768);
    ss_tuple_z patch(1522, 1), denoised(289, 1);
    load_table(img, CONFIG_DIR"noisyImage.txt", 576, 768);

    double sum_time = 0;
    const int num_idx = 2;
    int rand_idx[num_idx][2];
    for (int i = 0; i < num_idx; ++i) {
        rand_idx[i][0] = rand() % (576 - 39);
        rand_idx[i][1] = rand() % (768 - 39);
    }

    denoise_init();
    for (int i = 0; i < num_idx; ++i) {
        std::cout << "patch " << i + 1 << std::endl;
        matrix_flatten<matrix_z>(img.block<PATCH_IN_H, PATCH_IN_W>(rand_idx[i][0], rand_idx[i][1]), patch.plain, flat_row);
        patch.plain(1521) = 1 << CONFIG_S;
        patch.encrypt();

        sum_time += eval_patch_unit(patch.share, denoised.share);
    }

    std::cout << "cloud computation - per-patch : \n\t"
        << sum_time / num_idx / 2
        << "ns" << std::endl;
}

void eval_suit()
{
    eval_local();
    eval_cloud();
    eval_patch();
}