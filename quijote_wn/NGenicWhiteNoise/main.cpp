#include <timer.hpp>
#include <task_distributor.hpp>
#include <dft_tools.hpp>
#include <io_tools.hpp>
#include <gsl/gsl_rng.h>

int main(int argc, char ** argv) {

    if(argc < 6) {
        std::cout << "Usage: ngenic_white_noise num_mesh_1d num_part_1d random_seed white_noise_filename num_threads\n";
        exit(0);
    }

    const std::uint64_t num_mesh_1d = std::stoull(argv[1]);
    const std::uint64_t half_num_mesh_1d = num_mesh_1d / 2;
    const std::uint64_t num_modes_last_d = half_num_mesh_1d + 1;
    const std::uint64_t num_modes = num_mesh_1d * num_mesh_1d * num_modes_last_d;
    std::vector<bool> skip_mode(num_mesh_1d);
    {
        const std::uint64_t num_part_1d = std::stoull(argv[2]);
        const std::uint64_t half_num_part_1d = num_part_1d / 2;
        for (std::uint64_t i = 0; i < num_mesh_1d; i++)
            skip_mode[i] = (i < half_num_mesh_1d) ? (i > half_num_part_1d) : (num_mesh_1d - i > half_num_part_1d);
        skip_mode[half_num_mesh_1d] = true;
    }
    const std::uint32_t random_seed = std::stoul(argv[3]);
    std::string white_noise_filename = argv[4];
    assert(checkFileIsWritable(white_noise_filename), "Error: could not open white_noise_filename: " + white_noise_filename + " for writing", 1);
    const std::uint32_t num_threads = std::stoull(argv[5]);
    omp_set_num_threads(num_threads);
    std::cout << "Running ngenic_white_noise with " + std::to_string(num_threads) + " threads\n";
    Timer timer(std::string("Inverse Fourier transforming... ").size());

    // *** Allocate and draw the random seed table for each column of the mesh
    timer.printStart("Drawing white noise...");
    std::vector<std::uint32_t> seed_table(num_mesh_1d * num_mesh_1d);
    {
        gsl_rng * random_seed_generator = gsl_rng_alloc(gsl_rng_ranlxd1);
        gsl_rng_set(random_seed_generator, random_seed);
        std::uint64_t i, j;
        for(i = 0; i < half_num_mesh_1d; i++) {
            for(j = 0; j < i; j++)
                seed_table[i * num_mesh_1d + j] = 0x7fffffff * gsl_rng_uniform(random_seed_generator);
            for(j = 0; j < i + 1; j++)
                seed_table[j * num_mesh_1d + i] = 0x7fffffff * gsl_rng_uniform(random_seed_generator);
            for(j = 0; j < i; j++)
                seed_table[(num_mesh_1d - 1 - i) * num_mesh_1d + j] = 0x7fffffff * gsl_rng_uniform(random_seed_generator);
            for( j = 0; j < i + 1; j++)
                seed_table[(num_mesh_1d - 1 - j) * num_mesh_1d + i] = 0x7fffffff * gsl_rng_uniform(random_seed_generator);
            for(j = 0; j < i; j++)
                seed_table[i * num_mesh_1d + (num_mesh_1d - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_seed_generator);
            for(j = 0; j < i + 1; j++)
                seed_table[j * num_mesh_1d + (num_mesh_1d - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_seed_generator);
            for(j = 0; j < i; j++)
                seed_table[(num_mesh_1d - 1 - i) * num_mesh_1d + (num_mesh_1d - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_seed_generator);
            for(j = 0; j < i + 1; j++)
                seed_table[(num_mesh_1d - 1 - j) * num_mesh_1d + (num_mesh_1d - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_seed_generator);
        }
        gsl_rng_free(random_seed_generator);
    }

    // *** Allocate white noise fftw vector and initialize to zero
    std::vector<std::complex<double>> white_noise_modes(num_modes);
    #pragma omp parallel
    {
        TaskDistributor tasks(white_noise_modes.size());
        std::fill(white_noise_modes.begin() + tasks.start, white_noise_modes.begin() + tasks.stop, 0);
    }

    // *** Draw white noise modes in Fourier space
    {
        #pragma omp parallel
        {
            gsl_rng * thread_random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);
            double amplitude, phase;
            const double two_pi = 2. * acos(-1.);
            std::uint64_t i, j, k, index, conj_index;
            #pragma omp for collapse(2)
            for (i = 0; i < num_mesh_1d; i++) {
                for (j = 0; j < num_mesh_1d; j++) {
                    gsl_rng_set(thread_random_generator, seed_table[j + num_mesh_1d * i]);		  
                    for (k = 0; k < half_num_mesh_1d; k++) {
                        phase = two_pi * gsl_rng_uniform(thread_random_generator);
                        do
                            amplitude = gsl_rng_uniform(thread_random_generator);
                        while (amplitude == 0.);
                        amplitude = sqrt(-log(amplitude));
                        if (skip_mode[i] || skip_mode[j] || skip_mode[k])
                            continue;
                        index = k + num_modes_last_d * (j + num_mesh_1d * i);
                        if(k == 0) {
                            if (index == 0)
                                continue;
                            if (i == 0) {
                                if (j > half_num_mesh_1d)
                                    continue;
                                else
                                    conj_index = num_modes_last_d * (num_mesh_1d - j);
                            }
                            else {
                                if(i > half_num_mesh_1d)
                                    continue;
                                else
                                    conj_index = num_modes_last_d * (((num_mesh_1d - j) % num_mesh_1d) + num_mesh_1d * (num_mesh_1d - i));
                            }
                            white_noise_modes[conj_index] = std::complex<double>(cos(phase), -sin(phase)) * amplitude;
                        }
                        white_noise_modes[index] = std::complex<double>(cos(phase), sin(phase)) * amplitude;
                    }
                }
            }
            gsl_rng_free(thread_random_generator);
        }
    }
    timer.printDone();

    timer.printStart("Saving...");
    writeVector(white_noise_filename, white_noise_modes);
    timer.printDone();

    return 0;

}
