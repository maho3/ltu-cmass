#include <vector>
#include <complex>
#include <fftw3.h>
#include <algorithm>

void transformInverseDFT3D(std::complex<double> * t_modes, double * t_mesh, std::uint64_t t_num_mesh_1d, double t_normalization = 1.) {
	// inverse Fourier transforms 3D complex hermitian modes to a real mesh
	fftw_init_threads();
	const std::uint32_t num_threads = omp_get_max_threads();
	fftw_plan_with_nthreads(num_threads);
	fftw_complex * modes = reinterpret_cast<fftw_complex *>(t_modes);
	fftw_plan dft_plan = fftw_plan_dft_c2r_3d(t_num_mesh_1d, t_num_mesh_1d, t_num_mesh_1d, modes, t_mesh, FFTW_ESTIMATE);
	fftw_execute(dft_plan);
	fftw_destroy_plan(dft_plan);
	fftw_cleanup_threads();
	if (t_normalization != 1.) {
        const std::uint64_t num_modes = (t_num_mesh_1d / 2 + 1) * t_num_mesh_1d * t_num_mesh_1d;
		#pragma omp  parallel for
		for (std::uint64_t i = 0; i < 2 * num_modes; i++) {
			t_mesh[i] /= t_normalization;
		}
	}
	return;
}

void transformInverseDFT3D(std::vector<double> & t_mesh, double t_normalization = 1.) {
	// inverse Fourier transforms a 3D real mesh that was previous Fourier transformed in place or otherwise set up as a real space array
	std::uint64_t num_mesh_1d = static_cast<std::uint64_t>(floor(pow(t_mesh.size(), 1. / 3.)));
	double * mesh = static_cast<double *>(t_mesh.data());
	std::complex<double> * modes = reinterpret_cast<std::complex<double> *>(t_mesh.data());
	transformInverseDFT3D(modes, mesh, num_mesh_1d, 1.);
	if (t_normalization != 1.) {
		#pragma omp parallel
        {
            TaskDistributor tasks(t_mesh.size());
            std::for_each(t_mesh.begin() + tasks.start, t_mesh.begin() + tasks.stop, [t_normalization](double & t_m){t_m /= t_normalization;});
		}
	}
	return;
}

void unformatDFTMesh(std::vector<double> & t_mesh) {
    // Format padded DFT structured array to unpadded array
	const std::uint64_t num_mesh_1d = static_cast<std::uint64_t>(floor(pow(t_mesh.size(), 1./3.)));
	const std::uint64_t two_num_modes_last_d = 2 * (num_mesh_1d / 2 + 1);
	std::vector<double> mesh_2(num_mesh_1d * num_mesh_1d * num_mesh_1d);
    #pragma omp parallel
    {
        TaskDistributor tasks(num_mesh_1d * num_mesh_1d);
        std::vector<double>::iterator source_start = t_mesh.begin() + two_num_modes_last_d * tasks.start;
        std::vector<double>::iterator source_stop = t_mesh.begin() + two_num_modes_last_d * tasks.stop;
        std::vector<double>::iterator destination = mesh_2.begin() + num_mesh_1d * tasks.start;
        while (source_start != source_stop) {
            std::copy(source_start, source_start + num_mesh_1d, destination);
            std::advance(source_start, two_num_modes_last_d);
            std::advance(destination, num_mesh_1d);
        }
    }
    t_mesh.resize(mesh_2.size());
    #pragma omp parallel
    {
        TaskDistributor tasks(num_mesh_1d * num_mesh_1d);
        std::vector<double>::iterator source_start = mesh_2.begin() + num_mesh_1d * tasks.start;
        std::vector<double>::iterator source_stop = mesh_2.begin() + num_mesh_1d * tasks.stop;
        std::vector<double>::iterator destination = t_mesh.begin() + num_mesh_1d * tasks.start;
        while (source_start != source_stop) {
            std::copy(source_start, source_start + num_mesh_1d, destination);
            std::advance(source_start, num_mesh_1d);
            std::advance(destination, num_mesh_1d);
        }
    }
    return;
}



