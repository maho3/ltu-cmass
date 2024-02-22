#include <omp.h>

class TaskDistributor {
    // Class for distributing parallel tasks
	public :
	TaskDistributor(std::uint64_t t_total_num_tasks,  std::uint64_t t_start = 0,  std::uint64_t t_num_threads = 0) {
		t_num_threads = (t_num_threads == 0) ? static_cast<std::uint64_t>(omp_get_max_threads()) : t_num_threads;
		const std::uint64_t final_task = t_start + t_total_num_tasks;
		const std::uint64_t id = static_cast<std::uint64_t>(omp_get_thread_num());
		const std::uint64_t remainder = t_total_num_tasks % t_num_threads;
		num_tasks = (id < remainder) ? t_total_num_tasks / t_num_threads + 1 : t_total_num_tasks / t_num_threads;
		start = (id < remainder) ? t_start + id * num_tasks : t_start + id * num_tasks + remainder;
		stop = std::min(start + num_tasks, final_task);
		worker = (start < final_task) ? true : false;
		first_worker = (id == 0) ? true : false;
		last_worker = (worker && stop == final_task) ? true : false;
	};
	bool worker, first_worker, last_worker;
	std::uint64_t num_tasks, start, stop;
};
