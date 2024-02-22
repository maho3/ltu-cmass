#include <iostream>
#include <chrono>
#include <cmath>

class Timer {

	public:

    Timer(std::uint64_t t_max_message_length = 0) {
        start_time = std::chrono::steady_clock::now();
        last_time = std::chrono::steady_clock::now();
        max_message_length = t_max_message_length;
    }

	void setMaxMessageLength(std::uint64_t t_max_message_length) {max_message_length = t_max_message_length;};

    void printStart(std::string t_message) {
        current_message_length = t_message.size();
        std::cout << t_message << std::flush;
        return;
    }

    void printDone(std::uint64_t t_num_spaces = 0) {
        std::chrono::steady_clock::time_point this_time = std::chrono::steady_clock::now();
        double step_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(this_time - last_time).count()) * 1.e-3;
        double cumulative_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(this_time - start_time).count()) * 1.e-3;
        std::uint64_t step_time_h = static_cast<std::uint64_t>(floor(step_time / 60. / 60.));
        std::uint64_t step_time_m = static_cast<std::uint64_t>(floor(step_time / 60. - step_time_h * 60));
        double step_time_s = step_time - step_time_m * 60. - step_time_h * 60. * 60.;
        std::uint64_t cumulative_time_h = static_cast<std::uint64_t>(floor(cumulative_time / 60. / 60.));
        std::uint64_t cumulative_time_m = static_cast<std::uint64_t>(floor(cumulative_time / 60. - cumulative_time_h * 60));
        double cumulative_time_s = cumulative_time - cumulative_time_m * 60. - cumulative_time_h * 60. * 60.;
        if (max_message_length == 0 || t_num_spaces != 0) {
            for (std::uint64_t i = 0; i < t_num_spaces; i++)
                std::cout << " ";
        }
        else if (max_message_length > current_message_length) {
            for (std::uint64_t i =0; i < max_message_length - current_message_length; i++)
                std::cout << " ";
        }
        else
            std::cout << " ";
        std::cout << "done ";
        printf("[%02lu:%02lu:%05.2f, %02lu:%02lu:%05.2f]\n", step_time_h, step_time_m, step_time_s, cumulative_time_h, cumulative_time_m, cumulative_time_s);
        last_time = this_time;
        return;
    }

	private:
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point last_time;
	std::uint64_t max_message_length;
	std::uint64_t current_message_length;
};
