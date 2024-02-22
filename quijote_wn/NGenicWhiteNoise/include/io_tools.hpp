#include <fstream>

void assert(bool t_condition, std::string t_error_message, std::uint64_t t_code = 1) {
	if (!t_condition) {
		std::cerr << "\nError: " << t_error_message << std::endl;
		exit(t_code);
	}
	return;
};

bool checkFileIsWritable(std::string t_filename) {
	std::ofstream write_file(t_filename, std::ios::binary);
	return write_file.is_open();
}

template <typename t_type_in, typename t_type_out = t_type_in>
void writeVector(std::ofstream & t_write_file, std::vector<t_type_in> & t_vector) {
    // writes a c++ vector to binary file with a header indicating the number of t_type_out written
	const std::uint32_t block_size = t_vector.size();
	t_write_file.write(reinterpret_cast<const char *>(&block_size), sizeof(std::uint32_t));
	if constexpr(std::is_same_v<t_type_in, t_type_out>)
		t_write_file.write(reinterpret_cast<const char *>(t_vector.data()), block_size * sizeof(t_type_in));
	else {
		std::vector<t_type_out> out_vector(t_vector.begin(), t_vector.end());
		t_write_file.write(reinterpret_cast<const char *>(out_vector.data()), block_size * sizeof(t_type_out));
	}
	return;
};

std::ofstream openWriteFile(std::string t_filename, std::string t_error_message = "could not open file for writing") {
	std::ofstream write_file(t_filename, std::ios::binary);
    assert(write_file.is_open(), t_error_message);
	return write_file;
}

template <typename t_type_in, typename t_type_out = t_type_in>
void writeVector(std::string t_filename, std::vector<t_type_in> & t_vector) {
    // writes a c++ vector to binary file named t_filename with a header indicating the number of t_type_out written
	std::ofstream write_file = openWriteFile(t_filename);
	writeVector<t_type_in, t_type_out>(write_file, t_vector);
	write_file.close();
	return;
};
