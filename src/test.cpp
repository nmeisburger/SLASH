#include <fstream>
#include <iostream>
#include <sstream>

void check(std::string filename) {
    std::ifstream file(filename);
    std::string s;
    while (std::getline(file, s)) {
        std::cout << s << std::endl;
    }
}

size_t read(std::string filename, size_t seek) {
    std::ifstream file(filename);
    file.seekg(seek);
    std::string s;
    int ct = 0;
    // while (std::getline(file, s)) {
    //     std::cout << s << std::endl;
    //     ct++;
    //     if (ct == 2)
    //         break;
    // }
    std::getline(file, s);
    std::cout << s << std::endl;
    std::getline(file, s);
    std::cout << s << std::endl;
    std::cout << "done" << std::endl;
    size_t x = file.tellg();
    file.close();
    return x;
}

void write(std::string filename, int x) {
    std::ofstream file(filename, std::ios::app);
    file << x << " " << x << "\n";
    file.close();
}

// int main() {
//     std::string x("./txt");
//     write(x, 2);
//     write(x, 3);
//     return 0;
// }