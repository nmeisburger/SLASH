#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

void count(string filename, unsigned int n, unsigned int offset) {

    ifstream file(filename);
    string str;

    size_t ct = 0;                  // Counting the input vectors.
    size_t totalLen = 0;            // Counting all the elements.
    while (std::getline(file, str)) // Get one vector (one vector per line).
    {
        if (ct < offset) { // If reading with an offset, skip < offset vectors.
            ct++;
            continue;
        }
        // Constructs an istringstream object iss with a copy of str as content.
        istringstream iss(str);
        // Removes label.
        string sub;
        iss >> sub;
        int pos;
        float val;
        do {
            std::string sub;
            iss >> sub;

            totalLen++;
        } while (iss);

        ct++;
        if (ct == (offset + n)) {
            break;
        }
    }

    printf("Read %u lines - total dim = %d\n", ct, totalLen);
}

int main() {
    count("../../dataset/criteo/criteo_tb", 1000000, 0);
    return 0;
}
