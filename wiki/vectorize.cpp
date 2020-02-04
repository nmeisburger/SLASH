#include <algorithm>
#include <bitset>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

using namespace std;

bool sort_by_key(const pair<unsigned int, unsigned int> a,
                 const pair<unsigned int, unsigned int> b) {
    return a.first < b.first;
}

class Unigram_Hasher {
  private:
    hash<string> hasher;
    unsigned int _ngrams;
    unsigned int _log_2_range;

  public:
    Unigram_Hasher(unsigned int ngrams, unsigned int log_2_range) {
        _ngrams = ngrams;
        _log_2_range = log_2_range;
    }

    void hash_line(string line, vector<pair<unsigned int, unsigned int>> *hashes) {
        map<unsigned, unsigned int> store;
        for (size_t i = 0; i < line.length() - _ngrams + 1; i++) {
            string unigram = line.substr(i, _ngrams);
            unsigned int hash = hasher(unigram) >> 1;
            unsigned int bucket = hash >> (31 - _log_2_range);
            auto value = store.find(bucket);
            if (value != store.end()) {
                if (value->second < hash) {
                    continue;
                }
            }
            store[bucket] = hash;
        }
        for (pair<unsigned int, unsigned int> entry : store) {
            hashes->push_back(entry);
        }
        sort(hashes->begin(), hashes->end(), sort_by_key);
    }

    void hash_file(string input_filename, string output_filename, unsigned int offset) {
        ifstream input;
        input.open(input_filename);
        ofstream output;
        output.open(output_filename);
        unsigned int n_lines;
        unsigned int max_length = 0;
        unsigned int total_dim = 0;

        vector<pair<unsigned int, unsigned int>> *line_hashes =
            new vector<pair<unsigned int, unsigned int>>();
        string line;
        while (getline(input, line)) {
            unsigned int id_loc = line.find_first_of(":");
            if (id_loc == string::npos || n_lines < offset) {
                continue;
            }
            n_lines++;
            string id = line.substr(0, id_loc);
            hash_line(line.substr(id_loc, line.length() - id_loc), line_hashes);
            unsigned int line_length = line_hashes->size();
            total_dim += line_length;
            if (line_length > max_length) {
                max_length = line_length;
            }
            output << id << " ";
            for (pair<unsigned int, unsigned int> entry : *line_hashes) {
                // output << entry.first << ":" << entry.second << " "; // For bucket:hash value
                output << entry.second << ":"
                       << "1 "; // For hash value:1
            }
            output << "\n";
            line_hashes->clear();
            if (n_lines % 10000 == 0) {
                printf("At %d\n", n_lines);
            }
        }
        printf("Read %u lines - Largest Dimension %u - Average Dimension %u\n", n_lines, max_length,
               (total_dim / n_lines));
        input.close();
        output.close();
    }
};

int main() {
    Unigram_Hasher h(5, 20);

    h.hash_file("wiki_paragraphs", "wiki_hashes", 0);

    return 0;
}