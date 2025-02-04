// Compile: g++ -Ofast -fopenmp encode.cpp DiskVec.cpp -o diskvec
#include "DiskVec.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

using namespace std;

int main(int argc, char* argv[]) {
    if(argc != 4) {
        cerr << "Usage: " << argv[0] << " <vectorsFilename> <numVectors> <DIM>" << endl;
        return EXIT_FAILURE;
    }

    // Get command line arguments.
    string vectorsFilename = argv[1];
    int32_t numVectors = std::atoi(argv[2]);
    int32_t DIM = std::atoi(argv[3]);

    // Create DiskVec instance.
    DiskVec index(vectorsFilename);

    cout << "Creating index..." << endl;
    if(!index.create(numVectors, DIM)) {
        cerr << "Index creation failed." << endl;
        return EXIT_FAILURE;
    }

    cout << "Index creation succeeded." << endl;
    return EXIT_SUCCESS;
}
