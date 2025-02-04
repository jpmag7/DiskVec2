#ifndef DISK_VEC_HPP
#define DISK_VEC_HPP

#include <string>
#include <vector>

class MappedFile;
struct Edge;

// The DiskVec class definition.
class DiskVec {
public:
  // Metadata.
  int32_t numVectors;
  int32_t DIM;

  // File names.
  std::string vectorsFilename;
  std::string graphFilename;
  std::string metadataFilename;

  // Memory-mapped files.
  MappedFile* vectorsMap;
  MappedFile* graphMap;

  // Pointers to mapped memory.
  float* vectors;
  Edge* graph;

  // Constructor with initialization parameters.
  DiskVec(const std::string &vecFilename);

  // Default destructor to free memory.
  ~DiskVec();

  // Create the index file from a given vectors file.
  bool create(int32_t numVectors, int32_t DIM);

  // Load metadata and map files.
  int load();

  // Greedy search: returns index of the nearest neighbor.
  int32_t search(const std::vector<float>& query, const int mul);
};

#endif
