// Compile: g++ -Ofast -fopenmp DiskVec.cpp -o diskvec
#include "DiskVec.hpp"

// ------------------- Cross-Platform Memory Mapping -------------------
#if defined(_WIN32)   // Windows Implementation
  #include <windows.h>
  #undef byte
#else                   // Linux/Unix Implementation
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <algorithm>

using namespace std;
using namespace std::chrono;


// ----------------- Edge Structure -----------------
#pragma pack(push, 1)
struct Edge {
    int32_t candidate;
    float distance;
};
#pragma pack(pop)


// ----------------- MappedFile Class -----------------
class MappedFile {
public:
  void* ptr;
  size_t size;
  
#if defined(_WIN32)
  HANDLE hFile;
  HANDLE hMap;
#else
  int fd;
#endif

  MappedFile() : ptr(nullptr), size(0)
  {
#if defined(_WIN32)
    hFile = INVALID_HANDLE_VALUE;
    hMap = NULL;
#else
    fd = -1;
#endif
  }

  ~MappedFile() { close(); }

  // Open an existing file for mapping. If readWrite is true, open for read/write.
  bool open(const std::string &filename, size_t size, bool readWrite) {
    this->size = size;
#if defined(_WIN32)
    // Use ANSI version for simplicity.
    hFile = CreateFileA(filename.c_str(),
                        readWrite ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ,
                        0,
                        NULL,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        NULL);
    if(hFile == INVALID_HANDLE_VALUE) {
      cerr << "Error opening file: " << filename << endl;
      return false;
    }
    hMap = CreateFileMappingA(hFile,
                              NULL,
                              readWrite ? PAGE_READWRITE : PAGE_READONLY,
                              0,
                              0,
                              NULL);
    if(hMap == NULL) {
      cerr << "Error creating file mapping." << endl;
      CloseHandle(hFile);
      return false;
    }
    ptr = MapViewOfFile(hMap,
                        readWrite ? FILE_MAP_WRITE : FILE_MAP_READ,
                        0, 0, size);
    if(ptr == NULL) {
      cerr << "Error mapping view of file." << endl;
      CloseHandle(hMap);
      CloseHandle(hFile);
      return false;
    }
#else
    fd = ::open(filename.c_str(), readWrite ? O_RDWR : O_RDONLY);
    if(fd < 0) {
      perror("Error opening file");
      return false;
    }
    ptr = mmap(nullptr, size, readWrite ? (PROT_READ | PROT_WRITE) : PROT_READ,
               MAP_SHARED, fd, 0);
    if(ptr == MAP_FAILED) {
      perror("Error mapping file");
      ::close(fd);
      fd = -1;
      return false;
    }
#endif
    return true;
  }

  // Create (or overwrite) a file for mapping.
  bool create(const std::string &filename, size_t size) {
    this->size = size;
#if defined(_WIN32)
    hFile = CreateFileA(filename.c_str(),
                        GENERIC_READ | GENERIC_WRITE,
                        0,
                        NULL,
                        CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL,
                        NULL);
    if(hFile == INVALID_HANDLE_VALUE) {
      cerr << "Error creating file: " << filename << endl;
      return false;
    }
    hMap = CreateFileMappingA(hFile,
                              NULL,
                              PAGE_READWRITE,
                              0,
                              (DWORD)size,
                              NULL);
    if(hMap == NULL) {
      cerr << "Error creating file mapping." << endl;
      CloseHandle(hFile);
      return false;
    }
    ptr = MapViewOfFile(hMap,
                        FILE_MAP_WRITE,
                        0, 0, size);
    if(ptr == NULL) {
      cerr << "Error mapping view of file." << endl;
      CloseHandle(hMap);
      CloseHandle(hFile);
      return false;
    }
#else
    fd = ::open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if(fd < 0) {
      perror("Error creating file");
      return false;
    }
    // Set the file size.
    if(lseek(fd, size - 1, SEEK_SET) == -1) {
      perror("Error with lseek()");
      ::close(fd);
      return false;
    }
    if(write(fd, "", 1) != 1) {
      perror("Error writing last byte");
      ::close(fd);
      return false;
    }
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(ptr == MAP_FAILED) {
      perror("Error mapping file");
      ::close(fd);
      return false;
    }
#endif
    return true;
  }

  void close() {
#if defined(_WIN32)
    if(ptr) { UnmapViewOfFile(ptr); ptr = nullptr; }
    if(hMap) { CloseHandle(hMap); hMap = NULL; }
    if(hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
#else
    if(ptr && ptr != MAP_FAILED) { munmap(ptr, size); ptr = nullptr; }
    if(fd >= 0) { ::close(fd); fd = -1; }
#endif
  }
};


// --------------- Helper Functions ----------------
inline void compute_dirs_and_dist(int32_t p, int32_t c,
                                  const float* points,
                                  int32_t &c_p_dir,
                                  int32_t &p_c_dir,
                                  float &c_p_dist,
                                  int32_t DIM)
{
    float dist_sq = 0.0f;
    float diff;
    float max_abs = 0;
    float max_val = -numeric_limits<float>::infinity();
    int32_t max_idx = 0;
    const float* rowp = points + p * DIM;
    const float* rowc = points + c * DIM;
    for (int32_t i = 0; i < DIM; ++i) {
        diff = rowc[i] - rowp[i];
        dist_sq += diff * diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_val = diff;
            max_idx = i;
        } else if (-diff > max_abs) {
            max_abs = -diff;
            max_val = diff;
            max_idx = i;
        }
    }
    c_p_dist = std::sqrt(dist_sq);
    c_p_dir = 2 * max_idx + (max_val > 0 ? 0 : 1);
    p_c_dir = 2 * max_idx + (max_val > 0 ? 1 : 0);
}

inline float squared_distance(const float* a, const float* b, int32_t DIM) {
    float dist_sq = 0.0f;
    for (int32_t i = 0; i < DIM; ++i) {
        float diff = a[i] - b[i];
        dist_sq += diff * diff;
    }
    return std::sqrt(dist_sq);
}


// Random generator for candidate list.
random_device rd;
mt19937 gen(rd());
void compute_candidate_list(int32_t v_id, int32_t sqrt_size,
                            Edge* graph,
                            vector<int32_t>& out_candidates,
                            int32_t numVectors,
                            int32_t DIM)
{
    int32_t row_len = DIM * 2;
    vector<Edge> row(row_len);
    int32_t base_index = v_id * row_len;
    for (int32_t i = 0; i < row_len; ++i) {
        row[i] = graph[base_index + i];
    }
    // Sort rows by ascending distance.
    sort(row.begin(), row.end(),
         [](const Edge &a, const Edge &b) {
             return a.distance < b.distance;
         });
    out_candidates.clear();
    for (int32_t i = 0;
         i < static_cast<int32_t>(row.size()) &&
         out_candidates.size() < static_cast<size_t>(sqrt_size); ++i) {
        for (int32_t j = 0;
             j < static_cast<int32_t>(row.size()) &&
             out_candidates.size() < static_cast<size_t>(sqrt_size); ++j) {
            if (row[i].candidate > -1 &&
                graph[row[i].candidate * row_len + j].candidate > -1) {
                out_candidates.push_back(
                    graph[row[i].candidate * row_len + j].candidate);
            }
        }
    }
    // Pad with random indices if needed.
    while (out_candidates.size() < static_cast<size_t>(sqrt_size))
      out_candidates.push_back(gen() % numVectors);
}



// ----------------- DiskVec Implementation -----------------

// Constructor: receives initialization info and builds the index.
DiskVec::DiskVec(const std::string &vecFilename)
  : vectorsFilename(vecFilename), graphFilename("graph.bin"),
    metadataFilename("metadata.bin"),
    vectorsMap(new MappedFile()),
    graphMap(new MappedFile()),
    vectors(nullptr), graph(nullptr)
{
}

// Destructor: free allocated memory.
DiskVec::~DiskVec() {
  if (vectorsMap) { vectorsMap->close(); delete vectorsMap; }
  if (graphMap) { graphMap->close(); delete graphMap; }
}

// To create the DiskVec graph from mapped vectors
bool DiskVec::create(int32_t numVectors, int32_t DIM)
{
  this->numVectors = numVectors;
  this->DIM = DIM;

  // Calculate sizes.
  size_t vectorsSize = static_cast<size_t>(numVectors) * DIM * sizeof(float);
  size_t graphSize = static_cast<size_t>(numVectors) * (DIM * 2) *
                     sizeof(Edge);

  // Map the vectors file (read-only).
  if (!vectorsMap->open(vectorsFilename, vectorsSize, false)) {
    cerr << "Failed to open vectors file." << endl;
    return false;
  }
  vectors = reinterpret_cast<float*>(vectorsMap->ptr);

  // Create the graph file (read-write).
  if (!graphMap->create(graphFilename, graphSize)) {
    cerr << "Failed to create graph file." << endl;
    return false;
  }
  graph = reinterpret_cast<Edge*>(graphMap->ptr);

  // Initialize the graph: candidate=-1 and distance=infinity.
  #pragma omp parallel for
  for (int32_t i = 0; i < numVectors; ++i) {
    int32_t base = i * (DIM * 2);
    for (int32_t j = 0; j < DIM * 2; ++j) {
      graph[base + j].candidate = -1;
      graph[base + j].distance = std::numeric_limits<float>::infinity();
    }
  }

  // Build the graph index.
  int32_t iter = 0;
  bool global_improved = true;
  int32_t sqrt_size = static_cast<int>(std::sqrt(numVectors));
  sqrt_size = (sqrt_size > DIM * 2) ? sqrt_size : DIM * 2;

  auto start_time = high_resolution_clock::now();
  vector<vector<int32_t>> local_candidates(omp_get_max_threads());
  
  while (global_improved) {
    int improved_flag = 0;
    ++iter;
    cout << "Iteration: " << iter << endl;
    
    #pragma omp parallel for schedule(dynamic, 1000) reduction(|:improved_flag)
    for (int32_t v_id = 0; v_id < numVectors; ++v_id) {
        int32_t tid = omp_get_thread_num();
        vector<int32_t>& cand_list = local_candidates[tid];
        compute_candidate_list(v_id, sqrt_size, graph, cand_list,
                               numVectors, DIM);
        if (cand_list.empty())
            continue;
        for (int32_t cand : cand_list) {
            if (cand == v_id)
                continue;
            int32_t c_p_dir, p_c_dir;
            float c_p_dist;
            compute_dirs_and_dist(v_id, cand, vectors,
                                  c_p_dir, p_c_dir, c_p_dist, DIM);
            int32_t index_v = v_id * (DIM * 2) + p_c_dir;
            if (c_p_dist < graph[index_v].distance) {
                graph[index_v].candidate = cand;
                graph[index_v].distance = c_p_dist;
                improved_flag |= 1;
            }
            int32_t index_c = cand * (DIM * 2) + c_p_dir;
            if (c_p_dist < graph[index_c].distance) {
                graph[index_c].candidate = v_id;
                graph[index_c].distance = c_p_dist;
                improved_flag |= 1;
            }
        }
    }
    global_improved = (improved_flag != 0);
    cout << "After update, improved = " << global_improved << endl;
    if (!global_improved)
        break;
  }
  auto end_time = high_resolution_clock::now();
  duration<float> elapsed = end_time - start_time;
  cout << "Index built in " << elapsed.count() << " secs" << endl;

  // Write metadata in a binary format: [numVectors (int32_t)][DIM (int32_t)]
  ofstream metaOut(metadataFilename, ios::binary);
  if (!metaOut) {
    cerr << "Error opening metadata file for write." << endl;
    return false;
  }
  metaOut.write(reinterpret_cast<char*>(&numVectors), sizeof(int32_t));
  metaOut.write(reinterpret_cast<char*>(&DIM), sizeof(int32_t));
  metaOut.close();

  return true;
}

// To load a graph from mapped file
int DiskVec::load()
{
  // Read metadata.
  ifstream metaIn(metadataFilename, ios::binary);
  if (!metaIn) {
    cerr << "Error opening metadata file." << endl;
    return false;
  }
  metaIn.read(reinterpret_cast<char*>(&numVectors), sizeof(int32_t));
  metaIn.read(reinterpret_cast<char*>(&DIM), sizeof(int32_t));
  metaIn.close();

  // Map vectors file.
  size_t vectorsSize = static_cast<size_t>(numVectors) * DIM * sizeof(float);
  if (!vectorsMap->open(vectorsFilename, vectorsSize, false)) {
    cerr << "Failed to open vectors file in load()." << endl;
    return false;
  }
  vectors = reinterpret_cast<float*>(vectorsMap->ptr);

  // Map graph file.
  size_t graphSize = static_cast<size_t>(numVectors) * (DIM * 2) *
                     sizeof(Edge);
  if (!graphMap->open(graphFilename, graphSize, false)) {
    cerr << "Failed to open graph file in load()." << endl;
    return false;
  }
  graph = reinterpret_cast<Edge*>(graphMap->ptr);

  return this->numVectors;
}

// To search graph given a query
int32_t DiskVec::search(const vector<float>& query, const int mul)
{
  // Start from vector 0 (or random)
  int32_t current = 0;
  float current_best_dist = squared_distance(query.data(),
                                             vectors + current * DIM, DIM);
  int32_t best = current;
  float best_dist = current_best_dist;
  unordered_set<int32_t> visited;

  int32_t margin = DIM * mul;
  int32_t safety_steps = margin;
  int32_t row_len = DIM * 2;
  int32_t _current = current;
  while (safety_steps > 0) {
    safety_steps--;
    visited.insert(current);
    current_best_dist = numeric_limits<float>::infinity();
    _current = current;
    for (int32_t j = 0; j < row_len; ++j) {
      int32_t candidate = graph[_current * row_len + j].candidate;
      if (candidate == -1 || visited.count(candidate) > 0)
        continue;
      float d = squared_distance(query.data(),
                                 vectors + candidate * DIM, DIM);
      if (d < current_best_dist) {
        current_best_dist = d;
        current = candidate;
      }
      if (d < best_dist) {
        best_dist = d;
        best = candidate;
        safety_steps = margin;
      }
    }
  }
  return best;
}


int test() {
  // Filenames (adjust as needed).
  std::string vectorsFilename = "embeddings.dat";

  // For testing purposes, adjust number of vectors and dimension.
  int32_t numVectors = 10000;
  int32_t DIM = 128;

  DiskVec index(vectorsFilename);

  cout << "Creating index..." << endl;
  if (!index.create(numVectors, DIM)) {
    cerr << "Index creation failed." << endl;
    return EXIT_FAILURE;
  }

  cout << "\nLoading index..." << endl;
  if (!index.load()) {
    cerr << "Index loading failed." << endl;
    return EXIT_FAILURE;
  }

  // Testing greedy search.
  cout << "\nTesting greedy search with 1000 queries:" << endl;
  const int32_t NUM_QUERIES = 1000;
  int32_t num_correct = 0;
  double total_search_time_ms = 0.0;

  for (int32_t q = 0; q < NUM_QUERIES; q++) {
    // Generate a random query vector (values in [0,1]).
    vector<float> query(DIM);
    for (int32_t i = 0; i < DIM; i++) {
      query[i] = static_cast<float>(gen() % 100000) / 100000.0f;
    }
    auto t1 = chrono::high_resolution_clock::now();
    int32_t greedy_nn = index.search(query, 6);
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> search_time = t2 - t1;
    total_search_time_ms += search_time.count();

    // Ground truth via linear search.
    int32_t true_nn = -1;
    float best_dist = numeric_limits<float>::infinity();
    for (int32_t i = 0; i < numVectors; ++i) {
      float d = squared_distance(query.data(), index.vectors + i * DIM, DIM);
      if (d < best_dist) {
        best_dist = d;
        true_nn = i;
      }
    }
    cout << "Query " << q + 1 << ": Greedy NN = " << greedy_nn
         << ", True NN = " << true_nn
         << ", Time = " << search_time.count() << " ms" << endl;
    if (greedy_nn == true_nn)
      num_correct++;
  }
  float avg_search_time = total_search_time_ms / NUM_QUERIES;
  float accuracy = (static_cast<float>(num_correct) / NUM_QUERIES) * 100.0f;
  cout << "\nAverage search time: " << avg_search_time
       << " ms" << endl;
  cout << "Accuracy (exact match): " << accuracy << " %" << endl;

  // Memory-mapped files will be unmapped when DiskVec goes out of scope.
  return 0;
}