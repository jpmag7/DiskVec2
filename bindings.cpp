// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "DiskVec.hpp"

namespace py = pybind11;

PYBIND11_MODULE(DiskVec, m) {
    m.doc() = "Python bindings for the DiskVec module";

    py::class_<DiskVec>(m, "DiskVec")
        // Bind the constructor that takes (vecFilename)
        .def(py::init<const std::string&>(),
             py::arg("vecFilename"),
             "Construct DiskVec and create the index")
        // Expose the create() method
        .def("create", &DiskVec::create,
             py::arg("numVectors"),
             py::arg("DIM"),
             "Create the index in disk")
        // Expose the load() method
        .def("load", &DiskVec::load,
             "Load the index from disk (reads metadata and maps the files)")
        // Expose the search() method
        .def("search", &DiskVec::search,
             py::arg("query"),
             py::arg("mul"),
             "Perform a greedy search and return the best index");
}
