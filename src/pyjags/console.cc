// Copyright (C) 2015-2016 Tomasz Miasko
//               2020 Michael Nowotny
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

#include <Python.h>
#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Console.h>
#include <model/Model.h>
#include <rng/RNG.h>
#include <rng/RNGFactory.h>
#include <util/nainf.h>
#include <version.h>

#include <cstring>
#include <sstream>

namespace py = pybind11;

namespace jags {}

namespace {

using namespace jags;

// RAII style holder for FILE*.
class file_handle {
  FILE *file_;

  file_handle(const file_handle &) = delete;
  file_handle &operator=(const file_handle &) = delete;

public:
  // Takes ownership of given file handle if any.
  file_handle(FILE *file) : file_(file) {}

  // Close the file handle if any, ignoring errors in the process.
  ~file_handle() {
    if (file_)
      fclose(file_);
  }

  // Returns owned file or nullptr.
  FILE *file() const {
    return file_;
  }

  // Returns true when holding a file.
  explicit operator bool() {
    return file_;
  }
};

// Exception object used to report errors. Created during module initialization.
py::object JagsError;

// Converts numpy array to JAGS SArray.
SArray to_jags(py::object src) {
  // Ensure we have a source numpy array.
  const py::object src_array = py::reinterpret_steal<py::object>(
      PyArray_FromAny(src.ptr(), NULL, 1, 0, 0, 0));
  if (!src_array) {
    throw py::error_already_set();
  }
  PyArrayObject *src_numpy = (PyArrayObject *)src_array.ptr();
  const int ndim = PyArray_NDIM(src_numpy);
  npy_intp *dims = PyArray_DIMS(src_numpy);

  SArray dst{{dims, dims + ndim}};
  double *data = const_cast<double *>(dst.value().data());

  // Create numpy view onto destination SArray. Its elements are in fortran
  // order.
  py::object dst_array = py::reinterpret_steal<py::object>(
      PyArray_New(&PyArray_Type, ndim, dims, NPY_DOUBLE, NULL, data, 0,
                  NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_WRITEABLE, NULL));
  if (!dst_array) {
    throw py::error_already_set();
  }
  PyArrayObject *dst_numpy = (PyArrayObject *)dst_array.ptr();
  if (PyArray_CopyInto(dst_numpy, src_numpy) != 0) {
    throw py::error_already_set();
  }
  return dst;
}

// Converts JAGS SArray to numpy array.
py::array to_python(const SArray &sarray) {
  std::vector<npy_intp> dims{sarray.dim(false).begin(),
                             sarray.dim(false).end()};
  double *data = const_cast<double *>(sarray.value().data());

  // Creat a view over sarray data. Its elements are in fortran order.
  py::object view = py::reinterpret_steal<py::object>(
      PyArray_New(&PyArray_Type, dims.size(), dims.data(), NPY_DOUBLE, NULL,
                  data, 0, NPY_ARRAY_F_CONTIGUOUS, NULL));
  if (!view) {
    throw py::error_already_set();
  }

  return py::reinterpret_steal<py::object>(
      PyArray_NewCopy((PyArrayObject *)view.ptr(), NPY_ANYORDER));
}

// Converts Python dictionary to JAGS map.
std::map<std::string, SArray> to_jags(py::dict dictionary) {
  std::map<std::string, SArray> result;
  for (const auto &item : dictionary) {
    const std::string key = item.first.cast<std::string>();
    result.emplace(key,
                   to_jags(py::reinterpret_borrow<py::object>(item.second)));
  }
  return result;
}

// Converts JAGS map to Python dictionary.
py::dict to_python(const std::map<std::string, SArray> &map) {
  py::dict result;
  for (const auto &item : map) {
    result[item.first.c_str()] = to_python(item.second);
  }
  return result;
}

// Thin wrapper around Console class from JAGS.
class JagsConsole {
  std::stringstream out_stream_;
  std::stringstream err_stream_;
  Console console_;

  JagsConsole(const JagsConsole &) = delete;
  JagsConsole &operator=(const JagsConsole &) = delete;

  template <typename T> void invoke(const T &f) {
    out_stream_.str(std::string());
    err_stream_.str(std::string());
    out_stream_.clear();
    err_stream_.clear();

    bool success = f();
    // Reports error when f returned false or written something to error stream.
    if (!success || err_stream_.rdbuf()->in_avail()) {
      PyErr_SetString(JagsError.ptr(), err_stream_.str().c_str());
      throw py::error_already_set();
    }
  }

public:
  JagsConsole() : console_(out_stream_, err_stream_) {}

  void checkModel(const std::string &path) {
    file_handle fh(fopen(path.c_str(), "rb"));
    if (!fh) {
      PyErr_SetFromErrnoWithFilename(JagsError.ptr(), path.c_str());
      throw py::error_already_set();
    }
    invoke([&] { return console_.checkModel(fh.file()); });
  }

  void compile(const py::dict &data, unsigned int chains, bool generate_data) {
    auto jags_data = to_jags(data);
    invoke([&] { return console_.compile(jags_data, chains, generate_data); });
  }

  void setParameters(const py::dict &parameters, unsigned int chain) {
    invoke([&] { return console_.setParameters(to_jags(parameters), chain); });
  }

  void setRNGname(std::string const &name, unsigned int chain) {
    invoke([&] { return console_.setRNGname(name, chain); });
  }

  void initialize() {
    invoke([&] { return console_.initialize(); });
  }

  void update(unsigned int iterations) {
    invoke([&] {
      py::gil_scoped_release release;
      return console_.update(iterations);
    });
  }

  void setMonitor(const std::string &name, unsigned int thin,
                  const std::string &type) {
    invoke([&] { return console_.setMonitor(name, Range(), thin, type); });
  }

  void setMonitors(const std::vector<std::string> names, unsigned int thin,
                   const std::string &type) {
    for(auto&& name: names) {
        invoke([&] { return console_.setMonitor(name, Range(), thin, type); });
    }
  }

  void clearMonitor(const std::string &name, const std::string &type) {
    invoke([&] { return console_.clearMonitor(name, Range(), type); });
  }

  py::dict dumpState(DumpType type, unsigned int chain) {
    std::map<std::string, SArray> data;
    std::string rng_name;
    invoke([&] { return console_.dumpState(data, rng_name, type, chain); });
    py::dict result = to_python(data);
    if (!rng_name.empty()) {
      result[".RNG.name"] = py::cast(rng_name);
    }
    return result;
  }

  unsigned int iter() const {
    return console_.iter();
  }

  const std::vector<std::string> &variableNames() const {
    return console_.variableNames();
  }

  unsigned int nchain() const {
    return console_.nchain();
  }

  py::dict dumpMonitors(const std::string &type, bool flat) {
    std::map<std::string, SArray> data;
    invoke([&] { return console_.dumpMonitors(data, type, flat); });
    return to_python(data);
  }

  std::vector<std::vector<std::string>> dumpSamplers() {
    std::vector<std::vector<std::string>> samplers;
    invoke([&] { return console_.dumpSamplers(samplers); });
    return samplers;
  }

  void adaptOff() {
    invoke([&] { return console_.adaptOff(); });
  }

  bool checkAdaptation() {
    bool status = false;
    invoke([&] { return console_.checkAdaptation(status); });
    return status;
  }

  bool isAdapting() const {
    return console_.isAdapting();
  }

  void clearModel() {
    console_.clearModel();
  }

  static void loadModule(const std::string &name) {
    if (!Console::loadModule(name)) {
      PyErr_Format(JagsError.ptr(), "Error loading module: %s", name.c_str());
      throw py::error_already_set();
    }
  }

  static void unloadModule(const std::string &name) {
    if (!Console::unloadModule(name)) {
      PyErr_Format(JagsError.ptr(), "Error unloading module: %s", name.c_str());
      throw py::error_already_set();
    }
  }

  static std::vector<std::string> listModules() {
    return Console::listModules();
  }

  static std::vector<std::pair<std::string, bool>>
  listFactories(FactoryType type) {
    return Console::listFactories(type);
  }

  static void setFactoryActive(const std::string &name, FactoryType type,
                               bool active) {
    if (!Console::setFactoryActive(name, type, active)) {
      PyErr_Format(JagsError.ptr(),
                   "Error activating / deactivating factory: %s", name.c_str());
      throw py::error_already_set();
    }
  }

  static double na() {
    return JAGS_NA;
  }

  static const char *version() {
    return jags_version();
  }

  static py::list parallel_rngs(const std::string &factory,
                                unsigned int chains) {
    std::string error;
    std::vector<RNG *> rngs;

    const auto &factories = Model::rngFactories();
    for (const auto &f : factories) {
      if (f.first->name() == factory) {
        if (!f.second) {
          PyErr_Format(JagsError.ptr(), "RNG factory not active: %s",
                       factory.c_str());
          throw py::error_already_set();
        }
        rngs = f.first->makeRNGs(chains);
        break;
      }
    }

    if (rngs.empty()) {
      PyErr_Format(JagsError.ptr(), "RNG factory not found: %s",
                   factory.c_str());
      throw py::error_already_set();
    }

    py::list result;
    for (auto rng : rngs) {
      std::vector<int> state;
      rng->getState(state);
      py::dict d;
      d[".RNG.name"] = py::str(rng->name());
      d[".RNG.state"] = py::cast(state);
      result.append(d);
    }
    return result;
  }
};

bool import_numpy() {
  import_array1(false);
  return true;
}

} // namespace

PYBIND11_MODULE(console, module) {
  if (!import_numpy()) {
    throw py::error_already_set();
  }

  JagsError = py::reinterpret_borrow<py::object>(
      PyErr_NewException("console.JagsError", NULL, NULL));
  if (!JagsError) {
    throw py::error_already_set();
  }

  if (PyModule_AddObject(module.ptr(), "JagsError", JagsError.ptr())) {
    throw py::error_already_set();
  }

//  if (std::strcmp(PYJAGS_JAGS_VERSION, jags_version()) != 0) {
//    PyErr_Format(JagsError.ptr(),
//                 "Incompatible JAGS version. "
//                 "Compiled against version %s, but using version %s.",
//                 PYJAGS_JAGS_VERSION, jags_version());
//    throw py::error_already_set();
//  }

  py::enum_<DumpType>(module, "DumpType",
                      "Flags for the function Console#dump_state")
      .value("DUMP_DATA", DUMP_DATA)
      .value("DUMP_PARAMETERS", DUMP_PARAMETERS)
      .value("DUMP_ALL", DUMP_ALL)
      .export_values();

  py::enum_<FactoryType>(module, "FactoryType",
                         "Enumerates factory types in a model")
      .value("SAMPLER_FACTORY", SAMPLER_FACTORY)
      .value("MONITOR_FACTORY", MONITOR_FACTORY)
      .value("RNG_FACTORY", RNG_FACTORY)
      .export_values();

  py::class_<JagsConsole>(module, "Console",
                          "Low-level wrapper around JAGS Console class.")
      .def(py::init<>())
      .def("checkModel", &JagsConsole::checkModel, py::arg("path"),
           "Load the model from a file and checks its syntactic correctness.")
      .def("compile", &JagsConsole::compile, py::arg("data"), py::arg("chains"),
           py::arg("generate_data"), "Compiles the model.")
      .def("setParameters", &JagsConsole::setParameters, py::arg("parameters"),
           py::arg("chain"),
           "Sets the parameters (unobserved variables) of the model.")
      .def("setRNGname", &JagsConsole::setRNGname, py::arg("name"),
           py::arg("chain"), "Sets the name of the RNG for the given chain.")
      .def("initialize", &JagsConsole::initialize, "Initializes the model.")
      .def("update", &JagsConsole::update, py::arg("iterations"),
           "Updates the Markov chain generated by the model.")
      .def("setMonitor", &JagsConsole::setMonitor, py::arg("name"),
           py::arg("thin"), py::arg("type"),
           "Sets a monitor for the given node array.")
      .def("setMonitors", &JagsConsole::setMonitors, py::arg("names"),
           py::arg("thin"), py::arg("type"),
           "Sets multiple monitors for the given node array.")
      .def("clearMonitor", &JagsConsole::clearMonitor, py::arg("name"),
           py::arg("type"), "Clears a monitor.")
      .def("dumpState", &JagsConsole::dumpState, py::arg("type"),
           py::arg("chain"), "Dumps the state of the model.")
      .def("iter", &JagsConsole::iter,
           "Returns the iteration number of the model.")
      .def("variableNames", &JagsConsole::variableNames,
           "Returns a vector of variable names used by the model.")
      .def("nchain", &JagsConsole::nchain,
           "Returns the number of chains in the model.")
      .def("dumpMonitors", &JagsConsole::dumpMonitors, py::arg("type"),
           py::arg("flat"), "Dumps the contents of monitors.")
      .def("dumpSamplers", &JagsConsole::dumpSamplers,
           "Dumps the names of the samplers, and the corresponding sampled "
           "nodes vectors")
      .def("adaptOff", &JagsConsole::adaptOff,
           "Turns off adaptive mode of the model.")
      .def("checkAdaptation", &JagsConsole::checkAdaptation,
           "Checks whether adaptation is complete.")
      .def("isAdapting", &JagsConsole::isAdapting,
           "Indicates whether model is in adaptive mode.")
      .def("clearModel", &JagsConsole::clearModel, "Clears the model.")
      .def_static("loadModule", &JagsConsole::loadModule, py::arg("name"),
                  "Loads a module by name")
      .def_static("unloadModule", &JagsConsole::unloadModule, py::arg("name"),
                  "Unloads a module by name")
      .def_static("listModules", &JagsConsole::listModules,
                  "Returns a list containing the names of loaded modules.")
      .def_static("listFactories", &JagsConsole::listFactories, py::arg("type"),
                  "Returns a list containing the names of currently loaded "
                  "factories and whether or not they are active.")
      .def_static("setFactoryActive", &JagsConsole::setFactoryActive,
                  py::arg("name"), py::arg("type"), py::arg("active"),
                  "Sets a factory to be active or inactive")
      // TODO move outside of Console
      .def_static("na", &JagsConsole::na, "Return value of JAGS_NA.")
      .def_static("version", &JagsConsole::version,
                  "Return version of JAGS library.")
      .def_static("parallel_rngs", &JagsConsole::parallel_rngs,
                  "RNGs for execution in parallel.");
}
