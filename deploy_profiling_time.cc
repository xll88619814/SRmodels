#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/profiling.h>
#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <errno.h>
#include <exception>

bool GetImageData(std::string file_path, float *data, uint32_t size) {
  try {
    std::ifstream file;
    file.exceptions(file.failbit | file.badbit);
    file.open(file_path);
    if (file.is_open()) {
      float tmp;
      uint32_t count = 0;
      while (file >> tmp) {
        *data++ = tmp;
        count++;
        if (count >= size) {
          break;
        }
      }
      file.close();
      return true;
    }
  } catch (std::ifstream::failure &e) {
    std::cout << "open file failed: " << e.what() << std::endl;
    return false;
  }
  return false;
}

bool SaveOutput(const std::string &file_path, float *data, uint32_t size) {
  try {
    std::ofstream out_file;
    out_file.exceptions(out_file.failbit | out_file.badbit);
    out_file.open(file_path);
    if (out_file.is_open()) {
      float tmp;
      for (uint32_t i = 0; i < size; ++i) {
        tmp = *data++;
        out_file << tmp;
        out_file << '\n';
      }
      out_file.close();
      return true;
    }
  } catch (std::ifstream::failure &e) {
    std::cout << "open file failed: " << e.what() << std::endl;
    return false;
  }
  return false;
}

// void DeployResnet50(std::string in_file, std::string out_file, std::string dev_type = "cpu") {
void DeployResnet50(std::string dev_type = "cpu") {
  LOG(INFO) << "Running C++ resnet50 graph executor...";

  float *data = new float[1 * 3 * 1080 * 1920];

  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory;
  // tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("data/model/sr_sim_batchsize_1_fp16.so");

  // load engine from file
  if (dev_type == "cpu") {
    dev.device_type = kDLCPU;
    mod_factory = tvm::runtime::Module::LoadFromFile("data/model/sr_sim_batchsize_1_llvm.so");
  } else if (dev_type == "igie") {
    dev.device_type = kDLILUVATAR;
    mod_factory = tvm::runtime::Module::LoadFromFile("data/model/sr_sim_batchsize_1_fp16.so");
  }else {
    std::cout << "ERROR: unrecognized device type!" << std::endl;
    return;
  }

  // create the graph executor module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc get_graph_json = mod_factory.GetFunction("get_graph_json");
  std::string json = get_graph_json();
  // std::cout << json  << std::endl;
  
  //获取模型输入个数
  tvm::runtime::PackedFunc get_num_inputs = gmod.GetFunction("get_num_inputs");
  int input_num = get_num_inputs();
  std::cout << "get_num_inputs: " << input_num << std::endl;

  //获取模型输出个数
  tvm::runtime::PackedFunc get_num_outputs = gmod.GetFunction("get_num_outputs");
  int output_num = get_num_outputs();
  std::cout << "get_num_outputs: " << output_num << std::endl;
  
  // 获取模型全部输入names
  tvm::runtime::PackedFunc get_input_names = gmod.GetFunction("get_input_names");
  void* input_names = get_input_names();

  std::vector<std::string>* vector_names = static_cast<std::vector<std::string>*>(input_names);
  for (const auto& str : *vector_names) {
    std::cout << "input_name: " << str << std::endl;
  }

  // 获取模型全部输出names
  tvm::runtime::PackedFunc get_output_names = gmod.GetFunction("get_output_names");
  void* output_names = get_output_names(); 

  vector_names = static_cast<std::vector<std::string>*>(output_names);
  for (const auto& str : *vector_names) {
    std::cout << "output_name: " << str << std::endl;
  }

  tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
  tvm::runtime::NDArray input_ = get_input(0);
  tvm::runtime::ShapeTuple input_shape = input_.Shape();
  tvm::runtime::DataType input_type = input_.DataType();
  std::cout << "Input Shape: [";
  for (int i=0; i< input_shape.size(); i++){
    std::cout << input_shape.data()[i]  << ",";
  }
  std::cout << "]" << std::endl;
  std::string dtype =  tvm::runtime::DLDataType2String(input_type);
  std::cout << "Input Type: " << dtype << std::endl;

  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  
  // Use the C++ API
  tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty({1, 3, 1080, 1920}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray output_gpu = tvm::runtime::NDArray::Empty({1, 3, 2160, 3840}, DLDataType{kDLFloat, 32, 1}, dev);

  // if (!GetImageData(in_file, data, 1*3*1080*1920)) {
  //   std::cout << "Get data failed!" << std::endl;
  //   return;
  // }

  // copy data from cpu to gpu
  input.CopyFromBytes(data, 1 * 3 * 1080 * 1920 * sizeof(float));

  //  copy data from gpu to cpu(D2H)
  float *output = (float*)malloc(24883200 * sizeof(float));
  // output_gpu.CopyToBytes((void*)output, 24883200 * sizeof(float));
  

  // warm-up
  for (int i = 0; i < 3; i++){
    set_input(0, input);
    run();
  }

  
  for (int i = 0; i < 10; i++){

    tvm::runtime::Timer t = tvm::runtime::Timer::Start(dev);
    
    // set input
    set_input(0, input);
    
    // run model
    run();
    t->Stop();
    int64_t t_nanos = t->SyncAndGetElapsedNanos();
    double duration_ms = t_nanos / 1e6;
    double FPS = 1000 / duration_ms;
    std::cout << "Model Run Time: " << duration_ms << "ms" << std::endl;
    std::cout << "FPS: " << FPS << std::endl;

    // get the output
    tvm::runtime::NDArray output_gpu;
    output_gpu = get_output(0);
  
    tvm::runtime::Timer t1 = tvm::runtime::Timer::Start(dev);
    output_gpu.CopyToBytes((void*)output, 24883200 * sizeof(float));
    t1->Stop();
    int64_t t1_nanos = t1->SyncAndGetElapsedNanos();
    double duration1_ms = t1_nanos / 1e6;
    std::cout << "Copy output from gpu to host Time: " << duration1_ms << "ms" << std::endl;
  }

  for (int32_t i = 0; i < 10; ++i) {
      std::cout << ((float*)output)[i] << '\t';
  }
  std::cout << std::endl;

  // 获取output shape和dtype
  tvm::runtime::Array<tvm::runtime::NDArray> outputs;
  DLDevice cpu_dev{kDLCPU, 0};
  for (int i = 0; i < output_num; i++) {
     tvm::runtime::NDArray out = get_output(i);
     tvm::runtime::NDArray a = tvm::runtime::NDArray::Empty(out.Shape(), out.DataType(), cpu_dev);
     a.CopyFrom(out);
     outputs.push_back(a);
  }
  for (int i = 0; i < outputs.size(); i++) {
    for (int32_t j = 0; j < 10; ++j) {
      std::cout << ((float*)outputs[i]->data)[j] << '\t';
    }
    std::cout << std::endl;
  }

  // save inference output to postprecess
  // SaveOutput(out_file, output, 1000);

  if (data != nullptr) {
    LOG(INFO) << "free data when device type is cuda!!!";
    delete[] data;
  }
  LOG(INFO) << "End running resnet50 graph executor...";
}


int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "ERROR: two arguments for main, but get " << argc - 1 << std::endl;
    return 1;
  }
  // std::string in_file(argv[1]);
  // std::string out_file(argv[2]);
  std::string dev_type(argv[1]);

  // std::cout << "in_file: " << in_file << ", out_file: " << out_file << std::endl;
  // DeployResnet50(in_file, out_file, dev_type);
  DeployResnet50(dev_type);
  return 0;
}
