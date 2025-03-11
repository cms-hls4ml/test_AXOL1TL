#include <string>
#include <array>
#include "ap_fixed.h"
#include "emulator.h"

int main()
{
  std::string modelName = "GTADModel_v5";
  hls4mlEmulator::ModelLoader loader = hls4mlEmulator::ModelLoader(modelName);
  std::shared_ptr<hls4mlEmulator::Model> model = loader.load_model();

  std::cout << "modelInput = [";
  ap_fixed<18,13> modelInput[57];
  for (int i = 0; i < 57; i++) {
    modelInput[i] = 10.25;
    std::cout << modelInput[i] << ", ";
  }
  std::cout << "]" << std::endl;
  
  ap_fixed<18,14,AP_RND_CONV,AP_SAT> modelResult; //changed for v5
  // std::array<ap_fixed<18,14,AP_RND_CONV,AP_SAT>,1> modelResult; //changed for v5
  // for (int i = 0; i < 13; i++) {
  // for (int i = 0; i < 1; i++) {
  //   modelResult[i] = -1;
  // }
  ap_ufixed<18,14> modelLoss;

  model->prepare_input(modelInput);
  model->predict();

  auto pairResult = std::make_pair(modelResult, modelLoss);
  // std::pair<std::array<ap_fixed<18,14,AP_RND_CONV,AP_SAT>, 1>, ap_ufixed<18,14>> pairResult;
  // pairResult.first.fill(-1);
  // pairResult.second = 0;
  // std::any any_pairResult = &pairResult;
  // std::any any_pairResult = std::make_any<std::pair<std::array<ap_fixed<18,14,AP_RND_CONV,AP_SAT>, 1>, ap_ufixed<18,14>>(pairResult);
  
  std::cout << "pairResult before read = [";
  // for (int i = 0; i < 13; i++){ //v1
  // for (int i = 0; i < 8; i++){ //v3
  for (int i = 0; i < 1; i++){ //v5
    std::cout << pairResult.first[i] << ", ";
  }
  std::cout << "], " << pairResult.second << std::endl;

  std::cout << "Type of pairResult.first: " << typeid(pairResult.first).name() << std::endl;
  std::cout << "Type of pairResult.second: " << typeid(pairResult.second).name() << std::endl;
  std::cout << "reading result..." << std::endl;
  model->read_result(&pairResult);
  std::cout << "DONE..." << std::endl;

  
  std::cout << "pairResult after read = [";
  // for (int i = 0; i < 13; i++){
  // for (int i = 0; i < 8; i++){
  // for (int i = 0; i < 1; i++){
  //   std::cout << pairResult.first[i] << ", ";
  // }
  // std::cout << "], " << pairResult.second << std::endl;

  std::cout << pairResult.first << " " << pairResult.second << std::endl;
  

  return 0;
}
 
