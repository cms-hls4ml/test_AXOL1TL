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
  ap_fixed<18,13> modelInput[57] = {2047, 0, 0, 255, 20, 140, 10, 8, 77, 7, -4, 62, 5, 34, 73, 4, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1023, 19, 139, 234, 5, 71, 49, 63, 55, 44, 41, 7, 25, -86, 107, 13, 37, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  for (int i = 0; i < 57; i++) {
    // modelInput[i] = 10.25;
    std::cout << modelInput[i] << ", ";
  }
  std::cout << "]" << std::endl;

  ap_fixed<14,6,AP_RND_CONV,AP_SAT> scaled[57];
  ap_fixed<18,13> unscaled[57];
  typedef ap_fixed<5,5> ad_shift_t;
  typedef ap_fixed<10,10> ad_offset_t;

  const ad_shift_t ad_shift[57] = {3, 0, 6, 2, 5, 6, 0, 5, 6, 0, 5, 6, -1, 5, 6, 2, 7, 8, 0, 7, 8, 0, 7, 8, 0, 7, 8, 4, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 2, 6, 6};
  const ad_offset_t ad_offsets[57] = {18, 0, 72, 7, 0, 73, 4, 0, 73, 4, 0, 72, 3, 0, 72, 6, -0, 286, 3, -2, 285, 3, -2, 282, 3, -2, 286, 29, 0, 72, 22, 0, 72, 18, 0, 72, 14, 0, 72, 11, 0, 72, 10, 0, 72, 10, 0, 73, 9, 0, 73, 9, 0, 72, 8, -2, 72};
 
  for (int i = 0; i < 57; i++)
    {
      ap_fixed<18,13> tmp0 = unscaled[i] - ad_offsets[i];
      ap_fixed<14,6,AP_RND_CONV,AP_SAT> tmp1 = tmp0 >> ad_shift[i];
      scaled[i] = tmp1;
    }

  std::cout << "scaledInput = [";
  for (int i = 0; i < 57; i++) {
    // modelInput[i] = 10.25;
    std::cout << scaled[i] << ", ";
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
 
