#include <vector>


const char *model_to_matlab_structure(mxArray *plhs[], struct model *model_);
const char *models_to_matlab_structure(mxArray *plhs[], std::vector<struct model *> &pmodels);
const char *matlab_matrix_to_model(struct model *model_, const mxArray *matlab_struct);
const char *matlab_matrix_to_models(std::vector<struct model *> &pmodels, const mxArray *model_);
