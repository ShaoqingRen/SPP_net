#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <vector>
#include <omp.h>
#include "../linear.h"

#include "mex.h"
#include "linear_model_matlab.h"

using std::vector;

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}
void print_string_matlab(const char *s) {mexPrintf(s);}

void exit_with_help()
{
	mexPrintf(
	"Usage: model = train(training_label_vector, training_instance_matrix, 'liblinear_options', 'col');\n"
	"liblinear_options:\n"
	"-s type : set type of solver (default 1)\n"
	"  for multi-class classification\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"  for regression\n"
	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n" 
	"	-s 1, 3, 4 and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	"col:\n"
	"	if 'col' is setted, training_instance_matrix is parsed in column format(each sample in a col), otherwise is in row format\n"
	);
}

// liblinear arguments
struct parameter param;		// set by parse_command_line
vector<struct problem> probs;		// set by read_problem
vector<struct model *> pmodels;
struct feature_node *x_space;
int cross_validation_flag;
int col_format_flag;
int nr_fold;
double bias;

double do_cross_validation()
{
	{
		mexPrintf("Undefined cross validation for this version\n");
		return -1;
	}

	return 0.0;
}

// nrhs should be 3
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
	int i, argc = 1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];
	void (*print_func)(const char *) = print_string_matlab;	// default printing to matlab display

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation_flag = 0;
	col_format_flag = 0;
	bias = -1;


	if(nrhs <= 1)
		return 1;

	if(nrhs == 4)
	{
		mxGetString(prhs[3], cmd, mxGetN(prhs[3])+1);
		if(strcmp(cmd, "col") == 0)
			col_format_flag = 1;
	}

	// put options in argv[]
	if(nrhs > 2)
	{
		mxGetString(prhs[2], cmd,  mxGetN(prhs[2]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		if(i>=argc && argv[i-1][1] != 'q') // since option -q has no parameter
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'B':
				bias = atof(argv[i]);
				break;
			case 'v':
				cross_validation_flag = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					mexPrintf("n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			default:
				mexPrintf("unknown option\n");
				return 1;
		}
	}

	set_print_string_function(print_func);

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR: 
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL: 
			case L2R_L1LOSS_SVC_DUAL: 
			case MCSVM_CS: 
			case L2R_LR_DUAL: 
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC: 
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
	return 0;
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

template <typename DtypeLabel, typename DtypeData>
int read_problem_sparse(const mxArray *label_vec, const mxArray *instance_mat)
{
	int i, j, k, low, high;
	mwIndex *ir, *jc;
	int elements, max_index, num_samples, label_vector_row_num;
	DtypeData *samples;
	DtypeLabel *labels;
	mxArray *instance_mat_col; // instance sparse matrix in column format

	if(col_format_flag)
		instance_mat_col = (mxArray *)instance_mat;
	else
	{
		// transpose instance matrix
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instance_mat);
		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose training instance matrix\n");
			return -1;
		}
		instance_mat_col = plhs[0];
		mxDestroyArray(prhs[0]);
	}

	// parse sample in cols

	labels = (DtypeLabel *)mxGetData(label_vec);
	samples = (DtypeData *)mxGetPr(instance_mat_col);
	label_vector_row_num = (int)mxGetM(label_vec);
	int nSample = (int) mxGetN(instance_mat_col);

	int label_cols = (int)mxGetN(label_vec);
	if(label_cols <= 0)
	{
		mexPrintf("Cols of label must >= 1.\n");
		return -1;
	}

	if(label_vector_row_num!=nSample)
	{
		mexPrintf("Length of label vector does not match # of instances.\n");
		return -1;
	}

	probs.resize(label_cols);
	for (int pn = 0; pn < probs.size(); ++pn)
	{
		probs[pn].l = nSample;   //sample number 
		probs[pn].bias=bias;
	}

	// each column is one instance
	ir = mxGetIr(instance_mat_col);
	jc = mxGetJc(instance_mat_col);

	num_samples = (int) mxGetNzmax(instance_mat_col);

	elements = num_samples + nSample*2;
	max_index = (int) mxGetM(instance_mat_col);

	struct feature_node **pX = Malloc(struct feature_node *,nSample);
	for (int pn = 0; pn < probs.size(); ++pn)
	{
		probs[pn].y = Malloc(double,nSample);
		probs[pn].x = pX;
	}
	x_space = Malloc(struct feature_node, elements);

	j = 0;
	for(i=0;i<nSample;i++)
	{
		probs[0].x[i] = &x_space[j];
		low = (int) jc[i], high = (int) jc[i+1];
		for(k=low;k<high;k++)
		{
			x_space[j].index = (int) ir[k]+1;
			x_space[j].value = samples[k];
			j++;
	 	}
		if(bias>=0)
		{
			x_space[j].index = max_index+1;
			x_space[j].value = probs[0].bias;
			j++;
		}
		x_space[j++].index = -1;
	}

	j = 0;
	for(int pn = 0; pn < probs.size(); ++pn)
	{
		for(i = 0; i < nSample; i++)
		{
			probs[pn].y[i] = labels[j];
			j++;
		}
	}

	for (int pn = 0; pn < probs.size(); ++pn)
	{
		if(bias>=0)
			probs[pn].n = max_index+1;
		else
			probs[pn].n = max_index;
	}

	return 0;
}

template <typename DtypeLabel, typename DtypeData>
int read_problem_dense(const mxArray *label_vec, const mxArray *instance_mat)
{
	int i, k;
	long long elements, j;
	int max_index, sc, label_vector_row_num;
	DtypeData *samples;
	DtypeLabel *labels;
	DtypeData *thissample;
	mxArray *instance_mat_col; // instance sparse matrix in column format
	int pn;

	if(col_format_flag)
		instance_mat_col = (mxArray *)instance_mat;
	else
	{
		// transpose instance matrix
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instance_mat);
		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose training instance matrix\n");
			return -1;
		}
		instance_mat_col = plhs[0];
		mxDestroyArray(prhs[0]);
	}

	labels = (DtypeLabel *)mxGetData(label_vec);
	samples = (DtypeData *)mxGetData(instance_mat_col);
	sc = (int)mxGetM(instance_mat_col);


	label_vector_row_num = (int)mxGetM(label_vec);

	int label_cols = (int)mxGetN(label_vec);
	if(label_cols <= 0)
	{
		mexPrintf("Cols of label must >= 1.\n");
		return -1;
	}

	int nSample = (int)mxGetN(instance_mat_col);
	if(label_vector_row_num!=nSample)
	{
		mexPrintf("Length of label vector does not match # of instances.\n");
		return -1;
	}

	probs.resize(label_cols);
	for (pn = 0; pn < probs.size(); ++pn)
	{
		probs[pn].l = nSample;   //sample number 
		probs[pn].bias=bias;
	}

	elements = 0;
	for(i = 0; i < nSample; i++)
	{
		thissample = samples + i * sc;
		for(k = 0; k < sc; k++)
			if( thissample[k] != 0)
				elements++;
		if(bias>=0)
		{
			elements++;
		}
		elements++;
	}

	struct feature_node **pX = Malloc(struct feature_node *,nSample);
	for (pn = 0; pn < probs.size(); ++pn)
	{
		probs[pn].y = Malloc(double,nSample);
		probs[pn].x = pX;
	}
	x_space = Malloc(struct feature_node, elements);

	max_index = sc;
	j = 0;
	for(i = 0; i < nSample; i++)
	{
		thissample = samples + i * sc;

		probs[0].x[i] = &x_space[j];

		for(k = 0; k < sc; k++)
		{
			if( thissample[k] != 0)
			{
				x_space[j].index = k + 1;
				x_space[j].value = thissample[k];
				j++;
			}
		}
		if(bias>=0)
		{
			x_space[j].index = max_index+1;
			x_space[j].value = bias;
			j++;
		}
		x_space[j++].index = -1;
	}
	j = 0;
	for(pn = 0; pn < probs.size(); ++pn)
	{
		for(i = 0; i < nSample; i++)
		{
			probs[pn].y[i] = labels[j];
			j++;
		}
	}

	for (pn = 0; pn < probs.size(); ++pn)
	{
		if(bias>=0)
			probs[pn].n = max_index+1;
		else
			probs[pn].n = max_index;
	}

	return 0;
}

void freeProbs()
{
	int pn;
	destroy_param(&param);
	free(probs[0].x);
	free(x_space);
	for (pn = 0; pn < probs.size(); ++pn)
	{
		free(probs[pn].y);
		probs[pn].x = NULL;
	}
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;
	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);

	// Transform the input Matrix to libsvm format
	if(nrhs > 1 && nrhs < 5)
	{
		int err=0;

		if((!mxIsDouble(prhs[0]) && !mxIsSingle(prhs[0])) || (!mxIsDouble(prhs[1]) && !mxIsSingle(prhs[1]))) {
			mexPrintf("Error: label vector and instance matrix must be double or single\n");
			fake_answer(plhs);
			return;
		}

		if(parse_command_line(nrhs, prhs, NULL))
		{
			exit_with_help();
			destroy_param(&param);
			fake_answer(plhs);
			return;
		}

		if(!mxIsSparse(prhs[1]))
		{
			if ( mxIsDouble(prhs[0]) )
			{
				if (mxIsDouble(prhs[1]))
					err = read_problem_dense<double, double>(prhs[0], prhs[1]);
				else
					err = read_problem_dense<double, float>(prhs[0], prhs[1]);
			}
			else
			{
				if (mxIsDouble(prhs[1]))
					err = read_problem_dense<float, double>(prhs[0], prhs[1]);
				else
					err = read_problem_dense<float, float>(prhs[0], prhs[1]);
			}
		}
		else
		{
			if ( mxIsDouble(prhs[0]) )
			{
				if (mxIsDouble(prhs[1]))
					err = read_problem_sparse<double, double>(prhs[0], prhs[1]);
				else
					err = read_problem_sparse<double, float>(prhs[0], prhs[1]);
			}
			else
			{
				if (mxIsDouble(prhs[1]))
					err = read_problem_sparse<float, double>(prhs[0], prhs[1]);
				else
					err = read_problem_sparse<float, float>(prhs[0], prhs[1]);
			}
		}

		// train's original code
		error_msg = check_parameter(&probs[0], &param);

		if(err || error_msg)
		{
			if (error_msg != NULL)
				mexPrintf("Error: %s\n", error_msg);
			freeProbs();
			fake_answer(plhs);
			return;
		}

		if(cross_validation_flag)
		{
			double *ptr;
			plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
			ptr = mxGetPr(plhs[0]);
			ptr[0] = do_cross_validation();
		}
		else
		{
			const char *error_msg;

			pmodels.resize(probs.size());
            
            if (probs.size() <= 1)
                pmodels[0] = train(&probs[0], &param);
            else
            {
#pragma omp parallel for
                for (int pn = 0; pn < probs.size(); ++pn)
                {
                    pmodels[pn] = train(&probs[pn], &param);
                }	
            }

			error_msg = models_to_matlab_structure(plhs, pmodels);
			if(error_msg)
				mexPrintf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
			
			for (int pn = 0; pn < pmodels.size(); ++pn)
				free_and_destroy_model(&pmodels[pn]);
		}
		freeProbs();
	}
	else
	{
		exit_with_help();
		fake_answer(plhs);
		return;
	}
}
