#include "mex.h"
#include <malloc.h>
#include <algorithm>
#include <ctime>
#include <vector>
using std::vector;

// usage:
// extract_features_mex(feats in cell, spm_divs, boxes_in_cnn_input_images, feats_idxs, [offset0, offset, min_times])
// all in matlab index
// feats in (dim, width, height), dim is fastest
// response in ([width, height, channel], num)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	///// get input arguments
	if (!mxIsCell(prhs[0]))
		mexErrMsgTxt("feats must be in cell");

	int feats_num = (int)mxGetNumberOfElements(prhs[0]);
	if (feats_num <= 0)
		mexErrMsgTxt("feats num must > 0");

	vector<int> rsp_dims(feats_num);
	vector<int> rsp_widths(feats_num), rsp_heights(feats_num);
	vector<const float *> feats(feats_num);

	for (int i = 0; i < feats_num; ++i)
	{
		mxArray * mxFeats = mxGetCell(prhs[0], i);
		if (mxGetClassID(mxFeats) != mxSINGLE_CLASS)
			mexErrMsgTxt("feats must be single");
		feats[i] = ((const float*)mxGetData(mxFeats));

		const mwSize *feats_dim = mxGetDimensions(mxFeats);
		rsp_dims[i] = feats_dim[0];
		rsp_widths[i] = feats_dim[1];
		rsp_heights[i] = feats_dim[2];
	}

	int dim = rsp_dims[0];
	for (int i = 1; i < feats_num; ++i)
		if (rsp_dims[i] != dim)
			mexErrMsgTxt("currently only support feats with the same dim.\n");

    /// parse spm_divs
    const mxArray * mxDivs = prhs[1];
    int nDivM = (int)mxGetM(mxDivs);
    int nDivN = (int)mxGetN(mxDivs);
    if (std::min(nDivM, nDivN) != 1)
        mexErrMsgTxt("spm_divs must be a vetctor.\n");
    
    vector<int> spm_bin_divs(nDivM * nDivN);
    const double * pDivs = (const double*)mxGetPr(mxDivs);
    int max_level = nDivM * nDivN;
    int spm_divs = 0;
    for (int i = 0; i < nDivM * nDivN; i++)
    {
        spm_bin_divs[i] = (int)pDivs[i];
        spm_divs += (int)pDivs[i] * (int)pDivs[i];
    }

    /// parse boxes_in_cnn_input_images
	if (mxGetClassID(prhs[2]) != mxDOUBLE_CLASS)
		mexErrMsgTxt("boxes must be double");
	double* boxes = mxGetPr(prhs[2]);
	int num_boxes = (int)mxGetN(prhs[2]);
	if (mxGetM(prhs[2]) != 4)
	{
		mexPrintf("boxes error.");
		return;
	}

	if (mxGetClassID(prhs[3]) != mxINT32_CLASS)
		mexErrMsgTxt("feats_idxs must be int32");
	if (num_boxes != mxGetNumberOfElements(prhs[3]))
		mexErrMsgTxt("feats_idxs num must be the same boxes num.\n");
	const int *feat_ids = (const int*)mxGetPr(prhs[3]);
	
	if (mxGetClassID(prhs[4]) != mxDOUBLE_CLASS)
		mexErrMsgTxt("[offset0 offset min_times]  must be double");
	double offset0 = ((double *)mxGetData(prhs[4]))[0];
	double offset = ((double *)mxGetData(prhs[4]))[1];
	double min_times = ((double *)mxGetData(prhs[4]))[2];

	///// normalize box
	int* boxes_norm = new int[num_boxes * 4];
	for (int i = 0; i < num_boxes; i ++)
	{
		int best_feat = feat_ids[i] - 1;

		double* box = boxes + i * 4;

		int* box_norm = boxes_norm + i * 4;

		double x0 = box[0];
		double y0 = box[1];
		double x1 = box[2];
		double y1 = box[3];

		int x0_norm = int(floor((x0 - offset0 + offset) / min_times + 0.5) + 1);
		int y0_norm = int(floor((y0 - offset0 + offset) / min_times + 0.5) + 1);

		int x1_norm = int(ceil((x1 - offset0 - offset) / min_times - 0.5) + 1);
		int y1_norm = int(ceil((y1 - offset0 - offset) / min_times - 0.5) + 1);

		if (x0_norm > x1_norm)
		{
			x0_norm = (x0_norm + x1_norm) / 2;
			x1_norm = x0_norm;
		}

		if (y0_norm > y1_norm)
		{
			y0_norm = (y0_norm + y1_norm) / 2;
			y1_norm = y0_norm;
		}

		box_norm[0] = std::min(rsp_heights[best_feat], std::max(1, y0_norm)); // top // must not change
		box_norm[2] = std::min(rsp_heights[best_feat], std::max(1, y1_norm)); // bottom // must not change

		box_norm[1] = std::min(rsp_widths[best_feat], std::max(1, x0_norm)); // left // must not change
		box_norm[3] = std::min(rsp_widths[best_feat], std::max(1, x1_norm)); //right // must not change
	}

	//////////////////////////////////////////////////////////////////////////

	///// normalize box
	const int dim_pooled = std::abs(spm_divs) * dim;
	plhs[0] = mxCreateNumericMatrix(dim_pooled, num_boxes, mxSINGLE_CLASS, mxREAL);
	float* pooled = (float*)mxGetData(plhs[0]);
	memset((void*)pooled, 0, sizeof(float) * num_boxes * dim_pooled);

	float* pooled_cache = (float*)_aligned_malloc(dim_pooled * sizeof(float), 16);
	if (pooled_cache == NULL)
		mexErrMsgTxt("malloc error.");

	const int dim_pack = dim / 4;
	if ( dim % 4 != 0 )
		mexErrMsgTxt("spm pool mex: only support channel % 4 == 0");

	for (int i = 0; i < num_boxes; i ++)
	{
		int best_feat = feat_ids[i] - 1;

		const int feats_stride = rsp_widths[best_feat] * dim;

		memset((void*)pooled_cache, 0, sizeof(float) * dim_pooled);

		const int* box_norm = boxes_norm + i * 4;

		const int boxwidth = box_norm[3] - box_norm[1] + 1;
		const int boxheight = box_norm[2] - box_norm[0] + 1;

		float* pooled_this_div_cache = pooled_cache;
		for (int lv = 0; lv < max_level; lv ++)
		{
			const float bin_divs = (float)spm_bin_divs[lv];

			//const float wunit = boxwidth / (float)bin_divs;
			//const float hunit = boxheight / (float)bin_divs;

			for (int yy = 0; yy < bin_divs; yy ++)
			{
				//int y_start = (int)floor((yy - 1) * hunit) + 1;
				//int y_end = (int)ceil(yy * hunit);

				int y_start = (int)floor(yy / bin_divs * boxheight) + box_norm[0] - 1;
				int y_end = (int)ceil((yy + 1) / bin_divs * boxheight) + box_norm[0] - 1;

				//assert((y_start >= 0) && (y_end <= height));

				for (int xx = 0; xx < bin_divs; xx ++)
				{
					int x_start = (int)floor(xx / bin_divs * boxwidth) + box_norm[1] - 1;
					int x_end = (int)ceil((xx + 1) / bin_divs * boxwidth) + box_norm[1] - 1;

					//assert((x_start >= 0) && (x_end <= width));

					const float* feats_ptr = feats[best_feat] + y_start * feats_stride;
					for (int y = y_start; y < y_end; y ++)
					{
						//const float* feats_this = feats_ptr + x_start * dim;
						const __m128* feats_this_sse = (__m128*)(feats_ptr + x_start * dim);
						__m128* pooled_this_div_sse = (__m128*)pooled_this_div_cache;

						for (int x = x_start; x < x_end; x ++)
							for (int d = 0; d < dim_pack; d ++)
								pooled_this_div_sse[d] = _mm_max_ps(pooled_this_div_sse[d], *feats_this_sse ++);

						feats_ptr += feats_stride;
					}//y
					pooled_this_div_cache += dim;
				}//xx
			}//yy
		}//lv

		{
			// trans from ([channel, width, height], num) to ([width, height, channel], num)
			float *pooled_this_box = pooled + i * dim_pooled;
			float *pooled_this_div = pooled_this_box, *pooled_this_div_cache = pooled_cache;
			for (int lv = 0; lv < max_level; lv ++)
			{
				const int bin_divs = spm_bin_divs[lv];
				for (int ii = 0; ii < bin_divs * bin_divs; ii ++)
				{
					for (int d = 0; d < dim; d ++)
					{
						pooled_this_div[(d * bin_divs * bin_divs + ii)] = pooled_this_div_cache[d];
					}
					pooled_this_div_cache += dim;
				}
				pooled_this_div += bin_divs * bin_divs * dim; 
			}//lv
		}
	}//i

	_aligned_free(pooled_cache);
	delete[] boxes_norm;

}