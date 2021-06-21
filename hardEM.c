
#include "_struct.h"
#include "HMM.Read.Matrix.h"
#include "nrutil.h"
#include "hmm.h"
#include "mvnorm.h"
#include "models.h"
#include <time.h>
#include <math.h>


void hardEM_multiMVN_regression(gsl_matrix *design_X, gsl_matrix *Y, gsl_matrix *weight_m, int *q, HMM *phmm, PARM_HMM *parm_hmm1, int *ERROR_IND)
{
  gsl_matrix *X_temp, *Y_temp, *w_temp, *resid_temp_matrix;
  gsl_vector *X_row = gsl_vector_alloc(design_X ->size2);
  gsl_vector *Y_row = gsl_vector_alloc(Y -> size2);
  gsl_vector *w_row = gsl_vector_alloc(weight_m -> size2);
  gsl_vector *count_state_number = gsl_vector_alloc(phmm->N);
  gsl_vector *resid = gsl_vector_alloc(design_X->size1);
  gsl_vector *resid_temp;
  gsl_matrix *resid_matrix_temp;
  gsl_matrix *cov = gsl_matrix_alloc( design_X ->size2, design_X->size2);
  gsl_matrix *resid_matrix = gsl_matrix_alloc(design_X->size1, Y->size2);
  gsl_vector *coef = gsl_vector_alloc(design_X ->size2);
  gsl_vector *Y_col = gsl_vector_alloc(Y->size1);
  gsl_vector *w_temp_col, *Y_temp_col;
 
  double chisq;
  int state;
  int row_temp, count, weight_col_num_gsl;
  gsl_multifit_linear_workspace *work;
  double *mean_0;
  int i, mix_comp;
  

  mean_0 = (double *) dvector(1, Y->size2);  
  for( i = 1; i<= Y->size2;i++){
    mean_0[i]=(double) 0;
  }
  
 
  CountStateNumber(q, Y->size1, count_state_number);
  

  for( state = 1; state <= phmm->N ; state++){

    row_temp = gsl_vector_get(count_state_number, state-1);

    X_temp = gsl_matrix_alloc(row_temp, design_X->size2);
    Y_temp = gsl_matrix_alloc(row_temp, Y->size2);
    w_temp = gsl_matrix_alloc(row_temp, weight_m->size2);
    resid_temp = gsl_vector_alloc(row_temp);
    resid_temp_matrix = gsl_matrix_alloc(row_temp, Y->size2);
    count = 0;
    for(i = 0; i < phmm->M ; i++){
      if(q[i+1] == state){
	gsl_matrix_get_row(X_row, design_X, i);
	gsl_matrix_set_row(X_temp, count, X_row);
	gsl_matrix_get_row(Y_row, Y, i);
	gsl_matrix_set_row(Y_temp, count, Y_row);
	gsl_matrix_get_row(w_row, weight_m, i);
	gsl_matrix_set_row(w_temp, count, w_row);
	count +=1;
      }
    }

    work = gsl_multifit_linear_alloc(X_temp ->size1, X_temp ->size2 );
    w_temp_col = gsl_vector_alloc(row_temp);
    Y_temp_col = gsl_vector_alloc(row_temp);

    for(mix_comp = 1 ; mix_comp <= phmm->mix_comp; mix_comp++){ 
      weight_col_num_gsl = (state-1)* phmm->mix_comp + mix_comp-1;

      
      gsl_matrix_get_col(w_temp_col, w_temp, weight_col_num_gsl);
      
      for(i = 0; i < Y->size2; i++){
	gsl_matrix_get_col(Y_temp_col, Y_temp, i);

	gsl_matrix_get_col(Y_col, Y, i);

	gsl_multifit_wlinear(X_temp, w_temp_col, Y_temp_col, coef, cov, 
			     &chisq, work);  
	gsl_vector_to_dvector(coef, 
			      parm_hmm1->reg_coef[(weight_col_num_gsl)*(Y->size2) +i+1]);
      
      
      gsl_multifit_linear_residuals(X_temp, Y_temp_col, coef, resid_temp); 
      gsl_matrix_set_col(resid_temp_matrix, i , resid_temp);

     
      gsl_multifit_linear_residuals(design_X, Y_col, coef, resid); 
    
      gsl_matrix_set_col(resid_matrix, i, resid);
    

      }
    
    
      WeightedVar_gsl(resid_temp_matrix, mean_0, w_temp_col, 
		      resid_temp_matrix->size1, 
		      resid_temp_matrix->size2, 
		      parm_hmm1->reg_cov[weight_col_num_gsl+1],
		      ERROR_IND);
      
      mvdnorm_matrix_regression(resid_matrix, weight_col_num_gsl+1 , parm_hmm1,  phmm, ERROR_IND);
    }
    gsl_matrix_free(X_temp);
    gsl_matrix_free(Y_temp);
    gsl_matrix_free(w_temp);
    gsl_vector_free(resid_temp);
    gsl_matrix_free(resid_temp_matrix);
  }

  gsl_vector_free(X_row);
  gsl_vector_free(Y_row);
  gsl_vector_free(w_row);
  gsl_vector_free(count_state_number);
  gsl_matrix_free(resid_matrix);
  gsl_vector_free(coef);
  gsl_matrix_free(cov);
  gsl_multifit_linear_free(work);
  free_dvector(mean_0, 1, Y->size2);
    
}


void MakeDesignMatrix(gsl_matrix *X, gsl_matrix *design_X){

  gsl_vector *X_col = gsl_vector_alloc(X -> size1);
  int i;
  
  gsl_vector_set_all(X_col, 1);
  gsl_matrix_set_col(design_X, 0, X_col);
  
  for( i = 0; i < X-> size2; i++)
    {
      gsl_matrix_get_col(X_col,  X, i);
      gsl_matrix_set_col(design_X, i+1, X_col );
    }

  gsl_vector_free(X_col);
}
