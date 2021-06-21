/* the functions are explained in models.h file  */

#include "_struct.h"
#include "HMM.Read.Matrix.h"
#include "nrutil.h"
#include "hmm.h"
#include "mvnorm.h"
#include "models.h"
#include <time.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <gsl/gsl_randist.h>

#define MAX(x,y)        ((x) > (y) ? (x) : (y))
#define MIN(x,y)        ((x) < (y) ? (x) : (y))

void multiMVN_weight_gamma_ij(gsl_matrix *X, int i, gsl_matrix *weight_m, gsl_matrix *gamma_ij, HMM *phmm, PARM_HMM *parm_hmm1 , INI *ini, int *ERROR_IND)
{
  int T, j, k, p;
   gsl_vector *weight_m_col = gsl_vector_alloc(X->size1);
   gsl_vector *weight_sub_col; 
   gsl_vector *X_col;
   double gamma_i, c, b_i, b, sum;
   double gamma_sum_over_t;
   

   gsl_matrix_get_col(weight_m_col, weight_m, i-1);
   weight_sub_col = gsl_vector_alloc(X->size1);
   X_col = gsl_vector_alloc(X->size1);
   
   T = phmm-> M;
 


      gamma_sum_over_t = gsl_vector_sum(weight_m_col);
      if(0) printf(" gamma_sum_over_t = %f\n", gamma_sum_over_t);
      for( j = 1 ; j <=  phmm->mix_comp ; j++){
	c = parm_hmm1 -> c[i][j];
	for( k = 1; k <= T ; k++){
	  gamma_i = gsl_matrix_get(weight_m, k-1, i-1);
	  b_i = phmm ->B1[i][k][j];
	  b = phmm-> B[k][i];
	  
	  if (b ==0){
	    sum = gamma_i * c;} 
	  else {
	    sum = gamma_i * c * b_i / b;}
	  
	  gsl_vector_set(weight_sub_col, k-1,  sum );
	  gsl_matrix_set(gamma_ij, k-1, (i-1) * phmm->mix_comp + (j-1), sum);
	  
	  if(0) printf("k= %d sums for weight_sub_col = %f\n", k, sum);
	  if(0) printf("gamma_i = %f \t b_i = %f \t b = %f, c = %f \n", gamma_i, b_i, b , c);
	}
    
	for( p = 1; p <= X->size2; p++){
	  gsl_matrix_get_col(X_col, X, p-1);
	  if(parm_hmm1->mu_user_para){
	    parm_hmm1->mu[(i-1) * (phmm->mix_comp) + j][p] = 
	      ini->mu[(i-1) * (phmm->mix_comp) + j][p];
	  } else {
	    parm_hmm1 -> mu[(i-1) * (phmm->mix_comp) + j ][p] = 
	    WeightedMean_gsl(X_col, weight_sub_col, T, ERROR_IND);
	  }
	  if(ERROR_IND[0]) {
	    gsl_vector_free(weight_sub_col);
	    gsl_vector_free(weight_m_col);
	    gsl_vector_free(X_col);
	    return;
	  }
	  
	}
	
	if(ERROR_IND[0]!=1){
	  WeightedVar_gsl(X, parm_hmm1->mu[(i-1)*phmm->mix_comp + j], 
			  weight_sub_col, T , X->size2, 
			  parm_hmm1->Sigma[(i-1)*phmm->mix_comp + j ], ERROR_IND);
	
 
	  parm_hmm1->c[i][j] = gsl_vector_sum(weight_sub_col)/gamma_sum_over_t;
	} else return; /* */
      }
      
}

void multiMVN_weight(gsl_matrix *X, gsl_matrix *weight_m, HMM *phmm, PARM_HMM *parm_hmm1, gsl_matrix *gamma_ij, double *time, INI *ini,  int *ERROR_IND){

  int i, T;
  clock_t gamma_ij_start, gamma_ij_end, mvnorm_matrix_start, mvnorm_matrix_end;
  /* mu, Sigma, c in parm_hmm1 */
  int tid, chunk=1, nthreads;
  
  

  gamma_ij_start = omp_get_wtime();
  T = phmm-> M;


#pragma omp parallel shared(X, weight_m,  phmm, parm_hmm1, gamma_ij, nthreads, chunk) private(i, tid)
  {
#pragma omp  for schedule (static, chunk)
    for( i = 1 ; i <= phmm-> N ; i++)
      multiMVN_weight_gamma_ij(X, i, weight_m, gamma_ij, phmm, parm_hmm1, ini,  ERROR_IND);
      
  }

  gamma_ij_end = omp_get_wtime();
  time[0] += (double)(gamma_ij_end - gamma_ij_start);
  
  	
  /* Calculate B1 */

      mvnorm_matrix_start = omp_get_wtime();
  
#pragma omp parallel shared(X, parm_hmm1, nthreads, chunk) private(i, tid)
      {
	nthreads = omp_get_num_threads();
	tid = omp_get_thread_num();
	
#pragma omp for schedule (static, chunk)
	for( i  = 1 ; i<= phmm->mix_comp * phmm->N ; i++){
	  mvdnorm_matrix_multiMVN(X, parm_hmm1->mu[i], parm_hmm1->Sigma[i], i, phmm , ERROR_IND);
	  /* printf("Thread %d, i = %d\n", tid, i); */
	  if(ERROR_IND[0]){
	    printf("Error in multiMVN_weight\n");
	  }
	}
      }
      mvnorm_matrix_end=omp_get_wtime();
      
      time[1] += mvnorm_matrix_end - mvnorm_matrix_start;
  
}
 


void multiMVN(gsl_matrix *X, gsl_matrix *weight_m, HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, int *ERROR_IND){

  int i, j, k, p, T;
  gsl_vector *weight_sub_col, *weight_m_col;
  gsl_vector *X_col;
  double gamma_i, c, b_i, b, sum;
  double  gamma_sum_over_t, temp;
  FILE *file_B  ;
  /* mu, Sigma, c in parm_hmm1 */
  
  weight_sub_col = gsl_vector_alloc(X->size1);
  weight_m_col = gsl_vector_alloc(X->size1);
  X_col = gsl_vector_alloc(X->size1);


  T = phmm-> M;
  for( i = 1 ; i <= phmm-> N ; i++){
    gsl_matrix_get_col(weight_m_col, weight_m, i-1);
    gamma_sum_over_t = gsl_vector_sum(weight_m_col);
    if(0) printf(" gamma_sum_over_t = %f\n",gamma_sum_over_t);
    for( j = 1 ; j <=  phmm->mix_comp ; j++){
      c = parm_hmm1 -> c[i][j];
      for( k = 1; k <= T ; k++){
	gamma_i = gsl_matrix_get(weight_m, k-1, i-1);
	b_i = phmm ->B1[i][k][j];
	b = phmm-> B[k][i];
	if (b == 0){ 
	  if(0){
	    printf("Warning: b = 0 and this nan: %d  \n", k);
	    file_B = fopen("B.tex", "w");
	    PrintHMM(file_B, phmm);
	    fclose(file_B);
	   
	  }
	  sum = gamma_i * c;} 
	else {
	  sum = gamma_i * c * b_i / b;}
	gsl_vector_set(weight_sub_col, k-1,  sum );
	
	if(0) printf("k= %d sum for weight_sub_col = %f\n", k, sum);
	if(0) printf("gamma_i = %f \t b_i = %f \t b = %f, c = %f \n", gamma_i, b_i, b , c);
      }
 
      for( p = 1; p <= X->size2; p++){
	gsl_matrix_get_col(X_col, X, p-1);
	if(parm_hmm1->mu_user_para){
	    parm_hmm1->mu[(i-1) * (phmm->mix_comp) + j][p] = 
	      ini->mu[(i-1) * (phmm->mix_comp) + j][p];
	} else {
	  temp=WeightedMean_gsl(X_col, weight_sub_col, T, ERROR_IND);
	  parm_hmm1->mu[(i-1) * (phmm->mix_comp) + j ][p] = temp;
	}
	if(ERROR_IND[0]){
	  gsl_vector_free(weight_sub_col);
	  gsl_vector_free(weight_m_col);
	  gsl_vector_free(X_col);
	  return;
	}
      }
            
      WeightedVar_gsl(X, parm_hmm1->mu[(i-1)*phmm->mix_comp + j], 
			weight_sub_col, T , X->size2, 
		      parm_hmm1->Sigma[(i-1)*phmm->mix_comp + j ], ERROR_IND);
      

      parm_hmm1->c[i][j] = gsl_vector_sum(weight_sub_col)/gamma_sum_over_t;
      
    }
  }

  /* Calculate observation probability */
  
  if(parm_hmm1->mu_user_para){
    for(i=1; i<= weight_m -> size2;i++){
      for(j=1; j<=ini->colX;j++){
	parm_hmm1->mu[i][j] = ini->mu[i][j];
      }
    }
  } 
  for( i  = 1 ; i<= phmm->mix_comp * phmm->N ; i++){
    mvdnorm_matrix_multiMVN(X, parm_hmm1->mu[i], parm_hmm1->Sigma[i], i, phmm, ERROR_IND);
  }
  
  
    gsl_vector_free(weight_sub_col);
    gsl_vector_free(weight_m_col);
    gsl_vector_free(X_col);
}



 


void singleMVN(gsl_matrix  *X, gsl_matrix *weight_m, HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, int *ERROR_IND) {
 
  int i,j;

  if(parm_hmm1->mu_user_para){
    for(i=1; i<= weight_m->size2;i++){
      for(j=1; j<=ini->colX;j++){
	parm_hmm1->mu[i][j] = ini->mu[i][j];
      }
    }
  } else {
  
    WeightedMean_gsl_matrix(X, weight_m, parm_hmm1, ERROR_IND) ;
    WeightedVar_gsl_matrix(X, weight_m, parm_hmm1, ERROR_IND);
    
    for( i = 1; i <= phmm->N; i++)
      {
	if(ERROR_IND[0]) break;
	mvdnorm_matrix(X, parm_hmm1-> mu[i], parm_hmm1->Sigma[i], i, phmm, ERROR_IND);
	
      }
  }
 
}

void singleMVN_regression_no_intercept( gsl_matrix *X, gsl_matrix *Y, gsl_matrix *weight_m,  HMM *phmm, PARM_HMM *parm_hmm1,double *time, int *ERROR_IND) 
{
  int i, j;
  
  gsl_vector *y = gsl_vector_alloc(Y-> size1);
  gsl_vector *w = gsl_vector_alloc(X-> size1);
  gsl_vector *X_col =gsl_vector_alloc(X -> size1);
  gsl_vector *coef = gsl_vector_alloc((X->size2));
  gsl_vector *resid = gsl_vector_alloc(X->size1);
  gsl_matrix *resid_matrix = gsl_matrix_alloc(X->size1, Y->size2);
  gsl_matrix *cov = gsl_matrix_alloc( (X->size2) , (X -> size2) );
  double *resid_mean;
  double chisq;
  double *mean_temp;
  FILE *fp;
  gsl_multifit_linear_workspace *work =  gsl_multifit_linear_alloc(X->size1, (X->size2) );
  clock_t multifit_wlinear_start, multifit_wlinear_end, residuals_start, residuals_end, weight_var_start, weight_var_end, mvdnorm_matrix_regression_start, mvdnorm_matrix_regression_end;

   
  resid_mean = (double *) dvector(1, Y->size2);
  mean_temp = (double *) dvector(1, Y->size2);
  for( i = 1; i<= Y->size2;i++){
    mean_temp[i]=(double) 0;
  }
  
  for( j = 0; j < weight_m->size2 ; j++){
    gsl_matrix_get_col(w, weight_m, j);
      for( i = 0; i < Y->size2; i++){
	gsl_matrix_get_col(y, Y, i);
	multifit_wlinear_start=clock();
	gsl_multifit_wlinear(X, w, y, coef, cov,  &chisq, work);  
	
	gsl_vector_to_dvector(coef, parm_hmm1->reg_coef[j*(Y->size2) +i+1]);
	multifit_wlinear_end=clock();
	time[0] += (double)((multifit_wlinear_end - multifit_wlinear_start)); 

	residuals_start =clock();
	gsl_multifit_linear_residuals(X, y, coef, resid);
	gsl_matrix_set_col(resid_matrix, i, resid);
	/* resid_mean[i+1]= 0; */
	resid_mean[i+1]= WeightedMean_gsl(resid, w, resid->size, ERROR_IND);
	residuals_end = clock();
	time[1] += (double)((residuals_end - residuals_start)); 
	
      }
      if(0)dvector_print(resid_mean, Y->size2);
      weight_var_start = clock();

      weight_var_end = clock();
      time[2] += (double)((weight_var_end - weight_var_start));
      WeightedVar_gsl(resid_matrix, resid_mean, w, resid_matrix->size1, 
		      resid_matrix->size2, parm_hmm1->reg_cov[j+1], ERROR_IND);
      
      mvdnorm_matrix_regression_start = clock();
      mvdnorm_matrix_regression(resid_matrix, j+1, parm_hmm1,  phmm, ERROR_IND);


     
      if(ERROR_IND[0]){
	fp = fopen("resid_matrix", "w");
	gsl_matrix_fprintf(fp, resid_matrix, "%e");
	fclose(fp);
	fp = fopen("weight_vector", "w");
	gsl_vector_fprintf(fp, w, "%e");
	fclose(fp);
	/* july 1 exit(1); */
      }
      mvdnorm_matrix_regression_end = clock();
      
      time[3] += (double)((mvdnorm_matrix_regression_end - mvdnorm_matrix_regression_start));
    
  }


  
      gsl_vector_free(y);
      gsl_vector_free(w); 
      gsl_vector_free(coef);
      gsl_vector_free(X_col);
      gsl_vector_free(resid);
      gsl_matrix_free(resid_matrix);
      gsl_matrix_free(cov); 
      gsl_multifit_linear_free(work);
      free_dvector(resid_mean, 1, Y->size2);
      free_dvector(mean_temp, 1, Y->size2);
    
}
void singleMVN_regression(gsl_matrix *X, gsl_matrix *Y, gsl_matrix *design_X,
			  gsl_matrix *weight_m,  HMM *phmm, PARM_HMM *parm_hmm1,			  double *time, int *ERROR_IND)
{
  int i, j, k;
  
  gsl_vector *y = gsl_vector_alloc(Y-> size1);
  gsl_vector *w = gsl_vector_alloc(X-> size1);
  gsl_vector *X_col =gsl_vector_alloc(X -> size1);
  gsl_vector *coef = gsl_vector_alloc((X->size2) + 1);
  gsl_vector *resid = gsl_vector_alloc(X->size1);
  gsl_matrix *resid_matrix = gsl_matrix_alloc(X->size1, Y->size2);
  gsl_matrix *cov = gsl_matrix_alloc( (X->size2) + 1, (X -> size2) + 1);
  double *resid_mean;
  double chisq;
  double *mean_temp;
  FILE *fp;
  gsl_multifit_linear_workspace *work =  gsl_multifit_linear_alloc(X->size1, (X->size2) +1 );
  clock_t multifit_wlinear_start, multifit_wlinear_end, residuals_start, residuals_end, weight_var_start, weight_var_end, mvdnorm_matrix_regression_start, mvdnorm_matrix_regression_end;

   
  resid_mean = (double *) dvector(1, Y->size2);
  mean_temp = (double *) dvector(1, Y->size2);
  for( i = 1; i<= Y->size2;i++){
    mean_temp[i]=(double) 0;
  }

 
  for( j = 0; j < weight_m->size2 ; j++){

    /* Update Regression Coefficients (reg.coef) */

    gsl_matrix_get_col(w, weight_m, j);
      for( i = 0; i < Y->size2; i++){
	gsl_matrix_get_col(y, Y, i);
	multifit_wlinear_start=clock();
	gsl_multifit_wlinear(design_X, w, y, coef, cov,  &chisq, work);  
	
	gsl_vector_to_dvector(coef, parm_hmm1->reg_coef[j*(Y->size2) +i+1]);
	multifit_wlinear_end=clock();

	time[0] += (double)((multifit_wlinear_end - multifit_wlinear_start)); 

	if(0) {
	  printf ("Regressioprn Coefficients for Y[%d] in state %d: \n", i, j);
	  for(k = 0; k < design_X->size2; k++){
	    printf("%f \t", gsl_vector_get(coef, (k)));
	  }
	  printf("\n");
	}

	if(0) {
	  printf ("Regressioprn Coefficients for Y[%d] in state %d: \n", i, j+1);
	  for(k = 0; k < design_X->size2; k++){
	    printf("%f \t", gsl_vector_get(coef, (k)));
	  }
	  printf("\n");
	}
	
	residuals_start =clock();
	gsl_multifit_linear_residuals(design_X, y, coef, resid);
	gsl_matrix_set_col(resid_matrix, i, resid);
	/* resid_mean[i+1]= 0; */
	resid_mean[i+1]= WeightedMean_gsl(resid, w, resid->size, ERROR_IND);
	residuals_end = clock();
	time[1] += (double)((residuals_end - residuals_start)); 
	
      }
      if(0)dvector_print(resid_mean, Y->size2);
      weight_var_start = clock();

      weight_var_end = clock();
      time[2] += (double)((weight_var_end - weight_var_start));

      /* Updata Regression Standard Error or variance (if d > 1) reg_cov */
      WeightedVar_gsl(resid_matrix, resid_mean, w, resid_matrix->size1, 
		      resid_matrix->size2, parm_hmm1->reg_cov[j+1], ERROR_IND);
      
      mvdnorm_matrix_regression_start = clock();

      /* Update Observation Probability */
      mvdnorm_matrix_regression(resid_matrix, j+1, parm_hmm1,  phmm, ERROR_IND);


     
      if(ERROR_IND[0]){
	fp = fopen("resid_matrix", "w");
	gsl_matrix_fprintf(fp, resid_matrix, "%e");
	fclose(fp);
	fp = fopen("weight_vector", "w");
	gsl_vector_fprintf(fp, w, "%e");
	fclose(fp);
	/* July 1st exit(1);*/
      }
      mvdnorm_matrix_regression_end = clock();
      
      time[3] += (double)((mvdnorm_matrix_regression_end - mvdnorm_matrix_regression_start));
    
  }


  
      gsl_vector_free(y);
      gsl_vector_free(w); 
      gsl_vector_free(coef);
      gsl_vector_free(X_col);
      gsl_vector_free(resid);
      gsl_matrix_free(resid_matrix);
      gsl_matrix_free(cov); 
      gsl_multifit_linear_free(work);
      free_dvector(resid_mean, 1, Y->size2);
      free_dvector(mean_temp, 1, Y->size2);
    
}
void user_defined_X_singleMVN_Regression(gsl_matrix *X, gsl_matrix *Y, gsl_matrix *design_X,  gsl_matrix *weight_m,  HMM *phmm, PARM_HMM *parm_hmm1, INI *ini,  double *time, int *ERROR_IND)
{
  int i, j, k, var_ind, var_count, count;
  
  gsl_vector *y = gsl_vector_alloc(Y-> size1);
  gsl_vector *w = gsl_vector_alloc(X-> size1);
  gsl_vector *X_col =gsl_vector_alloc(X -> size1);
  gsl_vector *coef = gsl_vector_alloc((X->size2) + 1);
  gsl_vector *resid = gsl_vector_alloc(X->size1);
  gsl_matrix *resid_matrix = gsl_matrix_alloc(X->size1, Y->size2);
  gsl_matrix *cov = gsl_matrix_alloc( (X->size2) + 1, (X -> size2) + 1);
  gsl_matrix *new_X; 
  gsl_vector *new_X_col = gsl_vector_alloc(X->size1);
  double *resid_mean;
  double chisq;
  double *mean_temp;
  FILE *fp;
  gsl_multifit_linear_workspace *work;

  clock_t multifit_wlinear_start, multifit_wlinear_end, residuals_start, residuals_end,  mvdnorm_matrix_regression_start, mvdnorm_matrix_regression_end;

   
  resid_mean = (double *) dvector(1, Y->size2);
  mean_temp = (double *) dvector(1, Y->size2);
  for( i = 1; i<= Y->size2;i++){
    mean_temp[i]=(double) 0;
  }
 
  for( j = 0; j < weight_m->size2 ; j++){
    var_ind=1;
    var_count = 0;
    
    if(j+1 <= parm_hmm1-> reg_coef_user_num){
      for(k=1; k <= (parm_hmm1-> p +1); k++){
	var_ind = var_ind* ini->reg_coef_fixed_parm[j+1][k];
	parm_hmm1->reg_coef_fixed_parm[j+1][k]=ini->reg_coef_fixed_parm[j+1][k];
	if(parm_hmm1->reg_coef_fixed_parm[j+1][k]==0) var_count += 1;
      }
     
      /* Update Regression Coefficients (reg.coef) */
      if(var_ind == 0){ 
	/* Only a part of covariates will be used */
	work =  gsl_multifit_linear_alloc(X->size1, var_count);
	coef = gsl_vector_alloc(var_count);
	cov = gsl_matrix_alloc(var_count, var_count);
	new_X = gsl_matrix_alloc((X->size1), var_count);
	new_X_col = gsl_vector_alloc(X->size1);
	count=0;
	for(k = 0; k< (parm_hmm1->p +1) ; k++){
	  if( ini->reg_coef_fixed_parm[j+1][k+1]==0){
	    gsl_matrix_get_col(new_X_col, design_X, k);
	    gsl_matrix_set_col(new_X, count, new_X_col);
	    count++;
	  }
	}

	gsl_matrix_get_col(w, weight_m, j);
	for( i = 0; i < Y->size2; i++){ 
	  gsl_matrix_get_col(y, Y, i);
	  gsl_multifit_wlinear(new_X, w, y, coef, cov,  &chisq, work);  
	  
	  count=0;
	  for(k = 1; k <= parm_hmm1->p + 1; k++){
	    if(parm_hmm1->reg_coef_fixed_parm[j + 1][k] == 0){
	      parm_hmm1->reg_coef[j*Y->size2 + i + 1][k]= gsl_vector_get(coef,count); 
	      count++;
	    } else { parm_hmm1->reg_coef[j* Y->size2 + i+1][k]=0;}
	  }
	  gsl_vector_print(coef);
	  /* caution The line above: look how reg_coef is defined*/
	  
	  residuals_start =clock();
	  gsl_multifit_linear_residuals(new_X, y, coef, resid);
	  gsl_matrix_set_col(resid_matrix, i, resid);
	  /* resid_mean[i+1]= 0; */
	  resid_mean[i+1]= WeightedMean_gsl(resid, w, resid->size, ERROR_IND);
	  residuals_end = clock();
	}

	gsl_multifit_linear_free(work);
	gsl_vector_free(coef);
	gsl_matrix_free(cov);
	gsl_matrix_free(new_X); 
        gsl_vector_free(new_X_col);
	
      } else { /* Fixed regression coefficients - find the residuals only */
	work =  gsl_multifit_linear_alloc(X->size1, (X->size2) +1 );
	coef = gsl_vector_alloc((X->size2) +1);
	cov = gsl_matrix_alloc((X->size2) +1, (X->size2) +1);
	
	gsl_matrix_get_col(w, weight_m, j);
	 for( i = 0; i < Y->size2; i++){

	  gsl_matrix_get_col(y, Y, i);
	  /* copy fixed_coef to coef */
	  dvector_to_gsl_vector(parm_hmm1->reg_coef[j * Y->size2 + i + 1], coef);
	  gsl_multifit_linear_residuals(design_X, y, coef, resid);
	  gsl_matrix_set_col(resid_matrix, i, resid);
	  resid_mean[i+1]= WeightedMean_gsl(resid, w, resid->size, ERROR_IND);
	   }
	gsl_multifit_linear_free(work);
	gsl_vector_free(coef);
	gsl_matrix_free(cov);
      }
      
    } else {/* No user-defined regression coefficents and using all covariates */
	work =  gsl_multifit_linear_alloc(X->size1, (X->size2) +1 );
	coef = gsl_vector_alloc((X->size2) +1);
	cov = gsl_matrix_alloc((X->size2) +1, (X->size2) +1);

	gsl_matrix_get_col(w, weight_m, j);
	for( i = 0; i < Y->size2; i++){
	  gsl_matrix_get_col(y, Y, i);
	  multifit_wlinear_start=clock();
	  gsl_multifit_wlinear(design_X, w, y, coef, cov,  &chisq, work);  
	  
	  gsl_vector_to_dvector(coef, parm_hmm1->reg_coef[j*(Y->size2) +i+1]);
	  multifit_wlinear_end=clock();
	  time[0] += (double)((multifit_wlinear_end - multifit_wlinear_start)); 
	  
	  
	  residuals_start =clock();
	  gsl_multifit_linear_residuals(design_X, y, coef, resid);
	  gsl_matrix_set_col(resid_matrix, i, resid);
	  /* resid_mean[i+1]= 0; */
	  resid_mean[i+1]= WeightedMean_gsl(resid, w, resid->size, ERROR_IND);
	  residuals_end = clock();
	  time[1] += (double)((residuals_end - residuals_start)); 
	  
	}
	
	gsl_multifit_linear_free(work);
	gsl_vector_free(coef);
	gsl_matrix_free(cov); 
  
    }

      /* Updata Regression Standard Error or variance (if d > 1) reg_cov */
      WeightedVar_gsl(resid_matrix, resid_mean, w, resid_matrix->size1, 
		      resid_matrix->size2, parm_hmm1->reg_cov[j+1], ERROR_IND);
      
      mvdnorm_matrix_regression_start = clock();

      /* Update Observation Probability */
      mvdnorm_matrix_regression(resid_matrix, j+1, parm_hmm1,  phmm, ERROR_IND);


     
      if(ERROR_IND[0]){
	fp = fopen("resid_matrix", "w");
	gsl_matrix_fprintf(fp, resid_matrix, "%e");
	fclose(fp);
	fp = fopen("weight_vector", "w");
	gsl_vector_fprintf(fp, w, "%e");
	fclose(fp);
      }
      mvdnorm_matrix_regression_end = clock();
      
      time[3] += (double)((mvdnorm_matrix_regression_end - mvdnorm_matrix_regression_start));
    
    
  }  
      gsl_vector_free(y);
      gsl_vector_free(w); 
      gsl_vector_free(X_col);
      gsl_vector_free(resid);
      gsl_matrix_free(resid_matrix);
      free_dvector(resid_mean, 1, Y->size2);
      free_dvector(mean_temp, 1, Y->size2);
    
}

void mvdnorm_matrix_regression(gsl_matrix *X, int state, PARM_HMM *parm_hmm1,  HMM *hmm1, int *ERROR_IND)
{
  

  int i, T, row_sq;
  int t;
  int B1_state, mix;
  gsl_vector *x_row;
  gsl_matrix *A_inv;
  double det, temp;
  int print_ind = 0;
  double *mu_vector;
  FILE *fp;

  mu_vector = (double *)dvector(1, X->size2);
  
  for( i = 1; i<= X->size2; i++){
    mu_vector[i] = 0;
  }
  
  T = (int) X->size1;
  row_sq = (int) X->size2;
  x_row = gsl_vector_alloc(row_sq);
  A_inv =  gsl_matrix_alloc(row_sq, row_sq);

  det = get_det(parm_hmm1->reg_cov[state], row_sq, A_inv, ERROR_IND);

  if(print_ind) printf("det = %2.52lf \n", fabs(det));
  if(0){
    fp = fopen("mvB.tex", "w");
    PrintHMM(fp, hmm1);
    fclose(fp);
  }
  if(!strcmp(hmm1->modelname, "singleMVN_regression") ){
    for (t = 1; t <= T; t++)
    {
      gsl_matrix_get_row(x_row, X, (size_t) t-1);
      /* hmm1-> B[t][state]  = .2*(hmm1-> B[t][state]) +  mvdnorm_B(x_row,  mu_vector, det, A_inv, 1);*/
      hmm1-> B[t][state] +=mvdnorm_B(x_row,  mu_vector, det, A_inv, 1);
    }
  }
  
  if(!strcmp(hmm1->modelname, "single_regression") ){
    for (t = 1; t <= T; t++)
      {
	gsl_matrix_get_row(x_row, X, (size_t) t-1);
	hmm1-> B[t][state] =mvdnorm_B(x_row,  mu_vector, det, A_inv, 1);
      }
  }

  
  if(!strcmp(hmm1->modelname, "multiMVN_regression")){
    mix = (state - 1) % hmm1->mix_comp +1;
    B1_state = (state - mix)/hmm1->mix_comp + 1;
    if(0)    printf(" B1_state  =  %d, mix_comp = %d \n", B1_state, mix);
     for (t = 1; t <= T; t++)
    {
      gsl_matrix_get_row(x_row, X, (size_t) t-1);
      
      temp = hmm1-> B1[B1_state][t][mix];
      hmm1-> B1[B1_state][t][mix] += mvdnorm_B(x_row,  mu_vector, det, A_inv, 1);
      if(det < 0){
	printf("B1[B1_state][t][mix] >  0. \n");
	printf("mvdnorm = %f \t regression = %f \t det = %e (state %d t %d mix %d)\n", temp, mvdnorm_B(x_row,  mu_vector, det, A_inv, 1), det, B1_state, t, mix);
	fp = fopen("A_inv", "w");
	gsl_matrix_fprintf(fp, A_inv, "%.15lf");
	fclose(fp);
		
	if(0) dmatrix_print(parm_hmm1->reg_cov[state], row_sq, row_sq);
	ERROR_IND[0]=1;
	if(1)  return;
	
      }
    }
  }
  if(!strcmp(hmm1->modelname, "multiMVN_regression_state")){
    B1_state = state;

    if(0)    printf(" B1_state  =  %d, mix_comp = %d \n", B1_state, mix);
    for (t = 1; t <= T; t++)
      {
	gsl_matrix_get_row(x_row, X, (size_t) t-1);
	
	temp = hmm1-> B1[B1_state][t][mix];
	hmm1-> B[t][state] += (double) mvdnorm_B(x_row,  mu_vector, det, A_inv, 1);
	if(det < 0){
	  printf("B1[B1_state][t][mix] >  0. \n");
	  printf("mvdnorm = %f \t regression = %f \t det = %e (state %d t %d mix %d)\n", temp, mvdnorm_B(x_row,  mu_vector, det, A_inv, 1), det, B1_state, t, mix);
	  fp = fopen("A_inv", "w");
	  gsl_matrix_fprintf(fp, A_inv, "%.15lf");
	  fclose(fp);
		
	  if(0) dmatrix_print(parm_hmm1->reg_cov[state], row_sq, row_sq);
	  ERROR_IND[0]=1;
	  if(1)  return;
	}
      }
  }
   
 
  gsl_matrix_free(A_inv);
  gsl_vector_free(x_row);
  free_dvector(mu_vector, 1, X->size2);

}


void mvdnorm_matrix(gsl_matrix *x, double *mu_vector, double **Sigma, int state, HMM *hmm1, int *ERROR_IND)
{
 
  int T, row_sq;
  int t;
  gsl_vector *x_row;
  gsl_matrix *A_inv;
  double det;
  int print_ind = 0;

  T = (int) x->size1;
  row_sq = (int) x->size2;
  x_row = gsl_vector_alloc(row_sq);
  A_inv =  gsl_matrix_alloc(row_sq, row_sq);
  
  det = get_det(Sigma, row_sq, A_inv, ERROR_IND);
  

  if(print_ind) printf("det = %2.52lf \n", fabs(det));
  for (t = 1; t <= T; t++)
    {
      gsl_matrix_get_row(x_row, x, (size_t) t-1);
      hmm1-> B[t][state] = mvdnorm_B(x_row,  mu_vector, det, A_inv, 1);
    }
 
  gsl_matrix_free(A_inv);
  gsl_vector_free(x_row);

}

void LogMaxScaleB(HMM *hmm1){
  double max_t = 0;
  double max_sum = 0;
  int i,j,t;

  for( t = 1; t<= hmm1-> M; t++){
    max_t = hmm1->B[t][1];
    for(i = 2; i <= hmm1-> N; i++){
      max_t = MAX(max_t, hmm1->B[t][i]);
    }

    for( j = 1; j<= hmm1 ->N; j++){
	 hmm1->B[t][j] = exp(hmm1->B[t][j] - max_t);
	 /* hmm1->B[t][j] = exp(hmm1->B[t][j]);*/
    }
    max_sum += max_t;
  }

 
  hmm1->B_scale = max_sum;
}

void Initial_state_singleMVN(HMM *hmm1, PARM_HMM *parm_hmm1, int *q, gsl_matrix *X)
{
  /* Calculate initial mean and variance based on initial states */
  
  gsl_matrix *weight_m = gsl_matrix_alloc(hmm1-> M, hmm1->N);
  int i, ERROR_IND[1]={0};
  
  
  InitialWeight_gsl_m(q, weight_m, 0);
  
  WeightedMean_gsl_matrix(X, weight_m, parm_hmm1, ERROR_IND); 
  WeightedVar_gsl_matrix(X, weight_m, parm_hmm1, ERROR_IND);

   for( i = 1; i <= hmm1->N; i++)
    {

      mvdnorm_matrix(X, parm_hmm1-> mu[i], parm_hmm1->Sigma[i], i, hmm1, ERROR_IND);
      if(ERROR_IND[0]) return;
      
    }

   LogMaxScaleB(hmm1);
   gsl_matrix_free(weight_m);

}
 
void Initial_state_multiMVN(HMM *hmm1, PARM_HMM *parm_hmm1, int *q, gsl_matrix *X)
{
  /* Calculate initial mean and variance based on initial states */
  
  gsl_matrix *weight_m_all = gsl_matrix_alloc(hmm1->M, 
					      hmm1->N* hmm1-> mix_comp); 
  gsl_matrix *weight_m = gsl_matrix_alloc(hmm1-> M, hmm1->N);
  gsl_matrix *weight_m_sub = gsl_matrix_alloc(hmm1->M, hmm1->mix_comp);
  gsl_vector *col = gsl_vector_alloc(hmm1->M);
  gsl_vector *col_sub = gsl_vector_alloc(hmm1->M);
  int *q_sub;
  int i, j, ERROR_IND[1]={0};
  
  q_sub = ivector(1, hmm1->M);

  
  
  InitialWeight_gsl_m(q, weight_m, 0);
  for( i = 1; i <= hmm1 -> N; i++){
    RandomSeq(parm_hmm1->c[i], hmm1->mix_comp, q_sub, hmm1->M);
    InitialWeight_gsl_m(q_sub, weight_m_sub, 0);
    gsl_matrix_get_col(col, weight_m, i-1);
    for(j = 1; j <= hmm1->mix_comp; j++){
      gsl_matrix_get_col(col_sub, weight_m_sub, j-1); 
      gsl_vector_mul(col_sub, col);
      gsl_matrix_set_col(weight_m_all, (i-1)*hmm1->mix_comp  + (j-1), col_sub);
    }
  }
   
  if(0) gsl_matrix_print(weight_m_all);   
  WeightedMean_gsl_matrix(X, weight_m_all, parm_hmm1, ERROR_IND); 
  WeightedVar_gsl_matrix(X, weight_m_all, parm_hmm1, ERROR_IND);

   for( i  = 1 ; i<=weight_m_all ->size2; i++){
     if(ERROR_IND[0]) break;
     mvdnorm_matrix_multiMVN(X, parm_hmm1->mu[i], parm_hmm1->Sigma[i], i, hmm1, ERROR_IND);
     
   }
   
   if(0){
     for( i = 1; i <= hmm1->N; i++){
       printf("State %d B1 \n", i);
       dmatrix_print(hmm1->B1[i], hmm1->M, hmm1->mix_comp);
     }
   }

  
   gsl_matrix_free(weight_m_all);
   gsl_matrix_free(weight_m);
   gsl_matrix_free(weight_m_sub);
   gsl_vector_free(col);
   gsl_vector_free(col_sub);
   free_ivector(q_sub, 1, hmm1->M);
   
}
 
void mvdnorm_matrix_multiMVN(gsl_matrix *x, double *mu_vector, double **Sigma, int col, HMM *hmm1, int *ERROR_IND)
{
 
  int T, row_sq;
  int t;
  gsl_vector *x_row;
  gsl_matrix *A_inv;
  double det;
  int print_ind = 0;
  int state, mix;

  mix = (col-1) % hmm1->mix_comp +1 ;
  state = (col - mix)/hmm1->mix_comp +1;

  if(0) printf("state %d, mix %d\n", state, mix);
  T = (int) x->size1;
  row_sq = (int) x->size2;
  x_row = gsl_vector_alloc(row_sq);
  A_inv =  gsl_matrix_alloc(row_sq, row_sq);
  det = get_det(Sigma, row_sq, A_inv, ERROR_IND);
  
 

  if(print_ind) printf("det = %2.52lf \n", fabs(det));
  for (t = 1; t <= T; t++)
    {
      gsl_matrix_get_row(x_row, x, (size_t) t-1);
      hmm1-> B1[state][t][mix] = mvdnorm_B(x_row,  mu_vector,  det, A_inv, 1); 
 	
    }
 
  
  gsl_matrix_free(A_inv);
  gsl_vector_free(x_row);

}


void LogMaxScaleB1(HMM *hmm1,  int *ERROR_IND){
  double max_t, max_sum;
  double *max_t_vec;
  double ***B_temp;
  int i, t, state;

  
  max_t_vec = dvector(1, hmm1->M);
  B_temp = AllocArray(hmm1->N, hmm1->M, hmm1->mix_comp);
  max_sum = 0;
    for( t = 1; t<= hmm1->M ; t++){
      for(state =1; state <= hmm1->N; state++){
	max_t = hmm1->B1[1][t][1];
	for(i = 1; i <= hmm1-> mix_comp ; i++){
	  max_t = MAX((double) max_t, (double) hmm1->B1[state][t][i]);
	}
      }
      max_t_vec[t] = (double) max_t;

      for(state =1; state <= hmm1->N; state++){
	for(i = 1; i <= hmm1->mix_comp; i++){
	  B_temp[state][t][i]= (double) (hmm1->B1[state][t][i]) ;
	  hmm1->B1[state][t][i] = exp(hmm1->B1[state][t][i] - (double) max_t); 
	}
      }
      max_sum +=(double) max_t;
      /* printf("max_sum and max_t = %d %f and %f\n", t, max_sum, max_t);*/
    }

    hmm1->B_scale =  max_sum;
    /*  max_sum > 0 is not an error   */
	/* 
     f( max_sum > 0){
      printf(" B_scale is positive (LogMaxScaleB1) \n");
      
      file_B = fopen ("B2.tex", "w");
      for(m = 1; m<=hmm1->M ; m++){
	for( m1 = 1; m1 <= hmm1->N; m1++){
	  for(m2 = 1; m2 <= hmm1-> mix_comp; m2++){
	    fprintf(file_B, "%e\t",B_temp[m1][m][m2]);
	  }
	}
	fprintf(file_B, "\n");
      }
      fclose(file_B);
          
      file_B = fopen("max_t","w");
      dvector_fprint(file_B, max_t_vec, hmm1->M);
      fclose(file_B);
     	 
      ERROR_IND[0]=0;
    }
    */
    free_dvector(max_t_vec, 1, hmm1->M);
    free_AllocArray(B_temp, hmm1->N, hmm1->M, hmm1->mix_comp);
    
}


void fixed_initial_values(HMM *hmm1, PARM_HMM *parm_hmm1)
{
  int i,j;
  

      for ( i = 1; i <= hmm1->N ; i++){
	hmm1->pi[i] = 1/(double)hmm1->N;
	for(j = 1; j<= hmm1->N; j++){
	  hmm1->A[i][j] = 1/(double)hmm1->N;
	}
      }
      
      
      if(!strcmp(hmm1->modelname, "multiMVN")){
	for ( i = 1; i <= hmm1->N ; i++){
	  for(j = 1; j <= hmm1->mix_comp; j++){
	    parm_hmm1->c[i][j] = 1/(double)hmm1->mix_comp;
	  }
	}
      }


}


void InitialMultiMVN(gsl_matrix *X, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini)
{

  FILE *file_B;
  int i, j;
  int ERROR_IND[1]={0};
  
         /* Initial Parameter */
       InitialParm(parm_hmm1, X, ini);
	         
       if(0){
	 file_B = fopen("parm.tex", "w");
	 PrintPARM(file_B, parm_hmm1, "non");
	 fclose(file_B);
       }

       if(parm_hmm1->mu_user_para){
	 for(i=1; i<= hmm1->N;i++){
	   for(j=1; j<=ini->colX;j++){
	     parm_hmm1->mu[i][j] = ini->mu[i][j];
	   }
	 }
       }
  
       
       for( i = 1; i <=hmm1->N * hmm1->mix_comp; i++){
	 mvdnorm_matrix_multiMVN(X, parm_hmm1->mu[i], parm_hmm1->Sigma[i], i, hmm1, ERROR_IND);
       }
     
    
  				   
}





void InitialMultiMVN_regression(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini)
{

  gsl_matrix *design_X;
  gsl_vector *X_col;
  gsl_matrix *weight_m;
  int i, ERROR_IND[1]={0};
     

       weight_m = gsl_matrix_alloc(hmm1->M, (hmm1->N)*(hmm1->mix_comp));
       /* Design matrix */
       design_X = gsl_matrix_alloc(hmm1 -> M, X->size2 +1 );
       X_col = gsl_vector_alloc(hmm1->M);
 
       gsl_vector_set_all(X_col, 1);
       gsl_matrix_set_col(design_X, 0, X_col);
       
       for( i = 0; i < X-> size2; i++)
	 {
	   gsl_matrix_get_col(X_col,  X, i);
	   gsl_matrix_set_col(design_X, i+1, X_col );
	 }

       /* Initial Parameter */
       InitialParm(parm_hmm1, X, ini);
	         
       
       for( i = 1; i <=hmm1->N * hmm1->mix_comp; i++){
	 mvdnorm_matrix_multiMVN(X, parm_hmm1->mu[i], parm_hmm1->Sigma[i], i, hmm1, ERROR_IND);
       }
       
       InitialsingleRegression(X, Y, parm_hmm1, hmm1, ini);



       
       gsl_matrix_free(design_X);
       gsl_vector_free(X_col);
       gsl_matrix_free(weight_m);
				   
}

void InitialModelE(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini)
{

  gsl_matrix *design_X;
  gsl_vector *X_col;
  gsl_matrix *weight_m;
  FILE *file_B;
  double time[4] = {0,0,0,0};
  int i, ERROR_IND[1]={0};
     

       weight_m = gsl_matrix_alloc(hmm1->M, (hmm1->N));
       /* Design matrix */
       design_X = gsl_matrix_alloc(hmm1 -> M, X->size2 +1 );
       X_col = gsl_vector_alloc(hmm1->M);
 
       gsl_vector_set_all(X_col, 1);
       gsl_matrix_set_col(design_X, 0, X_col);
       
       for( i = 0; i < X-> size2; i++)
	 {
	   gsl_matrix_get_col(X_col,  X, i);
	   gsl_matrix_set_col(design_X, i+1, X_col );
	 }

       /* Initial Parameter */
       InitialParm(parm_hmm1, X, ini);
       InitialParmSampleX(parm_hmm1, X);
       
       for( i = 1; i <=hmm1->N * hmm1->mix_comp; i++){
	 mvdnorm_matrix_multiMVN(X, parm_hmm1->mu[i], parm_hmm1->Sigma[i], i, hmm1, ERROR_IND);
       }
     
       if(0){
	 file_B = fopen("B.tex", "w");
	 dmatrix_fprint(file_B, hmm1->B1[1], hmm1->M,  hmm1->mix_comp);
	 fclose(file_B);
	 file_B = fopen("B.tex", "a");
	 dmatrix_fprint(file_B, hmm1->B1[2], hmm1->M,  hmm1->mix_comp);
	 fclose(file_B);
       }

     
       gsl_matrix_set_all(weight_m, 1);
       singleMVN_regression(X, Y, design_X,  weight_m, hmm1, parm_hmm1, time, ERROR_IND);
       if(0){
	 file_B = fopen("B2.tex", "w");
	 dmatrix_fprint(file_B, hmm1->B1[1], hmm1->M,  hmm1->mix_comp);
	 fclose(file_B);
	 file_B = fopen("B2.tex", "a");
	 dmatrix_fprint(file_B, hmm1->B1[2], hmm1->M,  hmm1->mix_comp);
	 fclose(file_B);
	 
       }

       if(0){
	 file_B = fopen("reg.prob.tex", "w");
	 PrintHMM(file_B, hmm1);
	 fclose(file_B);
       }
       
     
       LogMaxScaleB1(hmm1, ERROR_IND);
       CalculateB_from_B1(hmm1, parm_hmm1);
       
       gsl_matrix_free(design_X);
       gsl_vector_free(X_col);
       gsl_matrix_free(weight_m);
				   
}

void Initial_logistic(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini)
{

  gsl_matrix *design_X;
  gsl_vector *X_col;
  gsl_matrix *weight_m;
  gsl_matrix *coef_m;
  gsl_vector *weight_col, *Y_col;
  int i, j, r, state, ERROR_IND[1]={0};
  double logistic_tol = .0001;

     
       weight_m = gsl_matrix_alloc(hmm1->M, (hmm1->N));
       weight_col = gsl_vector_alloc(hmm1->M);
       /* Design matrix */
       design_X = gsl_matrix_alloc(hmm1 -> M, X->size2 +1 );
       X_col = gsl_vector_alloc(hmm1->M);
       Y_col = gsl_vector_alloc(hmm1->M);
       coef_m = gsl_matrix_alloc(hmm1->N, X->size2 + 1);
       
       gsl_vector_set_all(X_col, 1);
       gsl_matrix_set_col(design_X, 0, X_col);
       
       
      
       for( i = 0; i < X-> size2; i++)
	 {
	   gsl_matrix_get_col(X_col,  X, i);
	   gsl_matrix_set_col(design_X, i+1, X_col );
	 }

       for( i = 0; i < X-> size1; i++){
	 gsl_vector_set(Y_col,i,  gsl_matrix_get(Y, i,0));
       }

       
       srand(time(NULL));
       for(i = 0; i< weight_m->size1; i++){
	 for(j = 0; j< weight_m->size2; j++){
	   r = rand();
	   gsl_matrix_set(weight_m, i, j, (double) r/ ((double)(RAND_MAX) + 1));
	 }
       }
      
       for(state = 1; state <= hmm1->N; state++){
	 
	 gsl_matrix_get_col(weight_col, weight_m, state-1);
	 logistic_regression(design_X,  Y_col, weight_col, coef_m, state-1, logistic_tol, ERROR_IND);
	 if(ini->fixed_ini_indicator[4]==1){
	   dmatrix_to_gsl_matrix(ini->ini_reg_coef, coef_m);
	 }
	 Logistic_B(Y_col, design_X, coef_m, state-1, hmm1);
	
       }

       LogMaxScaleB(hmm1);
       
       gsl_matrix_free(design_X);
       gsl_vector_free(X_col);
       gsl_matrix_free(weight_m);
				   
}

void InitialsingleRegression(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini)
{

  gsl_matrix *design_X;
  gsl_vector *X_col;
  gsl_matrix *weight_m;
  double time_null[4] = {0,0,0,0};
  int i, j, r, t, ERROR_IND[1]={0}, mix, state;
  double det;
  gsl_multifit_linear_workspace *work;
  gsl_vector *coef = gsl_vector_alloc((X->size2) + 1);
  gsl_vector *resid = gsl_vector_alloc(X->size1);
  gsl_matrix *resid_matrix = gsl_matrix_alloc(X->size1, Y->size2);
  gsl_matrix *cov = gsl_matrix_alloc( (X->size2) + 1, (X -> size2) + 1);
  gsl_vector *y = gsl_vector_alloc(Y-> size1);
  gsl_vector *y_row;
  gsl_matrix *A_inv = gsl_matrix_alloc(Y->size2, Y->size2);
  gsl_vector *w = gsl_vector_alloc(Y->size1);
  double *resid_mean, chisq;


       resid_mean = (double *) dvector(1, Y->size2);
       gsl_vector_set_all(w, 1);
       if(!strcmp(hmm1->modelname, "multiMVN_regression_state")){
	 weight_m = gsl_matrix_alloc(hmm1->M, hmm1->N);} else {
	 weight_m = gsl_matrix_alloc(hmm1->M,  hmm1->mix_comp *hmm1->N);}
       /* Design matrix */
       design_X = gsl_matrix_alloc(hmm1 -> M, X->size2 +1 );
       X_col = gsl_vector_alloc(hmm1->M);
 
       gsl_vector_set_all(X_col, 1);
       gsl_matrix_set_col(design_X, 0, X_col);
       
       for( i = 0; i < X-> size2; i++)
	 {
	   gsl_matrix_get_col(X_col,  X, i);
	   gsl_matrix_set_col(design_X, i+1, X_col );
	 }

       srand(time(NULL));
       for(i = 0; i< weight_m->size1; i++){
	 for(j = 0; j< weight_m->size2; j++){
	   r = rand();
	   gsl_matrix_set(weight_m, i, j, (double) r/ ((double)(RAND_MAX) + 1));
	 }
       }
     

       if((ini->fixed_ini_indicator[4]!=1) & (ini->fixed_ini_indicator[5]!=1))
	 {
	   singleMVN_regression(X, Y, design_X,  weight_m, hmm1, parm_hmm1, time_null, ERROR_IND);
	  
	 }

       if((ini->fixed_ini_indicator[4]==1) & (ini->fixed_ini_indicator[5]==1))
	 {
	   work =  gsl_multifit_linear_alloc(X->size1, (X->size2) +1 );
	   y_row = gsl_vector_alloc(Y->size2);
	   for(j = 0; j<  weight_m->size2 ; j++){
	     for( i = 0; i < Y->size2; i++){
	       gsl_matrix_get_col(y, Y, i);
	       dvector_to_gsl_vector(ini->ini_reg_coef[j*(Y->size2) +i+1], coef);
	       gsl_multifit_linear_residuals(design_X, y, coef, resid);
	       gsl_matrix_set_col(resid_matrix, i, resid);
	       resid_mean[i+1]= gsl_vector_sum(resid)/(resid->size);
	     }
	     
	     det = get_det(ini->ini_reg_cov[j+1], Y->size2, A_inv, ERROR_IND);
	     
	     if(!strcmp(hmm1->modelname, "singleMVN_regression")){ 
	       for (t = 1; t <= Y->size1; t++)      {
		 gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		 hmm1-> B[t][j+1] +=mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
	       }
	     } 
	     if(!strcmp(hmm1->modelname, "single_regression")){
	       for (t = 1; t <= Y->size1; t++)      {
		 gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		 hmm1-> B[t][j+1] =mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
	       }
	     }

	     if(!strcmp(hmm1->modelname, "multiMVN_regression")){
	       
	       mix = ((j+1)-1) % hmm1->mix_comp +1 ;
	       state = ((j+1)-mix)/hmm1->mix_comp +1;

	       for (t = 1; t <= Y->size1; t++)      {
		 gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		 hmm1-> B1[state][t][mix] +=mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
	       }
	     }

	    if(!strcmp(hmm1->modelname, "multiMVN_regression_state")){
	      state = j+1;
	      for (t = 1; t <= Y->size1; t++)      {
		gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		for( mix = 1; mix <= hmm1->mix_comp; mix++){
		  hmm1-> B1[state][t][mix] +=mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
		}
	      }
	    }
	     
	   }
	 }

       if( (ini->fixed_ini_indicator[4]==1) & (ini->fixed_ini_indicator[5]!=1))
	 {
	  
	    work =  gsl_multifit_linear_alloc(X->size1, (X->size2) +1 );
	    y_row = gsl_vector_alloc(Y->size2);
	   for(j = 0; j<  weight_m->size2 ; j++){
	     for( i = 0; i < Y->size2; i++){
	       gsl_matrix_get_col(y, Y, i);
	       dvector_to_gsl_vector(ini->ini_reg_coef[j*(Y->size2) +i+1], coef);
	       gsl_multifit_linear_residuals(design_X, y, coef, resid);
	       gsl_matrix_set_col(resid_matrix, i, resid);
	       resid_mean[i+1]= gsl_vector_sum(resid)/(resid->size);
	     }
	     
	     WeightedVar_gsl(resid_matrix, resid_mean, w, resid_matrix->size1, 
		      resid_matrix->size2, parm_hmm1->reg_cov[j+1], ERROR_IND);
     
	     
	     det = get_det(parm_hmm1-> reg_cov[j+1], Y->size2, A_inv, ERROR_IND);
	     if(!strcmp(hmm1->modelname, "singleMVN_regression")){ 
	       for (t = 1; t <= Y->size1; t++)      {
		 gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		 hmm1-> B[t][j+1]+=mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
	       }
	     } 
	     if(!strcmp(hmm1->modelname, "single_regression")){
	       for (t = 1; t <= Y->size1; t++)      {
		 gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		 hmm1-> B[t][j+1] =mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
	       }
	     }
	   
	    if(!strcmp(hmm1->modelname, "multiMVN_regression")){
	       
	       mix = ((j+1)-1) % hmm1->mix_comp +1 ;
	       state = ((j+1)-mix)/hmm1->mix_comp +1;

	       for (t = 1; t <= Y->size1; t++)      {
		 gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		 hmm1-> B1[state][t][mix] +=mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
	       }
	    }


	    if(!strcmp(hmm1->modelname, "multiMVN_regression_state")){
	      state = j+1;
	      for (t = 1; t <= Y->size1; t++)      {
		gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		for( mix = 1; mix <= hmm1->mix_comp; mix++){
		  hmm1-> B1[state][t][mix] +=mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
		}
	      }
	    }

	   }
	 }
       
       if((ini->fixed_ini_indicator[4]!=1 )& (ini->fixed_ini_indicator[5]==1))
	 {
	   work =  gsl_multifit_linear_alloc(X->size1, (X->size2) +1 );
	    y_row = gsl_vector_alloc(Y->size2);
	   for(j = 0; weight_m->size2 ; j++){
	     for( i = 0; i < Y->size2; i++){
	       gsl_matrix_get_col(y, Y, i);
	       gsl_matrix_get_col(w, weight_m, j);
	       gsl_multifit_wlinear(design_X, w, y, coef, cov,  &chisq, work);
	       gsl_vector_to_dvector(coef, parm_hmm1->reg_coef[j*(Y->size2) +i+1]);  
	       gsl_multifit_linear_residuals(design_X, y, coef, resid);
	       gsl_matrix_set_col(resid_matrix, i, resid);
	       resid_mean[i+1]= gsl_vector_sum(resid)/(resid->size);
	     }

	    
	     det = get_det(ini-> ini_reg_cov[j+1], Y->size2, A_inv, ERROR_IND);
	     
	     if(!strcmp(hmm1->modelname, "singleMVN_regression")){ 
	       for (t = 1; t <= Y->size1; t++)      {
		 gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		 hmm1-> B[t][j+1] +=mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
	       }
	     } 
	     if(!strcmp(hmm1->modelname, "single_regression")){ 
	       for (t = 1; t <= Y->size1; t++)      {
		 gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		 hmm1-> B[t][j+1] =mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
	       }
	     }
	     
	   if(!strcmp(hmm1->modelname, "multiMVN_regression")){
	     
	      mix = ((j+1)-1) % hmm1->mix_comp +1 ;
	      state = ((j+1)-mix)/hmm1->mix_comp +1;

	      for (t = 1; t <= Y->size1; t++){
		  gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		  hmm1-> B1[state][t][mix] += mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
	      }
	    }
	      

	    if(!strcmp(hmm1->modelname, "multiMVN_regression_state")){
	      state = j+1;
	      for (t = 1; t <= Y->size1; t++)      {
		gsl_matrix_get_row(y_row, Y, (size_t) t-1);
		for( mix = 1; mix <= hmm1->mix_comp; mix++){
		  hmm1-> B1[state][t][mix] +=mvdnorm_B(y_row,  resid_mean, det, A_inv, 1);
		}
	      }
	    }
	    
	    


 
	   }
	 }
       

       gsl_matrix_free(design_X);
       gsl_vector_free(X_col);
       gsl_matrix_free(weight_m);
       gsl_vector_free(coef);
       gsl_vector_free(resid);
       gsl_matrix_free(resid_matrix);
       gsl_matrix_free(cov);
       gsl_vector_free(y);
       gsl_matrix_free(A_inv);
       
       if(!((ini->fixed_ini_indicator[4]!=1) & (ini->fixed_ini_indicator[5]!=1))){
	  gsl_vector_free(y_row);
	  gsl_multifit_linear_free(work);
	  }
    
       gsl_vector_free(w);

       free_dvector(resid_mean, 1, Y->size2);

				   
}


void InitialsingleLogistic(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini)
{

  gsl_matrix *design_X;
  gsl_vector *X_col;
  gsl_matrix *weight_m;
  gsl_matrix *coef_m;
  gsl_vector *weight_col, *Y_col;
  int i, j, r, state, ERROR_IND[1]={0};
  double logistic_tol = .0001;
  FILE *fp;
     
       weight_m = gsl_matrix_alloc(hmm1->M, (hmm1->N));
       weight_col = gsl_vector_alloc(hmm1->M);
       /* Design matrix */
       design_X = gsl_matrix_alloc(hmm1 -> M, X->size2 +1 );
       X_col = gsl_vector_alloc(hmm1->M);
       Y_col = gsl_vector_alloc(hmm1->M);
       coef_m = gsl_matrix_alloc(hmm1->N, X->size2 + 1);
       
       gsl_vector_set_all(X_col, 1);
       gsl_matrix_set_col(design_X, 0, X_col);
       
       
      
       for( i = 0; i < X-> size2; i++)
	 {
	   gsl_matrix_get_col(X_col,  X, i);
	   gsl_matrix_set_col(design_X, i+1, X_col );
	 }

       for( i = 0; i < X-> size1; i++){
	 gsl_vector_set(Y_col,i,  gsl_matrix_get(Y, i,0));
       }

       
       srand(time(NULL));
       for(i = 0; i< weight_m->size1; i++){
	 for(j = 0; j< weight_m->size2; j++){
	   r = rand();
	   gsl_matrix_set(weight_m, i, j, (double) r/ ((double)(RAND_MAX) + 1));
	 }
       }

       fp = fopen("weight", "w");
       gsl_matrix_fprintf(fp, weight_m, "%f");
       fclose(fp);

      
       for(state = 1; state <= hmm1->N; state++){
	 
	 gsl_matrix_get_col(weight_col, weight_m, state-1);
	 logistic_regression(design_X,  Y_col, weight_col, coef_m, state-1, logistic_tol, ERROR_IND);
	 if(ini->fixed_ini_indicator[4]==1){
	   dmatrix_to_gsl_matrix(ini->ini_reg_coef, coef_m);
	 }
	 Logistic_B(Y_col, design_X, coef_m, state-1, hmm1);
       }

       LogMaxScaleB(hmm1);
       
       gsl_matrix_free(design_X);
       gsl_vector_free(X_col);
       gsl_matrix_free(weight_m);
				   
}

void regression_coef(gsl_matrix *X, gsl_vector *y, gsl_vector *W, gsl_vector *coef){
  int i, j, k;
  gsl_matrix *temp_inv = gsl_matrix_alloc(X->size2, X->size2);
  gsl_matrix *inv = gsl_matrix_alloc(X->size2, X->size2);
  gsl_vector *temp_vec = gsl_vector_alloc(X->size2);
  gsl_permutation *p = gsl_permutation_alloc(X->size2);
  int s;
  double temp, temp1;

  /* X'WX */
  for(i = 0 ; i < X->size2; i++){
    for( j = 0; j< X->size2 ; j++){
      temp = 0;
      for( k = 0; k < X->size1; k++){
	temp += gsl_matrix_get(X, k , i) * gsl_vector_get(W,k) * gsl_matrix_get(X, k, j);
      }
      	gsl_matrix_set(temp_inv, i,j,temp);
    }
  }

  /* X'W y */

  for(i = 0; i < X->size2; i++){
    temp1 = 0;
    for(j = 0; j < X->size1; j++){
      temp1 += gsl_matrix_get(X, j, i) * gsl_vector_get(W,j) * gsl_vector_get(y, j);
    }
    gsl_vector_set(temp_vec, i, temp1);
  }
  
  /* solve(X'WX) */
  
  gsl_linalg_LU_decomp(temp_inv, p, &s);
  gsl_linalg_LU_invert(temp_inv, p, inv);  
  

  /* solve(X'WX)*X'Wy */

  gsl_matrix_vector_multiplication(inv, temp_vec, inv->size1, coef);



  gsl_matrix_free(temp_inv);
  gsl_matrix_free(inv);
  gsl_vector_free(temp_vec);
  gsl_permutation_free(p);

}



void CountStateNumber(int *q, int M, gsl_vector *count_state_number)
{
  int i, state, prev_count;
  
  gsl_vector_set_all(count_state_number,0);
  
  for (i = 1; i <= M; i++){
    state = q[i];
    prev_count = gsl_vector_get(count_state_number, state - 1);
    gsl_vector_set(count_state_number, state-1, prev_count + 1);
  }
}
   

void hardEM_multiMVN_regression_new(gsl_matrix *design_X, gsl_matrix *Y, gsl_matrix *weight_m, int *q, HMM *phmm, PARM_HMM *parm_hmm1, int *ERROR_IND)
{
  
  gsl_vector *count_state_number = gsl_vector_alloc(phmm->N);
  int state;
  int tid, chunk=1, nthreads ;

  CountStateNumber(q, Y->size1, count_state_number);
  gsl_vector_print(count_state_number);
  
  if(gsl_vector_min(count_state_number)< design_X->size2){
    printf("The number of observations in a certain state is not large enough to estimate the regression coefficients ( less than %d )", (int) design_X -> size2);
    gsl_vector_print(count_state_number);
    ERROR_IND[0]=1;
    return;
  }
  

#pragma omp parallel shared(count_state_number, q, design_X, Y, weight_m, phmm, parm_hmm1, ERROR_IND, nthreads, chunk) private(state, tid)
  {
    nthreads = omp_get_num_threads();
    tid = omp_get_thread_num();
    /* printf("The number of threads = %d\n", nthreads);
       printf("Thread %d starting ... \n",tid );
    */
#pragma omp for schedule (static, chunk)
    for( state = 1; state <= phmm->N ; state++){
      hardEM_multiMVN_regression_parallel( state,  count_state_number,  q,  design_X, Y,  weight_m, phmm, parm_hmm1, ERROR_IND);
    }
  }
  
gsl_vector_free(count_state_number);
    
}


void hardEM_multiMVN_regression_parallel(int state, gsl_vector *count_state_number, int *q, gsl_matrix *design_X, gsl_matrix *Y, gsl_matrix *weight_m, HMM *phmm, PARM_HMM *parm_hmm1, int *ERROR_IND)
{
  
  gsl_matrix *X_temp, *Y_temp, *w_temp, *resid_temp_matrix;
  gsl_vector *resid_temp;
  gsl_vector *resid = gsl_vector_alloc(design_X->size1);
  gsl_vector *X_row = gsl_vector_alloc(design_X ->size2);
  gsl_vector *Y_row = gsl_vector_alloc(Y -> size2);
  gsl_vector *w_row = gsl_vector_alloc(weight_m -> size2);
  gsl_multifit_linear_workspace *work;
  gsl_matrix *cov = gsl_matrix_alloc( design_X ->size2, design_X->size2);
  gsl_matrix *resid_matrix = gsl_matrix_alloc(design_X->size1, Y->size2);
  gsl_vector *coef = gsl_vector_alloc(design_X ->size2);
  gsl_vector *Y_col = gsl_vector_alloc(Y->size1);
  gsl_vector *w_temp_col, *Y_temp_col;
  double chisq;
  double *mean_0;
  int row_temp, count, i, mix_comp, weight_col_num_gsl;
		
  mean_0 = (double *) dvector(1, Y->size2);  
  for( i = 1; i<= Y->size2;i++){
    mean_0[i]=(double) 0;
  }
  
  
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
    

  gsl_vector_free(X_row);
  gsl_vector_free(Y_row);
  gsl_vector_free(w_row);
  
  gsl_matrix_free(resid_matrix);
  gsl_vector_free(coef);
  gsl_matrix_free(cov);
  gsl_multifit_linear_free(work);

}



void logistic_regression(gsl_matrix *design_X, gsl_vector *y_dich, gsl_vector *weight, gsl_matrix *coef_m, int state, double logistic_tol, int *ERROR_IND )
{
  int i;

  gsl_multifit_linear_workspace *work =  gsl_multifit_linear_alloc(design_X->size1, design_X->size2);;
  double chisq;
  gsl_matrix *cov = gsl_matrix_alloc( design_X ->size2, design_X->size2);
  gsl_vector *coef = gsl_vector_alloc(design_X ->size2);
  gsl_vector *coef_new = gsl_vector_alloc(design_X ->size2);
  gsl_vector *y_new = gsl_vector_alloc(y_dich->size);
  gsl_vector *X_row = gsl_vector_alloc(design_X -> size2); 
  gsl_vector *X_gamma = gsl_vector_alloc(design_X->size1);
  gsl_vector *h_ij = gsl_vector_alloc(design_X->size1);
  gsl_vector *W_ij_inv = gsl_vector_alloc(design_X -> size1);
  gsl_vector *W_ij = gsl_vector_alloc(design_X -> size1);
  gsl_rng_type *T;
  gsl_rng *rng;
  double a;


  gsl_rng_env_setup();
  T = gsl_rng_default;
  rng = gsl_rng_alloc(T);


  /* Initialize Coefficient */
  gsl_vector_set(coef, 0, log(gsl_vector_sum(y_dich)) - log((double) y_dich -> size));
  
  for( i = 1; i < design_X ->size2; i++){
    gsl_vector_set(coef, i, 0);
  }
  if(0) {
    printf("Initial coef\n");
    gsl_vector_print(coef);
  }
  

  /* Initial reweighted y  = W_i^{-1} G_i (y - h_i)  */
  



  /*
     W_ij = h_ij * (1- h_ij) * g_ij 
     h_ij = exp(gamma' X_j)/(1 + exp(gamma' X_j)  : gamma = coef
     G_i = P(q_t = s_i) = weight 
  */

  for( i = 0; i < y_dich->size ; i++){
    gsl_vector_set(y_new, i, gsl_vector_get(y_dich,i));
  }
  
 
  do {
  /* exp(X gamma ) */
    for( i = 0; i < y_dich->size; i++){
      gsl_matrix_get_row(X_row, design_X, i); 
      gsl_vector_mul(X_row, coef); 
      a =  (double) exp(gsl_vector_sum(X_row))/(1+exp(gsl_vector_sum(X_row)));
      gsl_vector_set(X_gamma, i,  a); 
      gsl_vector_set(h_ij, i, a); 
      
    }


    /* W_ij */
    for( i = 0; i < y_dich->size; i++){
      a = gsl_vector_get(h_ij, i);
      gsl_vector_set(W_ij_inv, i, pow(a*(1-a)*gsl_vector_get(weight,i), -1));
      gsl_vector_set(W_ij, i, a*(1-a)*gsl_vector_get(weight,i));
      gsl_vector_set(y_new, i, gsl_vector_get(y_dich,i));
    }


    /* h_ij */
    gsl_vector_sub(y_new, h_ij);
    gsl_vector_mul(y_new, weight);
    gsl_vector_mul(y_new, W_ij_inv);


    gsl_multifit_wlinear(design_X, W_ij, y_new, coef_new, cov, &chisq, work);  


    if(0){
      printf("new_coef\n");
      gsl_vector_print(coef_new);
    }
    gsl_vector_add(coef, coef_new);
    if(0){
      printf("coef \n");
      gsl_vector_print(coef);
    }
    gsl_vector_mul(coef_new, coef_new);


    if(isnan(gsl_vector_sum(coef_new))){
      printf("Logistic Regression failed to converge.\n");
      ERROR_IND[0]=1;
      /* printf("coef_new:\n");
      gsl_vector_print(coef_new);

      coef_new->data[0] = 1;
      gsl_vector_set(coef, 0, log((double) gsl_vector_sum(y_dich)) - log((double) y_dich -> size));
      
      for( i = 1; i < design_X ->size2; i++)
	{
	  gsl_vector_set(coef, i, gsl_ran_ugaussian(rng));
	  coef_new->data[i] = logistic_tol;
	}
           
  
      printf("New initial value is set: \n");
      gsl_vector_print(coef);*/
    }
      
  } while (gsl_vector_sum(coef_new) >logistic_tol);

  gsl_matrix_set_row(coef_m,  state, coef);

  gsl_vector_free(coef_new);
  gsl_vector_free(y_new);
  gsl_vector_free(X_row);
  gsl_vector_free(X_gamma);
  gsl_vector_free(h_ij);
  gsl_vector_free(W_ij_inv);
    
}

void Logistic_B(gsl_vector *y, gsl_matrix *design_X, gsl_matrix *coef_m, int state,  HMM *hmm1)
{ 
  int i;
  gsl_vector *X_row = gsl_vector_alloc(design_X ->size2);
  gsl_vector *coef_state = gsl_vector_alloc(design_X ->size2);
  double temp;

  gsl_matrix_get_row(coef_state, coef_m, (size_t) state);
  
  for( i = 0; i  < design_X->size1 ; i++)
    {
      gsl_matrix_get_row(X_row,  design_X, (size_t) i);
      gsl_vector_mul(X_row, coef_state);
      temp  = gsl_vector_sum(X_row);
      if(!strcmp(hmm1->modelname, "single_logistic")){
	if(gsl_vector_get(y, i) == 1){
	  hmm1->B[i+1][state +1] =  temp - log(1 + exp(temp));
	} else {
	  hmm1->B[i+1][state+1] = -log(1+exp(temp));}
      }
      if(!strcmp(hmm1->modelname, "singleMVN_logistic")){
	if(gsl_vector_get(y, i) == 1){
	  hmm1->B[i+1][state +1] += ( temp - log(1 + exp(temp)));
	} else {
	  hmm1->B[i+1][state+1] += -log(1+exp(temp));}
      }
    
    }

  gsl_vector_free(X_row);
  gsl_vector_free(coef_state);
}

void replace_parameters(PARM_HMM *hmm1_para, INI *ini)
{
  int i,j,k,l;
  if(hmm1_para-> reg_coef_user_num !=0){
    k=0;
    for(i = 1; i <= hmm1_para->reg_coef_user_num; i++){
      for(j = 1 ; j <= hmm1_para -> p ; j++){
	hmm1_para->reg_coef[i][j] = ini->reg_coef[i][j];
	k++;
      }
    }
  }

  if(hmm1_para-> reg_cov_user_para !=0){
    l=0;
    for(i = 1; i <= hmm1_para->reg_cov_user_para; i++){
      for(j = 1 ; j <=hmm1_para -> d ; j++){ 
	for(k = 1 ; k <=hmm1_para -> d ; k++){
	  hmm1_para->reg_cov[i][j][k] = ini-> reg_cov[i][j][k];
	}
      }
    }
  }

  if(hmm1_para-> mu_user_para !=0){
    k=0;
    for(i = 1; i <= hmm1_para->mu_user_para; i++){
      for(j = 1 ; j <=hmm1_para -> p ; j++){
	hmm1_para->mu[i][j] = ini->mu[i][j];
      }
    }
  }
  
  if(hmm1_para-> Sigma_user_para !=0){
    l=0;
    for(i = 1; i <= hmm1_para-> Sigma_user_para; i++){
      for(j = 1 ; j <=hmm1_para -> p ; j++){ 
	for(k = 1 ; k <=hmm1_para -> p ; k++){
	hmm1_para->Sigma[i][j][k] = ini-> Sigma[i][j][k];
	l++;
	}
      }
    }
  }
 if(hmm1_para-> c_user_para !=0){
    k=0;
    for(i = 1; i <= hmm1_para-> c_user_para; i++){
      for(j = 1 ; j <=hmm1_para -> mix_comp ; j++){
	hmm1_para->c[i][j] = ini-> c[i][j];
	k++;
      }
    }
  }
}


void combined_singleMVN_regression(gsl_matrix *X, gsl_matrix *Y,gsl_matrix *design_X,  gsl_matrix *weight_m,  HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, double *time, int *ERROR_IND){
  if(parm_hmm1-> user_para ==1 ){
    user_defined_X_singleMVN_Regression(X, Y, design_X, weight_m, phmm, parm_hmm1, ini, time, ERROR_IND);
  } else {
    singleMVN_regression(X, Y, design_X, weight_m, 
			 phmm, parm_hmm1, time, ERROR_IND);
  }
}


void Observation_prob(gsl_matrix *X, gsl_matrix *design_X, gsl_matrix *Y, gsl_vector *Y_col,gsl_matrix *weight_m,  gsl_matrix *coef_m, gsl_matrix *gamma_ij, HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, double *time, int *ERROR_IND){

  /* Find the observationi probability to write on  output file */
  int state;
  gsl_vector *weight_col;
  double logistic_tol = .00001;

   weight_col = gsl_vector_alloc(X -> size1);

  if(!strcmp(phmm->modelname, "singleMVN")){
    singleMVN(X, weight_m, phmm, parm_hmm1, ini,  ERROR_IND);  
  }
  
  if(!strcmp(phmm-> modelname, "multiMVN")){
    multiMVN(X, weight_m, phmm, parm_hmm1, ini, ERROR_IND);
    CalculateB_from_B1_Log(phmm, parm_hmm1);
  }
	
  if(!strcmp(phmm-> modelname, "singleMVN_regression")){
    singleMVN(X, weight_m, phmm, parm_hmm1, ini, ERROR_IND);
    combined_singleMVN_regression(X, Y, design_X, weight_m, phmm, parm_hmm1, ini, time, ERROR_IND);
  }


  if(!strcmp(phmm-> modelname, "single_regression")){
    combined_singleMVN_regression(X, Y, design_X, weight_m, phmm, parm_hmm1, ini, time, ERROR_IND);
  }
	
  if(!strcmp(phmm-> modelname, "single_logistic")){
    for(state = 1; state <= phmm->N; state++){
      gsl_matrix_get_col(weight_col, weight_m, state-1);
      logistic_regression(design_X,  Y_col, weight_col, coef_m, 
			  state-1, logistic_tol, ERROR_IND);
      Logistic_B(Y_col, design_X, coef_m, state-1, phmm);
    }
   
     
  }
  if(!strcmp(phmm-> modelname, "singleMVN_logistic")){
    
    singleMVN(X, weight_m, phmm, parm_hmm1, ini, ERROR_IND);
    
    for(state = 1; state <= phmm->N; state++){
      gsl_matrix_get_col(weight_col, weight_m, state-1);
      
      logistic_regression(design_X,  Y_col, weight_col, coef_m, 
			  state-1, logistic_tol, ERROR_IND);
      
      if(ERROR_IND[0]) break;
      
      Logistic_B(Y_col, design_X, coef_m, state-1, phmm);
      
    }
    
     
  }
						

  if(!strcmp(phmm-> modelname, "multiMVN_regression_state")){
    multiMVN(X, weight_m, phmm, parm_hmm1, ini, ERROR_IND);
    combined_singleMVN_regression(X, Y, design_X,  weight_m, 
				  phmm, parm_hmm1, ini, time, 
				  ERROR_IND);
  }

  if(!strcmp(phmm-> modelname, "multiMVN_regression")){
    multiMVN_weight(X, weight_m, phmm, parm_hmm1,gamma_ij, time,
		    ini,  ERROR_IND);
    combined_singleMVN_regression(X, Y, design_X, gamma_ij, 
				  phmm, parm_hmm1,ini, time, 
				  ERROR_IND);
    CalculateB_from_B1_Log(phmm, parm_hmm1);
  }
  
}
