#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_multifit.h>


/* Functions are used to evaluate emission probability for each model */
void multiMVN(gsl_matrix *X, gsl_matrix *weight_m, HMM *phmm, PARM_HMM *parm_HMM1, INI *ini, int *ERROR_IND);
void singleMVN(gsl_matrix  *X, gsl_matrix *weight_m, HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, int *ERROR_IND);
void singleMVN_regression(gsl_matrix *X, gsl_matrix *Y,gsl_matrix *design_X,  gsl_matrix *weight_m,  HMM *phmm, PARM_HMM *parm_hmm1, double *time, int *ERROR_IND);
void singleMVN_regression_no_intercept( gsl_matrix *X, gsl_matrix *Y, gsl_matrix *weight_m,  HMM *phmm, PARM_HMM *parm_hmm1,double *time, int *ERROR_IND) ;
void Logistic_B(gsl_vector *y, gsl_matrix *design_X, gsl_matrix *coef, int state,  HMM *hmm1);
     /* Logistic Regression analysis */
void logistic_regression(gsl_matrix *design_X, gsl_vector *y_dich, gsl_vector *weight, gsl_matrix *coef_m, int state, double logistic_tol, int *ERROR_IND );

/* For the user defined parameter model */
void user_defined_X_singleMVN_Regression(gsl_matrix *X, gsl_matrix *Y,gsl_matrix *design_X,  gsl_matrix *weight_m,  HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, double *time, int *ERROR_IND);
/* Wrapping function : 
   user_defined_X_singleMVN_Regression and singleMVN_regession*/ 
void combined_singleMVN_regression(gsl_matrix *X, gsl_matrix *Y, gsl_matrix *design_X,  gsl_matrix *weight_m,  HMM *phmm, PARM_HMM *parm_hmm1, INI *ini,  double *time, int *ERROR_IND);


/* Calculate log density of MVN */
void mvdnorm_matrix(gsl_matrix *x, double *mu_vector, double **Sigma, int state, HMM *hmm1, int *ERROR_IND);
void mvdnorm_matrix_regression(gsl_matrix *X, int state, PARM_HMM *parm_hmm1,  HMM *hmm1, int *ERROR_IND);
/* For mixture model */
      /* Evaluate log density of mixture model */
void mvdnorm_matrix_multiMVN(gsl_matrix *x, double *mu_vector, double **Sigma, int col, HMM *hmm1, int *ERROR_IND);
      /* Find c[i][j] */
void multiMVN_weight_gamma_ij(gsl_matrix *X, int i, gsl_matrix *weight_m, gsl_matrix *gamma_ij, HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, int *ERROR_IND );
       /* Wrapping function:
	  mvdnorm_matrix_multiMVN and  multiMVN_weight_gamma_ij */
void multiMVN_weight(gsl_matrix *X, gsl_matrix *weight_m, HMM *phmm, PARM_HMM *parm_hmm1, gsl_matrix *gamma_ij, double *time, INI *ini, int *ERROR_IND);




/* Calculate initial mean and variance based on initial states */
void Initial_state_singleMVN(HMM *hmm1, PARM_HMM *parm_hmm1, int *q, gsl_matrix *X);
/* Calculate initial mean and variance based on initial states */
void Initial_state_multiMVN(HMM *hmm1, PARM_HMM *parm_hmm1, int *q, gsl_matrix *X);


/* Obtain Initial values for the model */
void InitialMultiMVN(gsl_matrix *X,PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini);
void InitialMultiMVN_regression(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini);
void InitialsingleRegression(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini);
void InitialsingleLogistic(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini);
void Initial_logistic(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini);

/* Fix initial pi, A, c */
void fixed_initial_values(HMM *hmm1, PARM_HMM *parm_hmm1);


/* Used for scaling the emission probability */
void LogMaxScaleB(HMM *hmm1);
void LogMaxScaleB1(HMM *hmm1, int *ERROR_IND);


/* Not used in MRHMMs */
void InitialModelE(gsl_matrix *X, gsl_matrix *Y, PARM_HMM *parm_hmm1, HMM *hmm1, INI *ini);

void CountStateNumber(int *q, int M, gsl_vector *count_state_number);

void MakeDesignMatrix(gsl_matrix *X, gsl_matrix *design_X);
void hardEM_multiMVN_regression(gsl_matrix *design_X, gsl_matrix *Y, gsl_matrix *weight_m, int *q, HMM *phmm, PARM_HMM *parm_hmm, int *ERROR_IND);
void hardEM_multiMVN_regression_new(gsl_matrix *design_X, gsl_matrix *Y, gsl_matrix *weight_m, int *q, HMM *phmm, PARM_HMM *parm_hmm, int *ERROR_IND);
void hardEM_multiMVN_regression_parallel(int state, gsl_vector *count_state_number, int *q, gsl_matrix *design_X, gsl_matrix *Y, gsl_matrix *weight_m, HMM *phmm, PARM_HMM *parm_hmm1, int *ERROR_IND);

void Observation_prob(gsl_matrix *X, gsl_matrix *design_X, gsl_matrix *Y, gsl_vector *Y_col,gsl_matrix *weight_m,  gsl_matrix *coef_m, gsl_matrix *gamma_ij, HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, double *time, int *ERROR_IND);
