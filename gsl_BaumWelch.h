
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>



void BaumWelch(HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, gsl_matrix *X, gsl_matrix *Y, int T,  double **alpha, double **beta, double **gamma, int *niter,  double *plogprobinit, double *plogprobfinal, gsl_vector *dist,  int *ERROR_IND);
void BaumWelch_itr(int itr, HMM *hmm1, PARM_HMM *parm_hmm1, INI *ini, gsl_matrix *X, gsl_matrix *Y, int T,  double **alpha, double **beta, double **gamma, int *niter, int Max_itr, double Tolerance, double *logprobprevItr,  int *ERROR_IND);
void Read_multiple_data_sets(INI *ini, gsl_matrix *X, gsl_matrix *Y, int data_set_number, int read_Y_IND);
