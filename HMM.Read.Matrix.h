/* Include functisn in HMM.Read.Matrix.c */

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#include <stdio.h>
#include <stdlib.h>
#include "nrutil.h"
#include "_struct.h"

/* HMM.Read.Matrix.c */
int read_initial(char *int_location, INI *ini,  HMM *hmm1, PARM_HMM *hmm1_parm);
void PrintInitial(INI *ini, HMM *hmm1);

void dReadMatrix(char *file_location, int num_row, int num_col, double **mtx);
void dReadVector(char *file_location, int num, double *vec);
void ReadInitialParm(HMM *hmm, PARM_HMM *parm_hmm, INI *ini);

void InitialSeq(HMM *hmm1, int *q); 
/* InitialSeq : the output q is a state sequence that is randomly selected based on the initial probabilities and initial transition probabilities */
void RandomSeq(double *p, int len_p, int *q, int len_q);
void InitialWeight_gsl_m( int *q,  gsl_matrix *weight_m, int print_ind);
void Initial_Uniform (HMM *hmm1);
void InitialC (PARM_HMM *parm_hmm1);
void InitialA(HMM *hmm1, INI *ini);
void InitialParm(PARM_HMM *parm_hmm1, gsl_matrix *X, INI *ini);
void InitialParmSampleX(PARM_HMM *parm_hmm1, gsl_matrix *X);


double WeightedMean_gsl(gsl_vector *data, gsl_vector *weight, int len, int *ERROR_IND);
double WeightedCov_gsl(gsl_vector *data1, gsl_vector *data2, double mean1, double mean2,  gsl_vector *weight, int len, int *ERROR_IND);
void WeightedVar_gsl(gsl_matrix *Data, double *mean1,  gsl_vector *weight, int len, double num_col_data, double **Var, int *ERROR_IND);


void WeightedMean_gsl_matrix(gsl_matrix *X, gsl_matrix *weight_m, PARM_HMM *para_hmm1, int *ERROR_IND);
void WeightedVar_gsl_matrix(gsl_matrix *X, gsl_matrix *weight_m, PARM_HMM *para_hmm1, int *ERROR_IND);

/* gsl vector and matrix calculation */
double gsl_vector_sum(gsl_vector *a);

/* Print and fprintf */
void gsl_matrix_print(gsl_matrix *X);
void gsl_vector_print(gsl_vector *v);

/* gsl and regular vector-matrix transform */
void dmatrix_to_gsl_matrix(double **dmatrix, gsl_matrix *X);
void dvector_to_gsl_vector(double *vector, gsl_vector *V);
void gsl_matrix_to_dmatrix(gsl_matrix *X, double **dmatrix);
void gsl_vector_to_dvector(gsl_vector *v, double *dvector);
void gsl_vector_to_ivector(gsl_vector *v, int *ivector);

void WeightedSum_B_multiMVN(HMM *hmm1, PARM_HMM *parm_hmm1);
void CalculateB_from_B1(HMM *phmm, PARM_HMM *parm_hmm);
void CalculateB_from_B1_Log(HMM *phmm, PARM_HMM *parm_hmm);
