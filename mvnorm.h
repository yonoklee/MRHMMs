#ifndef mvnorm_h
#define mvnorm_h

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

/* Calculate Determinant */
double get_det(double **A, int row_sq, gsl_matrix *A_inv, int *ERROR_IND);
double get_det_gsl(gsl_matrix *A, gsl_matrix *A_inv, int *ERROR_IND);

/* Vector calculation */
void   vector_sub(gsl_vector *x, double *mu, int vec_length, gsl_vector *vec_sub);
void gsl_matrix_vector_multiplication(gsl_matrix *A, gsl_vector *b, 
				      int col_A, gsl_vector *Ab);

/* Calculate the Normal density: input formats are different */
double mvdnorm_A(gsl_vector *x,  double *mu_vector, double **Sigma, int log_ind, int *ERROR_IND);
double mvdnorm_B(gsl_vector *x, double *mu_vector, double det, gsl_matrix *A_inv, int log_ind);


#endif 
