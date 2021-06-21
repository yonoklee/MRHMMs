/* Functions are explained in mvnorm.h */

#include <gsl/gsl_errno.h>
#include "mvnorm.h"
#include "hmm.h"
#include "HMM.Read.Matrix.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


double get_det_gsl(gsl_matrix *A, gsl_matrix *A_inv, int *ERROR_IND) {

  double det=0.0, temp;
  int row_sq = A->size1;
  gsl_permutation * p = gsl_permutation_alloc(A->size1);
  gsl_matrix *tmp_ptr;
  int i, j, signum ;
  FILE *fp;
  if(0) gsl_set_error_handler_off();
  
  tmp_ptr = gsl_matrix_alloc(A->size1, A->size1);
  for(i = 0; i < A->size1; i++){
    for(j = 0; j< A->size1; j++){
      temp = gsl_matrix_get(A, (size_t) i, (size_t)j );
      gsl_matrix_set(tmp_ptr, (size_t) i, (size_t) j, temp);
    }
  }
  if(0){
    printf("\n");
    gsl_matrix_print(tmp_ptr);
    printf("\n");
  }
  if( gsl_linalg_LU_decomp(tmp_ptr, p, &signum)){
    ERROR_IND[0]=1;
    printf("Error in gsl_linalg_LU_decomp: logmvNormal.c");
    return(0.0);
  }

  temp = gsl_linalg_LU_det(tmp_ptr, signum);
  det = exp(gsl_linalg_LU_lndet(tmp_ptr));
  if(0)printf("det %e \t exp(logdet) %e \n", temp, det);
 
  if( gsl_linalg_LU_invert(tmp_ptr, p, A_inv) | (temp < 0) ){
    if(temp < 0){
      printf("determinant is %e and %e\n", temp, det);
      fp = fopen("A", "w");
      gsl_matrix_fprintf(fp, A, "%.15lf");
      fclose(fp);
      printf("\n");
      gsl_matrix_print(tmp_ptr);
      printf("log determinant = %f\n", gsl_linalg_LU_lndet(tmp_ptr));
      
      ERROR_IND[0] =1;
      return( temp);
    } else {
      printf("error! Matrix is singular in get_det_gsl function. \n");
      ERROR_IND[0]=1;}
    
    gsl_permutation_free(p);
    gsl_matrix_free(tmp_ptr);
    
    return(temp);
  }

  gsl_permutation_free(p);
  gsl_matrix_free(tmp_ptr);
  /*
  if(det == 0){
    printf("Determinant of Covariance is 0.\n");
    ERROR_IND[0]==1;
    break;
  } else {
    return(det);
  }
  */
  return(det);
}


double get_det(double **A, int row_sq, gsl_matrix *inv_A, int *ERROR_IND){

  int i, j;
  double det;
  gsl_matrix *tempA;
  
  
  tempA = gsl_matrix_alloc(row_sq, row_sq);
  for( i = 0; i < row_sq; i++)
    {
      for( j = 0; j< row_sq ; j++)
	{
	  gsl_matrix_set(tempA, i, j, A[i+1][j+1]);

	}
    }
  if(0){
    dmatrix_print(A, row_sq, row_sq);
    printf("\n");
    gsl_matrix_print(tempA);
  }

  det = get_det_gsl(tempA, inv_A, ERROR_IND);
  
  gsl_matrix_free(tempA);
  return(det);
}


void vector_sub(gsl_vector *x, double *mu, int vec_length, gsl_vector *vec_sub){
  double temp;
  int i;

  for(i=0; i < vec_length; i++)
    {
      temp = gsl_vector_get(x, (size_t) i)- mu[i+1];
      gsl_vector_set(vec_sub, i, temp);
    }
}


void gsl_matrix_vector_multiplication(gsl_matrix *A, gsl_vector *b, int col_A, gsl_vector *Ab)
{
  int i, k;
  double sum=0;
  
  for( i = 0; i< col_A; i++){
       sum=0;
       for( k = 0; k < col_A ;k++){
	 sum += gsl_matrix_get(A, i, k) * gsl_vector_get(b, k);
       }
       gsl_vector_set(Ab, i, sum);
  }
}

double mvdnorm_A(gsl_vector *x,  double *mu_vector, double **Sigma, int log_ind, int *ERROR_IND) {

  int i, j;
  gsl_matrix *A_inv = gsl_matrix_alloc(x->size, x->size);
  gsl_vector *xsubmu = gsl_vector_alloc(x->size);
  gsl_vector *lastTwo = gsl_vector_alloc(x->size);
  double det, c0, c1, *c2, all;
  int print_ind = 0;
  int row_sq = x->size;
 
  c2 = malloc(sizeof(double));
  det = get_det(Sigma, row_sq, A_inv, ERROR_IND);
  c0 = (double) -row_sq/2 *log(2 * M_PI);
  c1 =  -.5 * log(fabs(det));
  if(print_ind){
    printf("det = %2.52lf \n", fabs(det));
    printf("c0 = %lf \n", c0);
    printf("c0 *fabs(det) = %lf \n",  c0 * fabs(det));
    printf("c1 = %lf \n", c1);
  } 

  if(print_ind){
    for(i =0;i < row_sq; i++){
      printf("xsubmu = %2.10lf \t x = %2.10lf \t mu = %2.10lf \n", gsl_vector_get(xsubmu, i), gsl_vector_get(x, i), mu_vector[i+1] );
    }
  }
  
  /* gsl_blas_dgemv(CblasNoTrans, (double) 1, A_inv, xsubmu, (double)1, lastTwo);   */

   vector_sub(x, mu_vector, row_sq, xsubmu);  
   gsl_matrix_vector_multiplication(A_inv, xsubmu, row_sq, lastTwo);
   

  if(print_ind){
     for(i =0;i < row_sq; i++){
       printf("lastTwo = %2.10lf \n", gsl_vector_get(lastTwo, i));
     }
   }
  
  gsl_blas_ddot(xsubmu, lastTwo, c2);
  if(print_ind){
    printf("c2 = %lf \n", c2[0]);
    printf("A_inv\n");
    for(j = 0 ; j< row_sq ;j++){
      for(i =0; i < row_sq; i++){
	printf(" %2.10lf  ", gsl_matrix_get(A_inv, i, j));
      }
      printf("\n");
    }
  }
  all = c0 + c1 -.5*c2[0];
  if(print_ind)  printf("prob = %lf\n", all);
  gsl_vector_free(xsubmu);
  gsl_vector_free(lastTwo);
  gsl_matrix_free(A_inv);
  free(c2);
  if(log_ind == 1){
    return(all);} 
  if(log_ind == 0){
    return(exp(all));
  }
} 

  
double mvdnorm_B(gsl_vector *x, double *mu_vector,  double det, gsl_matrix *A_inv, int log_ind) {

  gsl_vector *xsubmu = gsl_vector_alloc(x->size);
  gsl_vector *lastTwo = gsl_vector_alloc(x->size);
  double c0, c1, *c2, all;
  int print_ind = 0, row_sq = x->size; 

  c2 = malloc(sizeof(double));
  c0 = (double) -.5 * row_sq * log(2 * M_PI);
  c1 = (double) -.5 * log(fabs(det));
 
  if(print_ind){
   printf("c0 = %lf \n", c0);
   printf("c0 *fabs(det) = %lf \n",  c0 * fabs(det));
   printf("c1 = %lf \n", c1);
  }
  vector_sub(x, mu_vector, x->size, xsubmu);
  gsl_matrix_vector_multiplication(A_inv, xsubmu, x->size, lastTwo);
  gsl_blas_ddot(xsubmu, lastTwo, c2);
  if(print_ind) printf("c2 = %lf \n", c2[0]);
  
  all = c0 + c1 -.5*c2[0];
  if(print_ind) printf("prob = %lf\n", all);

  gsl_vector_free(xsubmu);
  gsl_vector_free(lastTwo);
  free(c2);
  if(log_ind == 1){
    return(all);} else {return(exp(all));}
  
}  
  
