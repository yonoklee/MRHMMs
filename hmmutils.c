/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   hmmutils.c
**      Purpose: utilities for reading, writing HMM stuff. 
**      Organization: University of Maryland
**
**      $Id: hmmutils.c,v 1.4 1998/02/23 07:51:26 kanungo Exp kanungo $
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nrutil.h"
#include "hmm.h"


static char rcsid[] = "$Id: hmmutils.c,v 1.4 1998/02/23 07:51:26 kanungo Exp kanungo $";

void ReadHMM(FILE *fp, HMM *phmm)
{
	int i, j, k;

	fscanf(fp, "M= %d\n", &(phmm->M)); 

	fscanf(fp, "N= %d\n", &(phmm->N)); 

	fscanf(fp, "A:\n");
	phmm->A = (double **) dmatrix(1, phmm->N, 1, phmm->N);
	for (i = 1; i <= phmm->N; i++) { 
		for (j = 1; j <= phmm->N; j++) {
			fscanf(fp, "%lf", &(phmm->A[i][j])); 
		}
		fscanf(fp,"\n");
	}

	fscanf(fp, "B:\n");
	phmm->B = (double **) dmatrix(1, phmm->N, 1, phmm->M);
	for (j = 1; j <= phmm->N; j++) { 
		for (k = 1; k <= phmm->M; k++) {
			fscanf(fp, "%lf", &(phmm->B[j][k])); 
		}
		fscanf(fp,"\n");
	}

	fscanf(fp, "pi:\n");
	phmm->pi = (double *) dvector(1, phmm->N);
	for (i = 1; i <= phmm->N; i++) 
		fscanf(fp, "%lf", &(phmm->pi[i])); 

}

void FreeHMM(HMM *phmm)
{
	free_dmatrix(phmm->A, 1, phmm->N, 1, phmm->N);
	free_dmatrix(phmm->B, 1, phmm->N, 1, phmm->M);
	free_dvector(phmm->pi, 1, phmm->N);
}

/*
** InitHMM() This function initializes matrices A, B and vector pi with
**	random values. Not doing so can result in the BaumWelch behaving
**	quite weirdly.
*/ 

void InitHMM(HMM *phmm, int N, int M, int seed)
{
	int i, j, k;
	double sum;


	/* initialize random number generator */


	hmmsetseed(seed);	

       	phmm->M = M;
 
        phmm->N = N;
 
        phmm->A = (double **) dmatrix(1, phmm->N, 1, phmm->N);

        for (i = 1; i <= phmm->N; i++) {
		sum = 0.0;
                for (j = 1; j <= phmm->N; j++) {
                        phmm->A[i][j] = hmmgetrand(); 
			sum += phmm->A[i][j];
		}
                for (j = 1; j <= phmm->N; j++) 
			 phmm->A[i][j] /= sum;
	}
 
        phmm->B = (double **) dmatrix(1, phmm->N, 1, phmm->M);

        for (j = 1; j <= phmm->N; j++) {
		sum = 0.0;	
                for (k = 1; k <= phmm->M; k++) {
                        phmm->B[j][k] = hmmgetrand();
			sum += phmm->B[j][k];
		}
                for (k = 1; k <= phmm->M; k++) 
			phmm->B[j][k] /= sum;
	}
 
        phmm->pi = (double *) dvector(1, phmm->N);
	sum = 0.0;
        for (i = 1; i <= phmm->N; i++) {
                phmm->pi[i] = hmmgetrand(); 
		sum += phmm->pi[i];
	}
        for (i = 1; i <= phmm->N; i++) 
		phmm->pi[i] /= sum;
}

void CopyHMM(HMM *phmm1, HMM *phmm2)
{
        int i, j, k;
 
        phmm2->M = phmm1->M;

 
        phmm2->N = phmm1->N;
 
        phmm2->A = (double **) dmatrix(1, phmm2->N, 1, phmm2->N);
 
        for (i = 1; i <= phmm2->N; i++)
                for (j = 1; j <= phmm2->N; j++)
                        phmm2->A[i][j] = phmm1->A[i][j];
 
        phmm2->B = (double **) dmatrix(1, phmm2->N, 1, phmm2->M);
        for (j = 1; j <= phmm2->N; j++)
                for (k = 1; k <= phmm2->M; k++)
                        phmm2->B[j][k] = phmm1->B[j][k];
 
        phmm2->pi = (double *) dvector(1, phmm2->N);
        for (i = 1; i <= phmm2->N; i++)
                phmm2->pi[i] = phmm1->pi[i]; 
 
}

/* 
void CopytoHMM_Braw(HMM *phmm, PARA_HMM *para_hmm1){
  int i, j, k;
  double temp;

   for (j = 1; j <= phmm->N; j++)
     for (k = 1; k <= phmm->M; k++)
       phmm->Braw[j][k] = phmm->B[j][k];
 
   for (j = 1; j <= phmm->N; j++){
      for (k = 1; k <= phmm->M; k++){
	temp=0;
	for(i = 1; i <= para_hmm1->mix_comp; i++){
	  temp+= exp(phmm->B1[j][k][i])* para_hmm->c[j][i] ;
	}
	phmm->Braw[k][j] = log(temp);
      }
   }
}
*/

void PrintHMM(FILE *fp, HMM *phmm)
{
  /* B : M by N */
        int i, j, k;

	fprintf(fp, "> Sample size M: %d\n", phmm->M); 
	fprintf(fp, "> Number of states N: %d\n", phmm->N); 
	/* fprintf(fp, "> B_scale = %f\n", phmm->B_scale); */

	fprintf(fp, "> pi:\n");
        for (i = 1; i <= phmm->N; i++) {
		fprintf(fp, "%f ", phmm->pi[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "> A:\n");
        for (i = 1; i <= phmm->N; i++) {
                for (j = 1; j <= phmm->N; j++) {
                        fprintf(fp, "%f ", phmm->A[i][j] );
		}
		fprintf(fp, "\n");
	}
	if(0){
	fprintf(fp, "> B:\n");
        for (j = 1; j <= phmm->M; j++) {
                for (k = 1; k <= phmm->N; k++){
                        fprintf(fp, "%e \t ", phmm->B[j][k]);
		}
		fprintf(fp, "\n");
	}
	}
}

void StringPaste(char *output, char *A, char *B)
{
  strcpy(output, A);
  strcat(output, B);
}

void PrintPARM(FILE *fp, PARM_HMM *parm_hmm1, char *model_name)
{
  int i, j, k, d , col;
  double temp;
        if(!(strcmp(model_name, "singleMVN"))|!(strcmp(model_name, "singleMVN_regression")) |!(strcmp(model_name, "singleMVN_logistic")) |!(strcmp(model_name, "multiMVN")) |!(strcmp(model_name, "multiMVN_regression")) |!(strcmp(model_name, "multiMVN_regression_state")) ){

        fprintf(fp, "> Mean:\n");
        for (i = 1; i <= parm_hmm1->N *parm_hmm1->mix_comp; i++) {
	  fprintf(fp, "> State %d mix_comp %d\n", 
		  (i- ((i-1)%parm_hmm1->mix_comp+1))/(parm_hmm1->mix_comp)+1,
		  (i-1)%(parm_hmm1->mix_comp)+1);
	  for (j = 1; j <= parm_hmm1->p; j++) {
	    fprintf(fp, "%f ", parm_hmm1->mu[i][j] );
	  }
	  fprintf(fp, "\n");
	}
 
	fprintf(fp, "> Sigma\n");
	for (i = 1; i <= (parm_hmm1-> N)*(parm_hmm1->mix_comp); i++) 
	  {
	    fprintf(fp, "> State %d mix_comp %d\n", 
		    (i- ((i-1)%parm_hmm1->mix_comp+1))/(parm_hmm1->mix_comp)+1,
		    (i-1)%(parm_hmm1->mix_comp)+1);
	    for (j = 1; j <= parm_hmm1->p; j++) {
	      for (k = 1; k <= parm_hmm1->p; k++){
		fprintf(fp, "%f \t", parm_hmm1->Sigma[i][j][k]);
	      }
	      fprintf(fp, "\n");
	    }
	  }
	    
  }
	
	if( parm_hmm1-> mix_comp > 1){
	  fprintf(fp, "> Proportion Probabilities\n");  
	  for( i = 1; i <= parm_hmm1-> N; i++){
	    for(j = 1; j <= parm_hmm1 -> mix_comp; j++){
	      fprintf(fp, "%f \t", parm_hmm1->c[i][j]);
	    }
	    fprintf(fp, "\n");
	  }
	}
	
	if(!(strcmp(model_name, "single_regression"))| !(strcmp(model_name, "singleMVN_regression")) |!(strcmp(model_name, "single_logistic")) | !(strcmp(model_name, "singleMVN_logistic")) |!(strcmp(model_name, "multiMVN_regression"))){
	  fprintf(fp, "> Regression Coefficients:\n");
	  for ( i = 1; i<= parm_hmm1->N ; i++){
	    for(j = 1; j<=parm_hmm1->mix_comp ; j++){
	      fprintf(fp, "> State %d mix_comp %d\n", i, j);
	      for(d = 1; d <= parm_hmm1->d; d++){
		col = parm_hmm1->mix_comp * parm_hmm1->d * (i-1)  + parm_hmm1->d *(j-1) + d;
		for( k = 1; k <= (parm_hmm1 -> p) + 1 ; k++){
		 
		    fprintf(fp, "%f \t",  parm_hmm1->reg_coef[col][k]);
		}
		fprintf(fp, "\n");
	      }
	    }
	  }
	}
	  
	if(!(strcmp(model_name, "multiMVN_regression_state"))){
	  fprintf(fp, "> Regression Coefficients:\n");
	  for ( i = 1; i<= parm_hmm1->N ; i++){
	      fprintf(fp, "> State %d \n", i);
	      for(d = 1; d <= parm_hmm1->d; d++){
		col =  parm_hmm1->d * (i-1)  + d;
		for( k = 1; k <= (parm_hmm1 -> p) + 1 ; k++){
		  fprintf(fp, " %f\t",  parm_hmm1->reg_coef[col][k]);
		}
		fprintf(fp, "\n");
	      }
	  }
	}
	
	if(!(strcmp(model_name, "single_regression"))| !(strcmp(model_name, "singleMVN_regression")) | !(strcmp(model_name, "multiMVN_regression"))){
	  fprintf(fp, "> Regression Error Covariance\n");
	  for ( i = 1; i<= parm_hmm1->mix_comp * parm_hmm1->N ; i++){
	    fprintf(fp, "> State %d mix_comp %d\n", 
		    (i- ((i-1)%parm_hmm1->mix_comp+1))/(parm_hmm1->mix_comp)+1,
		    (i-1)%(parm_hmm1->mix_comp)+1);
	    for( j = 1; j<= parm_hmm1 -> d  ; j++){
	      for( k = 1; k <=parm_hmm1 -> d ; k++){
		fprintf(fp, "%f \t", parm_hmm1->reg_cov[i][j][k]);
	      }
	      fprintf(fp, "\n");
	    }
	    fprintf(fp, " \n");
	  }

	}

	if(!(strcmp(model_name, "multiMVN_regression_state"))){
	  fprintf(fp, "> Regression Error Covariance\n");
	  for ( i = 1; i<= parm_hmm1->N ; i++){
	    fprintf(fp, "> State %d \n", i);
	    for( j = 1; j<= parm_hmm1 -> d  ; j++){
	      for( k = 1; k <=parm_hmm1 -> d ; k++){
		fprintf(fp, "%f \t", parm_hmm1->reg_cov[i][j][k]);
	      }
	      fprintf(fp, "\n");
	    }
	    fprintf(fp, " \n");
	  }
	}

}


void PrintViterbi( FILE *fp, int *q, int T)
{
  int i;
  
  for ( i = 1; i<= T; i++){
    fprintf(fp, "%d \n", q[i]);
  }
}

void PrintLoglike( FILE *fp, double loglike, double **gamma, double pprob, int T, int M)
{
  int i, j;
  
  fprintf(fp, "> loglikelihood \n");
  fprintf(fp, "%f \n", loglike);
  
  /* fprintf(fp, "> Viterbi loglikelihood \n");
     fprintf(fp, "%f \n", pprob );
  */
  fprintf(fp, "> gamma \n");
  for ( i = 1; i <= T; i++){
    for (j = 1; j <= M; j++){
      fprintf(fp, "%e \t", gamma[i][j]);
    }
    fprintf(fp, "\n");
  }
  
}

void PrintHMM_All_Results(char *output_filename, HMM *phmm, PARM_HMM *parm_hmm1, int *q , double loglike, double pprob, double **gamma)
{
  FILE *file_hmm, *file_parm, *file_viterbi,  *file_loglike;
  char *output_hmm, *output_parm, *output_viterbi, *output_loglike;
  
  output_hmm = malloc(100*sizeof(char));
  output_parm = malloc(100*sizeof(char));
  output_viterbi = malloc(100*sizeof(char));
  output_loglike = malloc(100*sizeof(char));


  StringPaste(output_hmm, output_filename, "_hmm");
  StringPaste(output_parm, output_filename, "_parm");
  StringPaste(output_viterbi, output_filename, "_viterbi");
  StringPaste(output_loglike, output_filename, "_loglike");

  file_hmm = fopen(output_hmm, "w");
  file_parm = fopen(output_parm, "w");
  file_viterbi = fopen(output_viterbi, "w");
  file_loglike = fopen(output_loglike, "w");
  
  if( file_hmm ==  NULL){
     printf("Error! Cannot write hmm output file. \n") ;
     exit(1);
   }
  
  if( file_parm ==  NULL){
     printf("Error! Cannot write hmm_parm output file. \n") ;
     exit(1);
   }
if( file_viterbi ==  NULL){
     printf("Error! Cannot write Viterbi output file. \n") ;
     exit(1);
   }

if( file_loglike ==  NULL){
     printf("Error! Cannot write loglikelihood output file. \n") ;
     exit(1);
   }



  PrintHMM(file_hmm, phmm);

  PrintPARM(file_parm, parm_hmm1, phmm->modelname);

  PrintViterbi(file_viterbi, q, phmm->M);
  
  PrintLoglike(file_loglike, loglike, gamma, pprob,  phmm->M, phmm->N);
  
  printf(" \n Note: The results are saved in \n %s, \n %s, \n %s, \n and, %s.\n", 
	 output_hmm, output_parm, output_viterbi, output_loglike);

  fclose(file_hmm);
  fclose(file_parm);
  fclose(file_viterbi);
  fclose(file_loglike);
  free(output_hmm);
  free(output_parm);
  free(output_viterbi);
  free(output_loglike);
}

void PrintHMM_All_Results_rep(char *output_filename, HMM *phmm, PARM_HMM *parm_hmm1, int *q , double loglike, double **gamma, double pprob, int rep)
{
  FILE *file_hmm, *file_parm, *file_viterbi,  *file_loglike;
  char *output_hmm, *output_parm, *output_viterbi, *output_loglike;
  
  output_hmm = malloc(100*sizeof(char));
  output_parm = malloc(100*sizeof(char));
  output_viterbi = malloc(100*sizeof(char));
  output_loglike = malloc(100*sizeof(char));


 
  StringPaste(output_hmm, output_filename, "_hmm_rep");
  StringPaste(output_parm, output_filename, "_parm_rep");
  StringPaste(output_viterbi, output_filename, "_viterbi_rep");
  StringPaste(output_loglike, output_filename, "_loglike_rep");

  
  sprintf(output_hmm,"%s%d", output_hmm, rep);
  sprintf(output_parm, "%s%d", output_parm, rep);
  sprintf(output_viterbi, "%s%d", output_viterbi, rep);
  sprintf(output_loglike, "%s%d", output_loglike, rep);


  file_hmm = fopen(output_hmm, "w");
  file_parm = fopen(output_parm, "w");
  file_viterbi = fopen(output_viterbi, "w");
  file_loglike = fopen(output_loglike, "w");
  
  PrintHMM(file_hmm, phmm);

  PrintPARM(file_parm, parm_hmm1, phmm->modelname);

  PrintViterbi(file_viterbi, q, phmm->M);
  
  if(0) PrintLoglike(file_loglike, loglike, gamma, pprob, phmm->M, phmm->N);
  
  printf(" \n Note: The results are saved in %s, \n %s,\n %s,\n and %s.\n", 
	 output_hmm, output_parm, output_viterbi, output_loglike);

  fclose(file_hmm);
  fclose(file_parm);
  fclose(file_viterbi);
  fclose(file_loglike);
  free(output_hmm);
  free(output_parm);
  free(output_viterbi);
  free(output_loglike);
}




void dmatrix_print(double **X, int row, int col)
{
  int i, j;
  
  for( i= 1; i<= row  ; i++){
    for( j = 1; j <= col ; j++){
      printf("%f   ", X[i][j]);
    }
    printf("\n");
  }
}

void dmatrix_fprint(FILE *fp, double **X, int row, int col)
{
  int i, j;
  double temp;
  for( i= 1; i<= row  ; i++){
    for( j = 1; j <= col ; j++){
      temp = X[i][j];
      fprintf(fp, "%e  \t ", temp);
    }
    fprintf(fp, "\n");
  }
}


void dvector_print(double *v, int vec_length)
{
 int i;
  for( i = 1; i <= vec_length ; i++)
    {
      printf("%f \n", v[i]);
    }
}

void dvector_fprint(FILE *fp, double *v, int vec_length)
{
   int i;
  double temp;
  for( i= 1; i<= vec_length; i++){
    temp = v[i];
    fprintf(fp, "%e  \n ", temp);
  }
}

