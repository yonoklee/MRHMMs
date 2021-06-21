/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   forward.c
**      Purpose: Foward algorithm for computing the probabilty 
**		of observing a sequence given a HMM model parameter.
**      Organization: University of Maryland
**
**      $Id: forward.c,v 1.2 1998/02/19 12:42:31 kanungo Exp kanungo $

*/

/* 
**      Forward  algorithms  for homogeneous transition probability 
        and multiple independent data sets 

       Forward : Forward algorithm 
       ForwardWithScale : Forward algorithm with scale
       ForwardWithScaleDist : Forward algorithm with scale and
                              nonhomogeneous transition probability
       ForwardWithScaleDistMulti : Forward algorithm with scale and 
                                   (non)homogeneous transition probability
				   with multiple independent data sets.
*/

#define _GNU_SOURCE
#include <stdio.h>
#include "hmm.h"

void Forward(HMM *phmm, int T,  double **alpha, double *pprob)
{
        int     i, j;   /* state indices */
        int     t;      /* time index */
 
        double sum;     /* partial sum */
 
        /* 1. Initialization */
 
        for (i = 1; i <= phmm->N; i++)
                alpha[1][i] = phmm->pi[i]* phmm->B[1][i];
 
        /* 2. Induction */
 
        for (t = 1; t < T; t++) {
                for (j = 1; j <= phmm->N; j++) {
                        sum = 0.0;
                        for (i = 1; i <= phmm->N; i++)
                                sum += alpha[t][i]* (phmm->A[i][j]);
 
                        alpha[t+1][j] = sum*(phmm->B[t+1][j]);
                }
        }
 
        /* 3. Termination */
        *pprob = 0.0;
        for (i = 1; i <= phmm->N; i++)
                *pprob += alpha[T][i];
 
}

void ForwardWithScale(HMM *phmm, int T,  double **alpha, 
		      double *scale, double *pprob, int *ERROR_IND)
/*  pprob is the LOG probability */
{
	int	i, j; 	/* state indices */
	int	t;	/* time index */

	double sum, temp;	/* partial sum */
	FILE *fp;

	/* 1. Initialization */
	
	scale[1] = 0.0;	
	for (i = 1; i <= phmm->N; i++) {
		alpha[1][i] = phmm->pi[i]* (phmm->B[1][i]);
		scale[1] += alpha[1][i];
	}
	for (i = 1; i <= phmm->N; i++) 
		alpha[1][i] /= scale[1]; 
	
	/* 2. Induction */

	for (t = 1; t <= T - 1; t++) {
		scale[t+1] = 0.0;
		for (j = 1; j <= phmm->N; j++) {
			sum = 0.0;
 			temp = 0;
			for (i = 1; i <= phmm->N; i++) {
			  sum += alpha[t][i]* (phmm->A[i][j]); 
			  temp += phmm->B[t+1][i];
			}

			alpha[t+1][j] = sum*(phmm->B[t+1][j]);
			scale[t+1] += alpha[t+1][j];
		
			if(1) {
			  if(temp ==0){
			    printf(" Obervation probabilities are all 0 (Forward Algorithm) at t = %d.\n",   t+1); 
                            printf("Try with different initial values. \n");
			    ERROR_IND[0] = 1;
			    
			  }
			}
			
		}
	
		for (j = 1; j <= phmm->N; j++) 
		  {
		    alpha[t+1][j] /= scale[t+1]; 
		   
		  }
		if(scale[t+1]==0){
		  printf("Warnings : scale is equal to 0 at t = %d. This will generate nan loglikelihood.\n", t);
		}
	
		

	}

	/* 3. Termination */
	pprob[0] = 0.0;

	for (t = 1; t <= T; t++)
	  {  
	    pprob[0] += log(scale[t]);
	  }
	
	if(0) printf("loglikelihood in forward algorithm = %f\n", pprob[0]);
	if(isnan(pprob[0]) | (pprob[0] > 0)){
	  
	  if(isnan(pprob[0]))
	    {
	      printf("log likelihood is nan (Forward algorithm).\n");
	      printf(" alpha.tex, scale.tex, B.tex will be writen.\n") ;
	      fp= fopen("alpha.tex", "w");
	      dmatrix_fprint(fp, alpha, phmm->M, phmm->N);
	      fclose(fp);
	      fp= fopen("scale.tex", "w");
	      dvector_fprint(fp, scale, phmm->M);
	      fclose(fp);
	      fp=fopen("B.tex", "w");
	      PrintHMM(fp, phmm);
	      fclose(fp);

	      ERROR_IND[0]=1;
	    }
	  /*  if(pprob[0]>0)
	      printf("loglikelihood is positive (Forward algorithm).\n"); */
	    
	}
	  
}


void ForwardWithScaleDist(HMM *phmm, int T,  double **alpha, 
		      double *scale, double *pprob, int *ERROR_IND)
/*  pprob is the LOG probability */
{
	int	i, j; 	/* state indices */
	int	t;	/* time index */
	int     l, m;   /* A related */

	double sum, temp;	/* partial sum */
	FILE *fp;

	/* 1. Initialization */
	
	scale[1] = 0.0;	
	for (i = 1; i <= phmm->N; i++) {
		alpha[1][i] = phmm->pi[i]* (phmm->B[1][i]);
		scale[1] += alpha[1][i];
	}
	for (i = 1; i <= phmm->N; i++) 
		alpha[1][i] /= scale[1]; 
	
	/* 2. Induction */

	for (t = 1; t <= T - 1; t++) {
		scale[t+1] = 0.0;
	
		for (j = 1; j <= phmm->N; j++) {
			sum = 0.0;
			temp = 0;
			for (i = 1; i <= phmm->N; i++) {
			  sum += alpha[t][i]* (phmm->A_t[t][i][j]); 
			  temp += phmm->B[t+1][i];
			}

			alpha[t+1][j] = sum*(phmm->B[t+1][j]);
			scale[t+1] += alpha[t+1][j];
		
			if(1) {
			  if(temp ==0){
			    printf(" Obervation probabilities are all 0 (Forward Algorithm) at t = %d.\n",   t+1); 
                            printf("Try with different initial values. \n");
			    ERROR_IND[0] = 1;
			    
			  }
			}
			
		}
	
		for (j = 1; j <= phmm->N; j++) 
		  {
		    alpha[t+1][j] /= scale[t+1]; 
		   
		  }
		if(scale[t+1]==0){
		  printf("Warnings : scale is equal to 0 at t = %d. This will generate nan loglikelihood.\n", t);
		}
		/* printf("\n");*/
		

	}

	/* 3. Termination */
	pprob[0] = 0.0;

	for (t = 1; t <= T; t++)
	  {  
	    pprob[0] += log(scale[t]);
	  }
	
	if(0) printf("loglikelihood in forward algorithm = %f\n", pprob[0]);
	if(isnan(pprob[0]) | (pprob[0] > 0)){
	  
	  if(isnan(pprob[0]))
	    {
	      printf("log likelihood is nan (Forward algorithm).\n");
	      printf(" alpha.tex, scale.tex, B.tex will be writen.\n") ;
	      fp= fopen("alpha.tex", "w");
	      dmatrix_fprint(fp, alpha, phmm->M, phmm->N);
	      fclose(fp);
	      fp= fopen("scale.tex", "w");
	      dvector_fprint(fp, scale, phmm->M);
	      fclose(fp);
	      fp=fopen("B.tex", "w");
	      PrintHMM(fp, phmm);
	      fclose(fp);

	      ERROR_IND[0]=1;
	    }
	  if(pprob[0]>0)
	    printf("loglikelihood is positive (Forward algorithm).\n");
	    
	}
	  
}


void ForwardWithScaleDistMulti(HMM *phmm, int T,  double **alpha, 
		      double *scale, double *pprob, int *ERROR_IND)
/*  pprob is the LOG probability */
{
	int	i, j; 	/* state indices */
	int	t;	/* time index */
	int     l, m, chr, loc, loc_end;   /* A related */

	double sum, temp;	/* partial sum */
	FILE *fp;

	/* 1. Initialization */
	
	
	for(chr = 1; chr <= phmm->num_chr; chr++){
	  loc = phmm-> new_chr[chr];
	  loc_end = phmm->new_chr[chr + 1]-1;
	  scale[loc] = 0.0;	
	  phmm->B_scale_multi[chr] = 1;
	  for (i = 1; i <= phmm->N; i++) {
	    alpha[loc][i] = phmm->pi[i]* (phmm->B[loc][i]);
	    scale[loc] += alpha[loc][i];
	  }
	  for (i = 1; i <= phmm->N; i++) 
	    alpha[loc][i] /= scale[loc]; 
	
	  /* 2. Induction */

	  for (t = loc; t <= (loc_end - 1); t++) {
		scale[t+1] = 0.0;
		

		for (j = 1; j <= phmm->N; j++) {
			sum = 0.0;
			temp = 0;
			for (i = 1; i <= phmm->N; i++) {
			  if(phmm->D1 > 0){
			    sum += alpha[t][i]* (phmm->A_t[t][i][j]); 
			  }
			  if(phmm->D1 == 0){
			    sum += alpha[t][i]* (phmm->A[i][j]); 
			  }
		
			  temp += phmm->B[t+1][i];
			}

			alpha[t+1][j] = sum*(phmm->B[t+1][j]);
			scale[t+1] += alpha[t+1][j];
		
			if(1) {
			  if(temp ==0){
			    printf(" Obervation probabilities are all 0 (Forward Algorithm) at t = %d.\n",   t+1); 
                            printf("Try with different initial values. \n");
			    ERROR_IND[0] = 1;
			    
			  }
			}
			
		}
	
		
		phmm->B_scale_multi[chr] *= scale[t+1];
		for (j = 1; j <= phmm->N; j++) 
		  {
		    alpha[t+1][j] /= scale[t+1]; ;
		    if(alpha[t+1][j] < 0){
		      printf("Warnings : alpha[%d][%d] = %f. \n", t+1, j, alpha[t+1][j]);
		      ERROR_IND[0]=1;
		    }
		  }
		if(scale[t+1]==0){
		  printf("Warnings : scale is equal to 0 at t = %d. This will generate nan loglikelihood.\n", t);
		}
		/* printf("\n");*/
		

	}
	}
	/* 3. Termination */
	pprob[0] = 0.0;

	for (t = 1; t <= T; t++)
	  {  
	    pprob[0] += log(scale[t]);
	  }
	



	if(0) printf("loglikelihood in forward algorithm = %f\n", pprob[0]);
	if(isnan(pprob[0]) | (pprob[0] > 0)){
	  
	  if(isnan(pprob[0]))
	    {
	      printf("log likelihood is nan (Forward algorithm).\n");
	      printf(" alpha.tex, scale.tex, B.tex will be writen.\n") ;
	      fp= fopen("alpha.tex", "w");
	      dmatrix_fprint(fp, alpha, phmm->M, phmm->N);
	      fclose(fp);
	      fp= fopen("scale.tex", "w");
	      dvector_fprint(fp, scale, phmm->M);
	      fclose(fp);
	      fp=fopen("B.tex", "w");
	      PrintHMM(fp, phmm);
	      fclose(fp);

	      ERROR_IND[0]=1;
	    }
	  /* if(pprob[0]>0)
	     printf("loglikelihood is positive (Forward algorithm).\n");*/
	    
	}



	  
	  
}
