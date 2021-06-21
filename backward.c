/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   backward.c
**      Purpose: Backward algorithm for computing the probabilty
**              of observing a sequence given a HMM model parameter.
**      Organization: University of Maryland
**
**      $Id: backward.c,v 1.3 1998/02/23 07:56:05 kanungo Exp kanungo $
*/

/* 
**      Backward algorithms  for homogeneous transition probability 
        and multiple independent data sets 

       Backward : Backward algorithm 
       BackwardWithScale : Backward algorithm with scale
       BackwardWithScaleDist : Backward algorithm with scale and
                              nonhomogeneous transition probability
       BackwardWithScaleDistMulti : Backward algorithm with scale and 
                                   (non)homogeneous transition probability
				   with multiple independent data sets.
*/

#include <stdio.h>
#include "hmm.h"
static char rcsid[] = "$Id: backward.c,v 1.3 1998/02/23 07:56:05 kanungo Exp kanungo $";

void Backward(HMM *phmm, int T,  double **beta, double *pprob)
{
        int     i, j;   /* state indices */
        int     t;      /* time index */
        double sum;
 
 
        /* 1. Initialization */
 
        for (i = 1; i <= phmm->N; i++)
                beta[T][i] = 1.0;
 
        /* 2. Induction */
 
        for (t = T - 1; t >= 1; t--) {
                for (i = 1; i <= phmm->N; i++) {
                        sum = 0.0;
                        for (j = 1; j <= phmm->N; j++)
                                sum += phmm->A[i][j] *
                                        (phmm->B[t+1][j])*beta[t+1][j];
                        beta[t][i] = sum;
 
                }
        }
 
        /* 3. Termination */
        *pprob = 0.0;
        for (i = 1; i <= phmm->N; i++)
                *pprob += beta[1][i];
 
}

void BackwardWithScale(HMM *phmm, int T,  double **beta, 
	double *scale, double *pprob)
{
  /* Inputs: A, B, and scale */
  /* Output: beta */
        int     i, j;   /* state indices */
        int     t;      /* time index */
	double sum;
 
 
        /* 1. Initialization */
 
        for (i = 1; i <= phmm->N; i++)
                beta[T][i] = 1.0/scale[T]; 
 
        /* 2. Induction */
 
        for (t = T - 1; t >= 1; t--) {
                for (i = 1; i <= phmm->N; i++) {
			sum = 0.0;
                        for (j = 1; j <= phmm->N; j++)
                        	sum += phmm->A[i][j] * 
				  (phmm->B[t+1][j])*beta[t+1][j];
                        beta[t][i] = sum/scale[t];
 
                }
        }
 
}


void BackwardWithScaleDist(HMM *phmm, int T,  double **beta, 
	double *scale, double *pprob)
{
  /* Inputs: A, B, and scale */
  /* Output: beta */
        int     i, j;   /* state indices */
        int     t;      /* time index */
	int     l, m;
	double sum;
 
 
        /* 1. Initialization */
 
        for (i = 1; i <= phmm->N; i++)
                beta[T][i] = 1.0/scale[T]; 
 
        /* 2. Induction */
 
        for (t = T - 1; t >= 1; t--) {
	       for(l = 1; l <= phmm->N ;l++){
	            for(m = 1; m <= phmm->N ; m++){
		      phmm-> A[l][m] = phmm -> A_t[t][l][m];
		    }
		}
		
	

                for (i = 1; i <= phmm->N; i++) {
			sum = 0.0;
                        for (j = 1; j <= phmm->N; j++)
                        	sum += phmm->A[i][j] * 
				  (phmm->B[t+1][j])*beta[t+1][j];
                        beta[t][i] = sum/scale[t];
 
                }
        }
 
}


void BackwardWithScaleDistMulti(HMM *phmm, int T,  double **beta, 
	double *scale, double *pprob)
{
  /* Inputs: A, B, and scale */
  /* Output: beta */
        int     i, j;   /* state indices */
        int     t;      /* time index */
	int     l, m, chr, loc, loc_start;
	double sum;
 
 
        /* 1. Initialization */
 	for( chr = phmm->num_chr; chr >= 1; chr--){
	  loc = phmm-> new_chr[chr+1] - 1;
	  loc_start = phmm->new_chr[chr];

	  for (i = 1; i <= phmm->N; i++)
	    beta[loc][i] = 1.0/scale[loc]; 
	  
	  /* 2. Induction */
	  
	  for (t = loc - 1; t >= loc_start; t--) {
	    if(phmm->D1 > 0){
	      for(l = 1; l <= phmm->N ;l++){
		for(m = 1; m <= phmm->N ; m++){
		  phmm-> A[l][m] = phmm -> A_t[t][l][m];
		}
	      }
	    }
	    for (i = 1; i <= phmm->N; i++) {
	      sum = 0.0;
	      for (j = 1; j <= phmm->N; j++)
		sum += phmm->A[i][j] * 
		  (phmm->B[t+1][j])*beta[t+1][j];
	      beta[t][i] = sum/scale[t];
	      
	    }
	  }
	}
}
