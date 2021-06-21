/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   viterbi.c
**      Purpose: Viterbi algorithm for computing the maximum likelihood
**		state sequence and probablity of observing a sequence
**		given the model. 
**      Organization: University of Maryland
**
**      $Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $
*/

#include <math.h>
#include "hmm.h"
#include "nrutil.h"
static char rcsid[] = "$Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $";

#define VITHUGE  100000000000.0

void Viterbi(HMM *phmm, int T,  double **delta, int **psi, 
	int *q, double *pprob)
{
	int 	i, j;	/* state indices */
	int  	t;	/* time index */	

	int	maxvalind;
	double	maxval, val;

	/* 1. Initialization  */
	
	for (i = 1; i <= phmm->N; i++) {
		delta[1][i] = phmm->pi[i] * (phmm->B[1][i]);
		psi[1][i] = 0;
	}	

	/* 2. Recursion */
	
	for (t = 2; t <= T; t++) {
		for (j = 1; j <= phmm->N; j++) {
			maxval = 0.0;
			maxvalind = 1;	
			for (i = 1; i <= phmm->N; i++) {
				val = delta[t-1][i]*(phmm->A[i][j]);
				if (val > maxval) {
					maxval = val;	
					maxvalind = i;	
				}
			}
			
			delta[t][j] = maxval*(phmm->B[t][j]);
			psi[t][j] = maxvalind; 

		}
	}

	/* 3. Termination */

	*pprob = 0.0;
	q[T] = 1;
	for (i = 1; i <= phmm->N; i++) {
                if (delta[T][i] > *pprob) {
			*pprob = delta[T][i];	
			q[T] = i;
		}
	}

	/* 4. Path (state sequence) backtracking */

	for (t = T - 1; t >= 1; t--)
		q[t] = psi[t+1][q[t+1]];

}

void ViterbiLog(HMM *phmm, int T, double **delta, int **psi,
        int *q, double *pprob)
{
        int     i, j;   /* state indices */
        int     t;      /* time index */
 
        int     maxvalind;
        double  maxval, val;
	double  **biot;
	double  **A_log, *pi_log;
	
	
	/* 0. Preprocessing */
	pi_log = dvector(1, phmm->N);
	for (i = 1; i <= phmm->N; i++) 
		pi_log[i] = log(phmm->pi[i]);
	
	A_log = dmatrix(1, phmm-> N, 1, phmm-> N);
	for (i = 1; i <= phmm->N; i++) 
	  for (j = 1; j <= phmm->N; j++) {
	    A_log[i][j] = log(phmm->A[i][j]);
	  }

	biot = dmatrix(1, phmm->N, 1, T);
	for (i = 1; i <= phmm->N; i++) 
		for (t = 1; t <= T; t++) {
			biot[i][t] = log(phmm->B[t][i]);
		}
 
        /* 1. Initialization  */
 
        for (i = 1; i <= phmm->N; i++) {
                delta[1][i] = pi_log[i] + biot[i][1];
                psi[1][i] = 0;
        }
 
        /* 2. Recursion */
 
        for (t = 2; t <= T; t++) {
                for (j = 1; j <= phmm->N; j++) {
                        maxval = -VITHUGE;
                        maxvalind = 1;
                        for (i = 1; i <= phmm->N; i++) {
			  val = delta[t-1][i] + (A_log[i][j]);
                                if (val > maxval) {
                                        maxval = val;
                                        maxvalind = i;
                                }
                        }
 
                        delta[t][j] = maxval + biot[j][t]; 
                        psi[t][j] = maxvalind;
 
                }
        }
 
        /* 3. Termination */
 
        *pprob = -VITHUGE;
        q[T] = 1;
        for (i = 1; i <= phmm->N; i++) {
                if (delta[T][i] > *pprob) {
                        *pprob = delta[T][i];
                        q[T] = i;
                }
        }
 
 
	/* 4. Path (state sequence) backtracking */

	for (t = T - 1; t >= 1; t--)
		q[t] = psi[t+1][q[t+1]];

	/* temp_file=fopen("in_viterbi_function", "w"); 
	   printHMM(temp_file, phmm);*/

	free_dmatrix(A_log, 1,  phmm-> N, 1, phmm-> N);
	free_dvector(pi_log, 1, phmm->N);
	free_dmatrix(biot, 1, phmm->N, 1, T);
}
 


void ViterbiLogDist(HMM *phmm, int T, double **delta, int **psi,
        int *q, double *pprob)
{
        int     i, j;   /* state indices */
        int     t;      /* time index */
	int     l, m;
        int     maxvalind;
        double  maxval, val;
	double  **biot;
	double  **A_log, *pi_log;
	
	
	/* 0. Preprocessing */
	pi_log = dvector(1, phmm->N);
	for (i = 1; i <= phmm->N; i++) 
		pi_log[i] = log(phmm->pi[i]);
	
	A_log = dmatrix(1, phmm-> N, 1, phmm-> N);
	biot = dmatrix(1, phmm->N, 1, T);
	for (i = 1; i <= phmm->N; i++) 
		for (t = 1; t <= T; t++) {
			biot[i][t] = log(phmm->B[t][i]);
		}
 
        /* 1. Initialization  */
 
        for (i = 1; i <= phmm->N; i++) {
                delta[1][i] = pi_log[i] + biot[i][1];
                psi[1][i] = 0;
        }
 
        /* 2. Recursion */
 
        for (t = 2; t <= T; t++) {

	  for(l = 1; l <= phmm->N ;l++){
	    for(m = 1; m <= phmm->N ; m++){
	      A_log[l][m] = log(phmm -> A_t[t-1][l][m]);
	    }
	  }

                for (j = 1; j <= phmm->N; j++) {
                        maxval = -VITHUGE;
                        maxvalind = 1;
                        for (i = 1; i <= phmm->N; i++) {
			  val = delta[t-1][i] + (A_log[i][j]);
                                if (val > maxval) {
                                        maxval = val;
                                        maxvalind = i;
                                }
                        }
 
                        delta[t][j] = maxval + biot[j][t]; 
                        psi[t][j] = maxvalind;
 
                }
        }
 
        /* 3. Termination */
 
        *pprob = -VITHUGE;
        q[T] = 1;
        for (i = 1; i <= phmm->N; i++) {
                if (delta[T][i] > *pprob) {
                        *pprob = delta[T][i];
                        q[T] = i;
                }
        }
 
 
	/* 4. Path (state sequence) backtracking */

	for (t = T - 1; t >= 1; t--)
		q[t] = psi[t+1][q[t+1]];



	free_dmatrix(A_log, 1,  phmm-> N, 1, phmm-> N);
	free_dvector(pi_log, 1, phmm->N);
	free_dmatrix(biot, 1, phmm->N, 1, T);
}
 

void ViterbiLogDistMulti(HMM *phmm, int T, double **delta, int **psi,
        int *q, double *pprob)
{
        int     i, j;   /* state indices */
        int     t;      /* time index */
	int     l, m, chr, loc_start, loc_end;
        int     maxvalind;
        double  maxval, val;
	double  **biot;
	double  **A_log, *pi_log;
	
	
	/* 0. Preprocessing */
	pi_log = dvector(1, phmm->N);
	
	for (i = 1; i <= phmm->N; i++) 
	  pi_log[i] = log(phmm->pi[i]);
	

	A_log = dmatrix(1, phmm-> N, 1, phmm-> N);
	biot = dmatrix(1, phmm->N, 1, T);
	for (i = 1; i <= phmm->N; i++) 
		for (t = 1; t <= T; t++) {
			biot[i][t] = log(phmm->B[t][i]);
		}
 
        /* 1. Initialization  */
 
	for( chr = 1 ; chr <= phmm->num_chr; chr++){
	  loc_start = phmm->new_chr[chr];
	  loc_end = phmm->new_chr[chr+1]-1;
	  
	  for (i = 1; i <= phmm->N; i++) {
	    delta[loc_start][i] = pi_log[i] + biot[i][loc_start];
	    psi[loc_start][i] = 0;
	  }
 	 
	 
	  /* 2. Recursion */
 
	  for (t = loc_start +1 ; t <= loc_end; t++) {
	    for(l = 1; l <= phmm->N ;l++){
	      for(m = 1; m <= phmm->N ; m++){
		if(phmm->D1 > 0){
		  A_log[l][m] = log(phmm -> A_t[t-1][l][m]);
		}
		if(phmm->D1 ==0){
		  A_log[l][m] = log(phmm->A[l][m]);
		}
	      }
	    }

	    for (j = 1; j <= phmm->N; j++) {
	      maxval = -VITHUGE;
	      maxvalind = 1;
	      for (i = 1; i <= phmm->N; i++) {
		val = delta[t-1][i] + (A_log[i][j]);
		if (val > maxval) {
		  maxval = val;
		  maxvalind = i;
		}
	      }
	      
	      delta[t][j] = maxval + biot[j][t]; 
	      psi[t][j] = maxvalind;
	      
	    }
	  }
 
        /* 3. Termination */
 
	  *pprob = -VITHUGE;
	  q[loc_end] = 1;
	  for (i = 1; i <= phmm->N; i++) {
	    /* printf("delta[%d][%d]= %lf\n", loc_end, i, delta[loc_end][i]);*/
	    if (delta[loc_end][i] > *pprob) {
	      *pprob = delta[loc_end][i];
	      q[loc_end] = i;
	    }
	  }
 
 
	/* 4. Path (state sequence) backtracking */

	  for (t = loc_end - 1; t >= loc_start; t--)
	    q[t] = psi[t+1][q[t+1]];
	  /*
	  printf("loc_start = %d and loc_end = %d.\n", loc_start, loc_end);
	  */

	}
	
	
	free_dmatrix(A_log, 1,  phmm-> N, 1, phmm-> N);
	free_dvector(pi_log, 1, phmm->N);
	free_dmatrix(biot, 1, phmm->N, 1, T);
}
 
