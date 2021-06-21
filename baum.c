/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   baumwelch.c
**      Purpose: Baum-Welch algorithm for estimating the parameters
**              of a HMM model, given an observation sequence. 
**      Organization: University of Maryland
**
**	Update: 
**	Author: Tapas Kanungo
**	Date:	19 April 1999
**	Purpose: Changed the convergence criterion from ratio
**		to absolute value. 
**
**      $Id: baumwelch.c,v 1.6 1999/04/24 15:58:43 kanungo Exp kanungo $
*/


/* 
Functions :

ComputeGamma : Compute probability of state at each time
ComputeXi : Compute probability of transition at each time
ComputeXiDist : Compute probability of transition at each time with nonhomogeneous probability
ComputeXiDistMulti : Compute probabilty of transition at each time with (non)homogeneous probability for multiple independent data sets

AllocXi : Allocate  memory for array
FreeXi: Free array memory

*/
#include <stdio.h> 
#include <math.h>
#include "nrutil.h"
#include "hmm.h"



#define DELTA 0.001 


static char rcsid[] = "$Id: baumwelch.c,v 1.6 1999/04/24 15:58:43 kanungo Exp kanungo $";


void ComputeGamma(HMM *phmm, int T, double **alpha, double **beta, 
		  double **gamma, gsl_matrix *weight_m)
{
  /* gamma and weight_m are the same except in their format */

	int 	i, j;
	int	t;
	double	denominator;

	for (t = 1; t <= T; t++) {
		denominator = 0.0;
		for (j = 1; j <= phmm->N; j++) {
			gamma[t][j] = alpha[t][j]*beta[t][j];
			denominator += gamma[t][j];
			if(0)
			  {
			    if(denominator == 0){
			      printf("nan occurs in ComputeGamma.\n");
			      printf(" t = %d\n", t);

			      for( i = 1; i<=phmm->N; i++){
				printf("%f \t", gamma[t][i]);
			      }
			      exit(1);
			    }
			  }
		
		}
			for (i = 1; i <= phmm->N; i++) 
			  {
			    gamma[t][i] = gamma[t][i]/denominator;
			    gsl_matrix_set(weight_m, t-1, i-1, gamma[t][i]);
			  }
			
	}
}


void ComputeXi(HMM *phmm, int T, double **alpha, double **beta, 
	double ***xi)
{
	int i, j;
	int t;
	double sum;

	for (t = 1; t <= T - 1; t++) {
		sum = 0.0;	
		for (i = 1; i <= phmm->N; i++) 
			for (j = 1; j <= phmm->N; j++) {
				xi[t][i][j] = alpha[t][i]*beta[t+1][j]
					*(phmm->A[i][j])
					*(phmm->B[t+1][j]);
				sum += xi[t][i][j];
			}

		for (i = 1; i <= phmm->N; i++) 
			for (j = 1; j <= phmm->N; j++) 
				xi[t][i][j]  /= sum;
	}
}


void ComputeXiDist(HMM *phmm, int T, double **alpha, double **beta, 
	double ***xi)
{
	int i, j;
	int l, m;
	int t;
	double sum;

	for (t = 1; t <= T - 1; t++) {

		/* Begin : hielim  */
	  /* for(l = 1; l <= phmm->N ;l++){
		 for(m = 1; m <= phmm->N ; m++){
		      phmm-> A[l][m] = phmm -> A_t[t][l][m];
		    }
		}
	  */	
		phmm-> A = phmm -> A_t[t];
		/* End : hielim   */

		sum = 0.0;	
		for (i = 1; i <= phmm->N; i++) 
			for (j = 1; j <= phmm->N; j++) {
				xi[t][i][j] = alpha[t][i]*beta[t+1][j]
					*(phmm->A[i][j])
					*(phmm->B[t+1][j]);
				sum += xi[t][i][j];
			}

		for (i = 1; i <= phmm->N; i++) 
			for (j = 1; j <= phmm->N; j++) 
				xi[t][i][j]  /= sum;
	}
}


void ComputeXiDistMulti(HMM *phmm, int T, double **alpha, double **beta, 
	double ***xi)
{
	int i, j;
	int l, m, chr, loc_start, loc_end;
	int t;
	double sum;
	
	for(chr = 1; chr <= phmm->num_chr; chr++){
	  loc_start = phmm->new_chr[chr];
	  loc_end = phmm->new_chr[chr + 1] -1 ;

	  for (t = loc_start; t <= loc_end - 1; t++) {
	    if(phmm->D1 > 0){
	      phmm-> A = phmm -> A_t[t];
	    }
  
	    
	    
	    sum = 0.0;	
	    for (i = 1; i <= phmm->N; i++) 
	      for (j = 1; j <= phmm->N; j++) {
		xi[t][i][j] = alpha[t][i]*beta[t+1][j]
		  *(phmm->A[i][j])
		  *(phmm->B[t+1][j]);
		sum += xi[t][i][j];
			}

		for (i = 1; i <= phmm->N; i++) 
			for (j = 1; j <= phmm->N; j++) 
				xi[t][i][j]  /= sum;
	}
	}
}


double *** AllocXi(int T, int N)
{
	int t;
	double ***xi;

	xi = (double ***) malloc(T*sizeof(double **));

	xi --;

	for (t = 1; t <= T; t++)
		xi[t] = dmatrix(1, N, 1, N);
	return xi;
}


void FreeXi(double *** xi, int T, int N)
{
	int t;



	for (t = 1; t <= T; t++)
		free_dmatrix(xi[t], 1, N, 1, N);

	xi ++;
	free(xi);

}

