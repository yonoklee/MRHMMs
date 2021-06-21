/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   hmm.h
**      Purpose: datastructures used for HMM. 
**      Organization: University of Maryland
**
**	Update:
**	Author: Tapas Kanungo
**	Purpose: include <math.h>. Not including this was
**		creating a problem with forward.c
**      $Id: hmm.h,v 1.9 1999/05/02 18:38:11 kanungo Exp kanungo $
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include "_struct.h"

void ReadHMM(FILE *fp, HMM *phmm);
void PrintHMM(FILE *fp, HMM *phmm);
void InitHMM(HMM *phmm, int N, int M, int seed);
void CopyHMM(HMM *phmm1, HMM *phmm2);
void FreeHMM(HMM *phmm);
void PrintPARM(FILE *fp, PARM_HMM *hmm_parm1, char *model_name);
void PrintViterbi(FILE *fp, int *q, int T);
void PrintLoglike(FILE *fp, double loglike, double **gamma, double pprob, int T, int M);
void StringPaste(char *output, char *A, char *B);
void PrintHMM_All_Results(char *output_filename, HMM *phmm, PARM_HMM *parm_hmm1, int *q, double loglike, double pprob,  double **gamma);
void PrintHMM_All_Results_rep(char *output_filename, HMM *phmm, PARM_HMM *parm_hmm1, int *q, double loglike, double **gamma, double pprob, int rep);

void dmatrix_print(double **X, int row, int col);
void dmatrix_fprint(FILE *fp, double **X, int row, int col);
void dvector_print(double *v, int vec_length);
void dvector_fprint(FILE *fp, double *v, int vec_length);




void ReadSequence(FILE *fp, int *pT, int **pO);
void PrintSequence(FILE *fp, int T, int *O);
void GenSequenceArray(HMM *phmm, int seed, int T, int *O, int *q);
int GenInitalState(HMM *phmm);
int GenNextState(HMM *phmm, int q_t);
int GenSymbol(HMM *phmm, int q_t);

/* Modified : int *O removed, dim(B) is changed to T * M */ 
void Forward(HMM *phmm, int T, double **alpha, double *pprob);
void ForwardWithScale(HMM *phmm, int T,  double **alpha,
		      double *scale, double *pprob, int *ERROR_IND);
void ForwardWithScaleDist(HMM *phmm, int T,  double **alpha,
		      double *scale, double *pprob, int *ERROR_IND);
void ForwardWithScaleDistMulti(HMM *phmm, int T,  double **alpha,
		      double *scale, double *pprob, int *ERROR_IND);

void Backward(HMM *phmm, int T,  double **beta, double *pprob);
void BackwardWithScale(HMM *phmm, int T,  double **beta,
        double *scale, double *pprob);
void BackwardWithScaleDist(HMM *phmm, int T,  double **beta,
        double *scale, double *pprob);
void BackwardWithScaleDistMulti(HMM *phmm, int T,  double **beta,
        double *scale, double *pprob);


/* Modified : int *O removed, dim(B) is changed to T * M */ 
double *** AllocXi(int T, int N);
void FreeXi(double *** xi, int T, int N);

void ComputeGamma(HMM *phmm, int T, double **alpha, double **beta,
		  double **gamma, gsl_matrix *weight_m);
void ComputeXi(HMM* phmm, int T,  double **alpha, double **beta,
        double ***xi);
void ComputeXiDist(HMM* phmm, int T,  double **alpha, double **beta,
        double ***xi);
void ComputeXiDistMulti(HMM* phmm, int T,  double **alpha, double **beta,
        double ***xi);
void Viterbi(HMM *phmm, int T, double **delta, int **psi,
        int *q, double *pprob);
void ViterbiLog(HMM *phmm, int T, double **delta, int **psi,
        int *q, double *pprob);
void ViterbiLogDist(HMM *phmm, int T, double **delta, int **psi,
        int *q, double *pprob);
void ViterbiLogDistMulti(HMM *phmm, int T, double **delta, int **psi,
        int *q, double *pprob);



/* random number generator related functions*/

int hmmgetseed(void);
void hmmsetseed(int seed);
double hmmgetrand(void);
 
#define MAX(x,y)        ((x) > (y) ? (x) : (y))
#define MIN(x,y)        ((x) < (y) ? (x) : (y))
 
