/* This file contains the main Baum-Welch algorithm 
   Marked with * if not used in MRHMMs 
   
   BaumWelch : Baum Welch algorithm
   BaumWelch_itr : Runs BaumWelch and 
                   compare the loglikelihood to decide to write the 
		   result file or not
		   
  * Read_multiple_data_sets : 
*/

#include "_struct.h"
#include "hmm.h"
#include "HMM.Read.Matrix.h"
#include "mvnorm.h"
#include "nrutil.h"
#include "models.h"
#include "gsl_BaumWelch.h"
#include <time.h>
#include <unistd.h>
#include <omp.h>


void BaumWelch(HMM *phmm, PARM_HMM *parm_hmm1, INI *ini, gsl_matrix *X, gsl_matrix *Y, int T,  double **alpha, double **beta, double **gamma, int *niter,  double *plogprobinit, double *plogprobfinal, gsl_vector *dist, int *ERROR_IND)
{

  /* Note : alpha and beta do not have to be passed */
        int	i, j, k ;
	int	t, l = 0;
	double DELTA;
	double	logprobf[1]={0}, logprobb;
	double	numeratorA, denominatorA;

	double ***xi, *scale;
	double Delta,  logprobprev;
	gsl_matrix *weight_m;
	gsl_matrix *gamma_ij;
	gsl_matrix *design_X ;
	gsl_vector  *X_col = gsl_vector_alloc(X -> size1);
	gsl_vector *Y_col;
	gsl_vector *weight_col;
	gsl_matrix *coef_m;
	double logistic_tol = .00001;
	int state, print_ind = 0, read_Y_IND=1, data_set_number_count = 0;
	int Max_itr, num_data_sets, data_set_number;
  
	double **delta,  A_t_sum, D1 = 30000;
	int **psi;
	int *q;

	clock_t A_start, A_end, multiMVN_weight_start, multiMVN_weight_end, 
	  singleMVN_regression_start, singleMVN_regression_end, LogMaxScale_start, LogMaxScale_end, CalBfromB1_start, CalBfromB1_end, forward_start, forward_end, backward_start, backward_end, gamma_start, gamma_end, xi_start, xi_end;
	double A_time=0, multiMVN_weight_time=0, singleMVN_regression_time = 0 , LogMaxScale_time = 0, CalBfromB1_time = 0, forward_time =0 , backward_time = 0, gamma_time =0, xi_time =0 ;
	double *time, *time2;

	Max_itr = ini-> MaxItr;
	DELTA = ini -> Tolerance;
	num_data_sets = ini -> num_data_sets;
	ERROR_IND[0] = 0;
	weight_m = gsl_matrix_alloc(T, phmm->N);
	time = malloc(4 * sizeof(double));
	time2 = malloc(2 * sizeof(double));
	time[0]=0; time[1]=0; time[2]=0; time[3]=0;
	time2[0] = 0; time2[1] = 0;
	D1 = (double) phmm->D1;


	/* Allocate memory specific for the model and
	   create the design Matrix */

	if(!strcmp(phmm-> modelname, "singleMVN_regression")|!strcmp(phmm-> modelname, "multiMVN_regression_state")|!strcmp(phmm-> modelname, "single_regression")| !strcmp(phmm-> modelname, "single_logistic")|  !strcmp(phmm-> modelname, "singleMVN_logistic")  ){
	  design_X = gsl_matrix_alloc(X->size1, (X->size2)+1 );
	  gsl_vector_set_all(X_col, 1);
	  gsl_matrix_set_col(design_X, 0, X_col);

	  for( i = 0; i < X-> size2; i++)
	    {
	      gsl_matrix_get_col(X_col,  X, i);
	      gsl_matrix_set_col(design_X, i+1, X_col );
	    }
  
	}

	if(!strcmp(phmm->modelname, "single_logistic") | !strcmp(phmm->modelname, "singleMVN_logistic") ){
	  coef_m = gsl_matrix_alloc(phmm->N, X->size2 +1);
	   Y_col = gsl_vector_alloc(X -> size1);
	   weight_col = gsl_vector_alloc(X -> size1);

	  
	   for( i = 0; i < X-> size1; i++)
	     {
	       gsl_vector_set(Y_col,i,  gsl_matrix_get(Y, i,0));
	     }
	   if(0)printf("BaumWelch: sum(y) = %f\n", gsl_vector_sum(Y_col));


	}
	if((!strcmp(phmm-> modelname, "multiMVN_regression")) |
	   (!strcmp(phmm-> modelname, "hardEM_multiMVN_regression"))){
	  /* design matrix */
	   design_X = gsl_matrix_alloc(X->size1, (X->size2)+1 );
	   gsl_vector_set_all(X_col, 1);
	   gsl_matrix_set_col(design_X, 0, X_col);

	  for( i = 0; i < X-> size2; i++)
	    {
	      gsl_matrix_get_col(X_col,  X, i);
	      gsl_matrix_set_col(design_X, i+1, X_col );
	    }
	  gamma_ij = gsl_matrix_alloc(X->size1, (phmm->N * phmm->mix_comp));

	  if(0) gsl_matrix_print(design_X);
	}

	/* Change here for HardEM */
	if(!strcmp(phmm-> modelname, "multiMVN_regression") |!strcmp(phmm-> modelname, "single_regression") ){
	
	  delta = dmatrix(1, phmm-> M , 1, phmm-> N);
	  psi = imatrix(1, phmm-> M, 1, phmm-> N);
	  q = (int *) ivector(1, phmm-> M);
	}
	
	xi = AllocXi(T, phmm->N);
	scale = dvector(1, T);

	
	/* Initial iteration */

        if(phmm->window_dist == 0){
	  ForwardWithScale(phmm, T, alpha, scale, logprobf, ERROR_IND);
	  if(ERROR_IND[0]) return;
	  *plogprobinit = logprobf[0] + phmm->B_scale;
	  /* log P(O |intial model) */
	  if(print_ind) printf("T = %d\n", T);
	  if(0) printf("logprob = %e, B_scale = %f \n", logprobf[0], phmm->B_scale);
	  BackwardWithScale(phmm, T, beta, scale, &logprobb);
	  ComputeGamma(phmm, T, alpha, beta, gamma, weight_m);
	  ComputeXi(phmm, T, alpha, beta, xi);
	}



	if(phmm->window_dist == 1){
	  /* A_t : Transition probabilities depend on window distance */
	  if(D1>0){
	      for(i = 1; i <=phmm-> M -1; i++){
		if( gsl_vector_get(dist, i-1) > 0){
		  for(j = 1; j<=phmm->N ; j++){
		    A_t_sum = 0;
		    for(k = 1; k <=phmm->N ;k++){
		      if(j != k){
			phmm->A_t[i][j][k] = (1 - exp(-gsl_vector_get(dist, i-1)/(double) D1 )) * phmm->A[j][k];
			 A_t_sum +=phmm->A_t[i][j][k];
		      } 
		    }
		    phmm->A_t[i][j][j] = 1 - A_t_sum;
		  }

 		} else {
		  for(j = 1; j<=phmm->N ; j++){
		      for(k = 1; k <=phmm->N ;k++){
			phmm->A_t[i][j][k] = phmm->pi[k];
		      }
		  }
		}
	      }
	  }
	
	}

	if( (phmm-> window_dist ==1) & (phmm-> num_chr ==1) ){
	  ForwardWithScaleDist(phmm, T,  alpha, scale, logprobf, ERROR_IND);	
	 
	  *plogprobinit = logprobf[0] + phmm->B_scale;	

	  BackwardWithScaleDist(phmm, T, beta, scale, &logprobb);
	  ComputeGamma(phmm, T, alpha, beta, gamma, weight_m);	
	  ComputeXiDist(phmm, T,  alpha, beta, xi);
	}

	if((phmm-> window_dist ==1) & (phmm-> num_chr > 1) ){
	  ForwardWithScaleDistMulti(phmm, T,  alpha, scale, logprobf, ERROR_IND);
	  *plogprobinit = logprobf[0] + phmm->B_scale;

	  BackwardWithScaleDistMulti(phmm, T, beta, scale, &logprobb);
	  ComputeGamma(phmm, T, alpha, beta, gamma, weight_m);
	  ComputeXiDistMulti(phmm, T,  alpha, beta, xi);
	}


	if(print_ind){
	  printf("logprobf = %f \n", logprobf[0] + phmm->B_scale);
	  printf("logprobb = %f \n", logprobb);
	}

	logprobprev = logprobf[0] + phmm->B_scale;

      

	/* Iteration begins */
	do  {	


	  printf("itr = %d : Loglikelihood %f\n", l, logprobprev);

	  /* Switch to another data set 
	  if(num_data_sets > 1){
	    
	    if(!strcmp(phmm-> modelname, "singleMVN")|
	       !strcmp(phmm->modelname, "multiMVN"))
	      { 
		read_Y_IND = 0;
	      }
	    data_set_number= ini-> dim_data_sets[data_set_number_count];
	    Read_multiple_data_sets(ini, X, Y, data_set_number, read_Y_IND);
	    data_set_number_count +=1;

	  }

	  */


	  /* reestimate frequency of state i in time t=1 */
	 if(phmm->num_chr == 1){
	   for (i = 1; i <= phmm->N; i++) {
	     phmm->pi[i] = gamma[1][i];
	   } 
	 }
	 
	 if( (phmm->window_dist == 1) & (phmm->num_chr > 1) ){
	    for (i = 1; i <= phmm->N; i++) {
	      phmm->pi[i]=0;
	      for( j = 1; j <= phmm->num_chr ;j++){
		phmm->pi[i] += gamma[phmm->new_chr[j]][i];
	      }
	      phmm->pi[i] /= (double)phmm->num_chr;
	      
	    }
	 }
	
	 /* reestimate transition matrix  and symbol prob in
		   each state */

	  if(phmm->window_dist == 0){

	    for (i = 1; i <= phmm->N; i++) {
	      phmm->pi[i] = gamma[1][i];
	    } 
	    A_start = clock();
	    for (i = 1; i <= phmm->N; i++) { 
	      denominatorA = 0.0;
	      for (t = 1; t <= T - 1; t++) 
		denominatorA += gamma[t][i];
	      
	      for (j = 1; j <= phmm->N; j++) {
		numeratorA = 0.0;
		for (t = 1; t <= T - 1; t++) 
		  numeratorA += xi[t][i][j];
		phmm->A[i][j] = numeratorA/denominatorA;
	      }
	      
	    }
	    A_end = clock();
		A_time += (double)(difftime(A_end , A_start));
	  
		/*  Independent set up 

		for(i= 1; i <= phmm->N; i++) { 
		  for (j = 1; j <= phmm->N; j++) {
		    phmm->A[i][j] = (double) 1/(double)phmm->N;
		  }
		} */
	  }

	  if( ( phmm->window_dist == 1) &( phmm->num_chr == 1) ){
	    if(D1 >0){
	      for (i = 1; i <= phmm->N; i++) { 
		denominatorA = 0.0;
		for (t = 1; t <= T - 1; t++) 
		  denominatorA +=  gamma[t][i];
		
		for (j = 1; j <= phmm->N; j++) {
		  numeratorA = 0.0;
		  for (t = 1; t <= T - 1; t++) {
		    if(gsl_vector_get(dist, t-1) > 0) /*Here */
		      numeratorA +=  xi[t][i][j];
		  }
		  phmm->A[i][j] = numeratorA/denominatorA;
		}
		
	      }
	      for(i = 1; i <=phmm-> M -1; i++){
		if( gsl_vector_get(dist, i-1) > 0){
		  for(j = 1; j<=phmm->N ; j++){
		    A_t_sum = 0;
		    for(k = 1; k <=phmm->N ;k++){
		      if(j != k){
			phmm->A_t[i][j][k] = (1 - exp(-gsl_vector_get(dist, i-1)/(double)D1 )) * phmm->A[j][k];
			A_t_sum +=phmm->A_t[i][j][k];
		      }
		    }
		    phmm->A_t[i][j][j] = 1 - A_t_sum;
		  }
		} else {
		  printf("Error: The distance is not positive.\n");
		  exit(1);
		}
	      }
	    }
	    if (D1 == 0){
	      printf("Error: D = 0 is for multiple independent data sets. The Obs.loc data may increasing continuously.\n");
		exit(1);
	      
	    }
	  }
	  
		      
	    

	 
	  if( (phmm->window_dist == 1) & (phmm->num_chr > 1) ){
	   
	    for (i = 1; i <= phmm->N; i++) { 
	      denominatorA = 0.0;
	      for (t = 1; t <= T - 1; t++) 
		denominatorA +=  gamma[t][i];
	      
	      for (j = 1; j <= phmm->N; j++) {
		numeratorA = 0.0;
		for (t = 1; t <= T - 1; t++) {
		  if(gsl_vector_get(dist, t-1) > 0) /*Here */
		    numeratorA +=  xi[t][i][j];
		}
		  phmm->A[i][j] = numeratorA/denominatorA;
	      }
	    }	     
	    if(D1 > 0){
	      /* A_t : Transition probabilities depend on window distance */
	      for(i = 1; i <=phmm-> M -1; i++){
		if( gsl_vector_get(dist, i-1) > 0){
		  for(j = 1; j<=phmm->N ; j++){
		      A_t_sum = 0;
		      for(k = 1; k <=phmm->N ;k++){
			if(j != k){
			  phmm->A_t[i][j][k] = (1 - exp(-gsl_vector_get(dist, i-1)/(double)D1 )) * phmm->A[j][k];
			  A_t_sum +=phmm->A_t[i][j][k];
			}
		      }
		      phmm->A_t[i][j][j] = 1 - A_t_sum;
		  }
		} else {
		  for(j = 1; j<=phmm->N ; j++){
		    for(k = 1; k <=phmm->N ;k++){
		      phmm->A_t[i][j][k] = phmm->pi[k];
		    }
		  }
		}
	      }
	    }
	    
	  }
	    
	    /* Find the probabilities of observations */
		if(print_ind){
		  printf("dim (weight_m) = %d, %d \n",  (int) weight_m->size1, 
			 (int) weight_m->size2);
		  gsl_matrix_print(weight_m);
		}
	       
		if(!strcmp(phmm-> modelname, "singleMVN")){
		  singleMVN(X, weight_m, phmm, parm_hmm1, ini,  ERROR_IND);
		  /* phmm->Braw = &(phmm->B);*/
		  LogMaxScaleB(phmm);
		}
				
		if(!strcmp(phmm-> modelname, "multiMVN")){
		  multiMVN(X, weight_m, phmm, parm_hmm1, ini, ERROR_IND);
		  /* Scale B1 given state */
		  LogMaxScaleB1(phmm, ERROR_IND);
		  
		  /* B1 and B in HMM */
		  CalculateB_from_B1(phmm, parm_hmm1); 
   
		 
		  if(print_ind){
		    printf("c\n");
		    dmatrix_print(parm_hmm1->c, 2,2);
		    printf("B_i \n");
		    dmatrix_print(phmm-> B1[1], phmm->M, 2);
		    dmatrix_print(phmm-> B1[2], phmm->M, 2);
		  }
		}
		
		if(!strcmp(phmm-> modelname, "singleMVN_regression")){
		  singleMVN(X, weight_m, phmm, parm_hmm1, ini, ERROR_IND);
		  if(ERROR_IND[0]) break;
		  combined_singleMVN_regression(X, Y, design_X, weight_m, 
						phmm, parm_hmm1, ini, time,
						ERROR_IND);
	
		  if(ERROR_IND[0]) break;
		  LogMaxScaleB(phmm);
		}
	
		if(!strcmp(phmm-> modelname, "multiMVN_regression_state")){
		  multiMVN(X, weight_m, phmm, parm_hmm1, ini, ERROR_IND);
		  if(ERROR_IND[0]) break;
  		  /* singleMVN_regression_no_intercept(X, Y,  weight_m, 
				       phmm, parm_hmm1, time, ERROR_IND);

				       printf("B_scale = %f \n", phmm->B_scale);
		  */
		  CalculateB_from_B1_Log(phmm, parm_hmm1);
		  combined_singleMVN_regression(X, Y, design_X,  weight_m, 
						phmm, parm_hmm1, ini, time, 
						ERROR_IND);
		  
		 

		 

		  if(ERROR_IND[0]) break;
		  LogMaxScaleB1(phmm, ERROR_IND);
		  if(ERROR_IND[0]) break;
		  CalculateB_from_B1(phmm, parm_hmm1); 
   
		}

		if(!strcmp(phmm-> modelname, "multiMVN_regression")){
		  multiMVN_weight_start = omp_get_wtime();
		  multiMVN_weight(X, weight_m, phmm, parm_hmm1, gamma_ij, time2,
				  ini,  ERROR_IND);
		  multiMVN_weight_end = omp_get_wtime();
		  multiMVN_weight_time += (double) (difftime(multiMVN_weight_end , multiMVN_weight_start));
		  
		  if(0)printf("multiMVN_weight ended\n");
		  if(ERROR_IND[0]){	
		    FreeXi(xi, T, phmm->N);
		    free_dvector(scale, 1, T);
		    gsl_matrix_free(weight_m);
		    free(time);
		    if(!strcmp(phmm->modelname, "singleMVN_regression")){
		      gsl_matrix_free(design_X);}
		    if(!strcmp(phmm->modelname, "multiMVN_regression")){
		      gsl_matrix_free(gamma_ij);}
		    gsl_vector_free(X_col);
		    break;
		  }
	
		  singleMVN_regression_start = omp_get_wtime();
		  /*ViterbiLog(phmm, T,  delta, psi, q, &pprob);
		  hardEM_multiMVN_regression_new(design_X, Y, gamma_ij, q, 
		    phmm, parm_hmm1, ERROR_IND);*/
		  combined_singleMVN_regression(X, Y, design_X, gamma_ij, 
						phmm, parm_hmm1,ini, time, 
						ERROR_IND);
		  singleMVN_regression_end = omp_get_wtime();
		  singleMVN_regression_time += (double)(difftime(singleMVN_regression_end, singleMVN_regression_start)); 
		  if(ERROR_IND[0]) break;
		

		  LogMaxScale_start = clock();
		  LogMaxScaleB1(phmm, ERROR_IND);
		  LogMaxScale_end = clock();
		  LogMaxScale_time += (double)(difftime(LogMaxScale_end ,
							LogMaxScale_start));
		  if(ERROR_IND[0]) break;
		  if(0)printf("Logmaxscale ended\n");
		  CalBfromB1_start = clock();
		  CalculateB_from_B1(phmm, parm_hmm1); 
		  CalBfromB1_end = clock();
		  CalBfromB1_time += (double)(difftime(CalBfromB1_end , 
						       CalBfromB1_start));
		  if(0)printf("CalBfromB1 ended\n");
   		}
		
		if(!strcmp(phmm-> modelname, "single_regression")){
		  /* ViterbiLog(phmm, T,  delta, psi, q, &pprob);
		  hardEM_multiMVN_regression_new(design_X, Y, weight_m, q, 
					     phmm, parm_hmm1, ERROR_IND);
		  
		  singleMVN_regression_no_intercept(X, Y,  weight_m, 
		  phmm, parm_hmm1, time, ERROR_IND);
		  */
		  combined_singleMVN_regression(X, Y, design_X, weight_m, phmm, parm_hmm1, ini, time, ERROR_IND);
		  
		  LogMaxScaleB(phmm);
		}
		
		
		if(!strcmp(phmm-> modelname, "single_logistic")){
		
		  for(state = 1; state <= phmm->N; state++){

		    gsl_matrix_get_col(weight_col, weight_m, state-1);
		    logistic_regression(design_X,  Y_col, weight_col, coef_m, 
					state-1, logistic_tol, ERROR_IND);
		    Logistic_B(Y_col, design_X, coef_m, state-1, phmm);
		  }
		  
		  gsl_matrix_to_dmatrix(coef_m, parm_hmm1->reg_coef);
		  LogMaxScaleB(phmm);
		   
		}
		
		if(!strcmp(phmm-> modelname, "singleMVN_logistic")){
		  
		  singleMVN(X, weight_m, phmm, parm_hmm1, ini, ERROR_IND);

		  for(state = 1; state <= phmm->N; state++){
		    gsl_matrix_get_col(weight_col, weight_m, state-1);

		    logistic_regression(design_X,  Y_col, weight_col, coef_m, 
					state-1, logistic_tol, ERROR_IND);
		    
		    if(ERROR_IND[0]) break;
		  
		    Logistic_B(Y_col, design_X, coef_m, state-1, phmm);

		  }
		  
		  gsl_matrix_to_dmatrix(coef_m, parm_hmm1->reg_coef);
		  LogMaxScaleB(phmm);
		   
		}
		
		
		
		if(phmm-> window_dist ==0){
		  if(print_ind) printf("BaumWelch_after mvdnorm_matrix");
		forward_start = clock();
		/*Forward(phmm, T, alpha, logprobf);*/
         	 ForwardWithScale(phmm, T,  alpha, scale, logprobf, ERROR_IND);
		forward_end = clock();
		if(0) printf("forward ended \n");
		forward_time += (double)(difftime(forward_end , forward_start));

		backward_start = clock();
		/*Backward(phmm, T, beta, &logprobb);*/
		BackwardWithScale(phmm, T, beta, scale, &logprobb);
		backward_end = clock();
		backward_time += (double)(difftime(backward_end, backward_start));

		if(0) printf("backward ended\n");
		gamma_start = clock();
		ComputeGamma(phmm, T, alpha, beta, gamma, weight_m);
		if(0)printf("Gamma ended\n");
		gamma_end = clock();
		gamma_time += (double)(difftime(gamma_end , gamma_start));
		
		xi_start = clock();
		ComputeXi(phmm, T,  alpha, beta, xi);
		xi_end = clock();
		xi_time += (double) (difftime( xi_end , xi_start));

		if(0)printf("xi ended\n");
		}
	
		if( (phmm-> window_dist ==1) & (phmm->num_chr == 1)){
		  ForwardWithScaleDist(phmm, T,  alpha, scale, logprobf, ERROR_IND);
		  BackwardWithScaleDist(phmm, T, beta, scale, &logprobb);
		  ComputeGamma(phmm, T, alpha, beta, gamma, weight_m);
		  ComputeXiDist(phmm, T,  alpha, beta, xi);
		}
		
		if( (phmm-> window_dist ==1) & (phmm-> num_chr > 1) ){
		 
		  ForwardWithScaleDistMulti(phmm, T,  alpha, scale, logprobf, ERROR_IND);
		  BackwardWithScaleDistMulti(phmm, T, beta, scale, &logprobb);
		  ComputeGamma(phmm, T, alpha, beta, gamma, weight_m);
		  ComputeXiDistMulti(phmm, T,  alpha, beta, xi);
		 
	
		}



		/* compute difference between log probability of 
		   two iterations */
		Delta = logprobf[0] + phmm->B_scale - logprobprev; 
		logprobprev = logprobf[0] + phmm->B_scale;
		/* 
		   if(0) printf("logprob = %e, B_scale = %f \n", logprobf[0], phmm->B_scale);
		   Positive likelihood is not an error. 
		   if(logprobprev > 0){
		   ERROR_IND[0]=1;
		   }

		*/
		l++;

	} while((Delta > DELTA) & (l < Max_itr) & (ERROR_IND[0] == 0) ); 
	/* if log probability does not change much, exit */ 

	/* Calculate the observation probability in logscale */

	if(ERROR_IND[0]==0){
	  printf("itr = %d : Loglikelihood  %f \n", l, logprobf[0] + phmm->B_scale);
	  /*Observation_prob(X, design_X, Y, Y_col, weight_m,  coef_m, gamma_ij, phmm,parm_hmm1, ini, time, ERROR_IND);*/
	  
	}
	

	if(0) printf(" A: %f \n multiMVN_weight : %f \n singleMVN_regression : %f \n LogMaxScale : %f \n CalBfromB1 : %f \n forward : %f \n backward : %f \n gamma : %f \n xi : %f \n", A_time/CLOCKS_PER_SEC , multiMVN_weight_time  , singleMVN_regression_time , LogMaxScale_time/CLOCKS_PER_SEC  , CalBfromB1_time/CLOCKS_PER_SEC  , forward_time/CLOCKS_PER_SEC  , backward_time/CLOCKS_PER_SEC , gamma_time/CLOCKS_PER_SEC , xi_time/CLOCKS_PER_SEC );
    
	if(0) printf(" Regression : %f \n  Residuals : %f \n Weighted Variance : %f \n  mvdnorm_matrix : %f \n", time[0], time[1], time[2], time[3]);

	if(0) printf(" multiMVN_Weight : gamma_ij %f \n  mvdnorm_matrix: %f \n", time2[0], time2[1]);


	*niter = l;
	*plogprobfinal = logprobf[0] + phmm->B_scale; 
	/* log P(O|estimated model) */
	FreeXi(xi, T, phmm->N);
	free_dvector(scale, 1, T);
	gsl_matrix_free(weight_m);
   	free(time);
	if(!strcmp(phmm->modelname, "singleMVN_regression")){
	  gsl_matrix_free(design_X);}
	/* here */
	if(!strcmp(phmm->modelname, "multiMVN_regression")){
	  gsl_matrix_free(gamma_ij);}
	gsl_vector_free(X_col);
	/* here */
	if(!strcmp(phmm->modelname, "multiMVN_regression")){
	  free_dmatrix(delta ,1, phmm->M, 1, phmm-> N);
	  free_imatrix(psi, 1, phmm->M, 1, phmm-> N);
	  free_ivector(q, 1,phmm->M);
	}
}


void BaumWelch_itr(int rep, HMM *hmm1, PARM_HMM *parm_hmm1, INI *ini, gsl_matrix *X, gsl_matrix *Y, int T,  double **alpha, double **beta, double **gamma, int *niter, int Max_itr, double Tolerance, double *logprobprevItr, int *ERROR_IND)
{

  double pprob;
  double **delta;
  int **psi;
  int i, *q, count;
  double  logprobinit, logprobfinal, temp, temp_dist ;
  gsl_vector *dist, *loc;
  char *loglikeitr;
  FILE *file_loglike_itr, *file_dist;

  /* Read window_dist file */
  if(hmm1->window_dist == 1){
    loc = gsl_vector_alloc(hmm1->M);
    dist =  gsl_vector_alloc(hmm1->M-1);
    if((file_dist = fopen(ini->loc_window_dist, "rb"))==NULL){
      printf("Error! Cannot open ini.loc_window_dist file: %s\n", ini->loc_window_dist) ;
      exit(1);
    }
   
    if(gsl_vector_fscanf(file_dist, loc)){
      printf("The distance may include NA or NaN.\n");
    }
    
    fclose(file_dist);
    for( i = 1; i < T; i++){
      temp_dist = gsl_vector_get(loc, i) - gsl_vector_get(loc, i-1);
      gsl_vector_set(dist, i-1, temp_dist);
    }
   
   count = 1;
   hmm1->new_chr[1]=1;
   for(i = 0; i< T -1 ; i++){
     if(gsl_vector_get(dist, i) < 0){
       count++;
       hmm1->new_chr[count] = i+2;
      }
   }
      hmm1->new_chr[count+1] = T +1;
      hmm1->num_chr = count;

         
      hmm1->B_scale_multi = dvector(1, count);

  }




		      
  loglikeitr = malloc(100*sizeof(char));
  ERROR_IND[0]=0;
  printf("Repeat %d \n", rep);
  /* print mu and all for test 
   delta = dmatrix(1, hmm1-> M , 1, hmm1-> N);
      psi = imatrix(1, hmm1-> M, 1, hmm1-> N);
      q = (int *) ivector(1, hmm1-> M);
    
  if((hmm1 -> window_dist == 1) & (hmm1->num_chr >1)){
	  ViterbiLogDistMulti(hmm1, T, delta, psi, q, &pprob);
	}

	PrintHMM_All_Results(ini->output_filename, hmm1, parm_hmm1, q, logprobfinal, pprob, gamma);
	
	free_dmatrix(delta ,1 hmm1->M, 1, hmm1-> N);
	free_imatrix(psi, 1, hmm1->M, 1, hmm1-> N);
	free_ivector(q, 1,hmm1->M);
	logprobprevItr[0] = logprobfinal
    exit(1);     
    *** till here */


  BaumWelch(hmm1, parm_hmm1,  ini, X, Y, T,  alpha, beta, gamma,  niter, &logprobinit, &logprobfinal, dist, ERROR_IND);

  /* Find if file *RepLoglike exists and compare the loglikelihood */
  if(ERROR_IND[0] == 0){
    sleep(.1);
    if(rep == 1){
      /* Get the maximum likelihood if there is a file already */
       StringPaste(loglikeitr, ini->output_filename, "_RepLoglike");
       count = 1;
       if(( file_loglike_itr = fopen(loglikeitr, "r")) !=NULL){
	 while(!feof(file_loglike_itr)){
	   fscanf(file_loglike_itr, "%lf", &temp);
	   fscanf(file_loglike_itr, "%lf", &temp);
	   if(count ==1){  logprobinit = temp; }
	   if(0) printf("Reploglike %f \t %f \n", temp, logprobinit);
	   logprobinit =MAX(logprobinit, temp);
	   count = count +1;
	 }
	 fclose(file_loglike_itr);
       } 
       logprobprevItr[0] = logprobinit;
    }
    
    if(0){
      delta = dmatrix(1, hmm1-> M , 1, hmm1-> N);
      psi = imatrix(1, hmm1-> M, 1, hmm1-> N);
      q = (int *) ivector(1, hmm1-> M);
      
      ViterbiLog(hmm1, T,  delta, psi, q, &pprob);
      PrintHMM_All_Results_rep(ini->output_filename, hmm1, parm_hmm1, q, logprobfinal, gamma, pprob, rep);
      
      free_dmatrix(delta ,1, hmm1->M, 1, hmm1-> N);
      free_imatrix(psi, 1, hmm1->M, 1, hmm1-> N);
      free_ivector(q, 1,hmm1->M);
    }

    
    printf("rep = %d, Loglikelihood = %lf: The largest loglikelihood = %lf \n", rep, logprobfinal, logprobprevItr[0]);
    
    if( logprobfinal > logprobprevItr[0])
      {
	delta = dmatrix(1, hmm1-> M , 1, hmm1-> N);
	psi = imatrix(1, hmm1-> M, 1, hmm1-> N);
	q = (int *) ivector(1, hmm1-> M);
	
	/* Estimate the best states using Viterbi algorithm */
	if(hmm1->window_dist ==0){
	  ViterbiLog(hmm1, T,  delta, psi, q, &pprob);
	}
	if((hmm1->window_dist == 1) & (hmm1->num_chr ==1 )){
	  ViterbiLogDist(hmm1, T, delta, psi, q, &pprob);
	}
	if((hmm1 -> window_dist == 1) & (hmm1->num_chr >1)){
	  ViterbiLogDistMulti(hmm1, T, delta, psi, q, &pprob);
	}

	/* Write the result */
	pprob = pprob + hmm1->B_scale;
	PrintHMM_All_Results(ini->output_filename, hmm1, parm_hmm1, q, logprobfinal, pprob, gamma);
	
	free_dmatrix(delta ,1, hmm1->M, 1, hmm1-> N);
	free_imatrix(psi, 1, hmm1->M, 1, hmm1-> N);
	free_ivector(q, 1,hmm1->M);
	logprobprevItr[0] = logprobfinal;
      }

    StringPaste(loglikeitr, ini->output_filename, "_RepLoglike");
    file_loglike_itr = fopen(loglikeitr, "a+");
    fprintf(file_loglike_itr, "%d \t %lf  \n", rep, logprobfinal);
    fclose(file_loglike_itr);

   

    
  }

  free(loglikeitr);
}


  /*
  BaumWelch(&hmm1, &parm_hmm1,  X, Y, T,  alpha, beta, gamma,  &niter,  MaxItr,  Tolerance, &logprobinit, &logprobfinal, ERROR_IND);
  if(ERROR_IND[0] == 0){
    sleep(.1);
    if(itr == 1){
      logprobprevItr = logprobinit;
      StringPaste(loglikeitr, ini.output_filename, "_RepLoglike");
    } 
   
    if(logprobfinal > logprobprevItr)
      {
	ViterbiLog(&hmm1, T,  delta, psi, q, &pprob);
	PrintHMM_All_Results(ini.output_filename, &hmm1, &parm_hmm1, q, logprobfinal, gamma);
      }
    
    fflush(stderr);
    StringPaste(loglikeitr, ini.output_filename, "_RepLoglike");
    file_loglike_itr = fopen(loglikeitr, "a+");
    fprintf(file_loglike_itr, "rep %d : %f \n", itr, logprobfinal);
    fclose(file_loglike_itr);
  } else {
    itr = itr - 1;
  }
    */


void Read_multiple_data_sets(INI *ini, gsl_matrix *X, gsl_matrix *Y, int data_set_number, int read_Y_IND)
{
  char *dataX_with_number;
  char *dataY_with_number;
  FILE *file;

  /* Reread X_number data */
  dataX_with_number = malloc(100*sizeof(char));
  gsl_matrix_free(X);
  gsl_matrix_alloc(data_set_number, ini->colX);

  sprintf(dataX_with_number, "%s_%d", ini->loc_Data_X, data_set_number);
  file = fopen(dataX_with_number, "rb");
  gsl_matrix_fscanf(file, X);

  free(dataX_with_number);

  if(read_Y_IND ==1){
    /* Reread Y_number data */
    dataY_with_number = malloc(100 * sizeof(char));
    gsl_matrix_free(Y);
    gsl_matrix_alloc(data_set_number, ini->colY);
    
    sprintf(dataY_with_number, "%s_%d", ini->loc_Data_Y, data_set_number);
    file = fopen(dataY_with_number, "rb");
    gsl_matrix_fscanf(file, Y);
    free(dataY_with_number);
  }
 

   
}
