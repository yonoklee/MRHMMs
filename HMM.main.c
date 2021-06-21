/*  
    File HMM.main.c
    Main function of MRHMMs
*/


     #include <math.h>
     #include <string.h>
     #include <malloc.h>
     #include <time.h>
     #include <stdlib.h>
     #include <stdio.h>
     #include <unistd.h>

     #include <gsl/gsl_matrix.h>
     #include <gsl/gsl_vector.h>
     #include <gsl/gsl_linalg.h>
     #include <gsl/gsl_blas.h>
     #include <gsl/gsl_rng.h>
     #include <gsl/gsl_randist.h>
   
     #include "nrutil.h"
     #include "mvnorm.h"
     #include "hmm.h"
     #include "HMM.Read.Matrix.h"
     #include "models.h"
     #include "gsl_BaumWelch.h"
     


double myround( double value, int prec);

int main( int argc, char **argv) 
{
  HMM hmm1;
  PARM_HMM parm_hmm1;
  INI ini;

  int i,j,k , itr;
  FILE  *file_weight;

  int *q;
  int T;
  double 	**alpha; 
  double	**beta;
  double	**gamma;
  int	niter;

  int MaxItr;
  double Tolerance, logprobprevItr[1];
  
  gsl_matrix *X, *Y, *weight_m, *design_X;
  gsl_vector *weight_col, *X_col, *X_row, *Y_col;
  double  tmp;
  int  *ERROR_IND, total_rep=0;


  
  if (argc != 2){
    printf("Error!\n");
    printf("Please enter the file location with initial model information. \n");
    exit(1);
  }
  printf("\t\t Information will be read from %s. \n", argv[1]);


  /* Read input file and print the reads */
  read_initial(argv[1], &ini, &hmm1, &parm_hmm1);
  PrintInitial(&ini, &hmm1);
  
  /* Initialization  */
  q =  (int *) ivector(1, hmm1.M); 
  for(i=1; i<=hmm1.M; i++){
    q[i] = 0;
  }
  
  MaxItr = ini.MaxItr;
  Tolerance = ini.Tolerance;
  parm_hmm1.N = hmm1.N;
  parm_hmm1.p = ini.colX;
  parm_hmm1.d = ini.colY;
  parm_hmm1.mix_comp = hmm1.mix_comp;
 
  

   hmm1.B = dmatrix(1, hmm1.M, 1, hmm1.N);
   hmm1.new_chr = ivector(1, 25);
   
     
   parm_hmm1.mu = dmatrix(1, hmm1.N * hmm1.mix_comp, 1, ini.colX);
   parm_hmm1.Sigma = AllocXi(hmm1.M * hmm1.mix_comp , ini.colX);
   if(hmm1.window_dist == 1){
   	hmm1.A_t = AllocArray(hmm1.M, hmm1.N, hmm1.N);
   }
  
  /* Memory allocation */
   weight_m = gsl_matrix_alloc(hmm1.M, hmm1.N);
   X = gsl_matrix_alloc(hmm1.M, ini.colX);
   Y = gsl_matrix_alloc(hmm1.M, ini.colY);

   
   design_X = gsl_matrix_alloc(hmm1.M, ini.colX+1 );
   weight_col = gsl_vector_alloc(hmm1.M);
   X_col = gsl_vector_alloc(hmm1.M);
   X_row = gsl_vector_alloc(ini.colX); 
   T = ini.rowX;  
   
   /* Read Data.X */
   if(( file_weight = fopen(ini.loc_Data_X, "rb")) == NULL){
     printf("Error! Cannot open Data.X file\n") ;
     exit(1);
   }
   gsl_matrix_fscanf(file_weight, X);
   fclose(file_weight);
   
   /* Read Data.Y */
   if((file_weight = fopen(ini.loc_Data_Y, "rb"))==NULL){
     printf("Error! Cannot open Data.Y file\n") ;
     exit(1);
   }
   gsl_matrix_fscanf(file_weight, Y);
   fclose(file_weight);

   
   if(0) printf("The first element of X= %lf\n", gsl_matrix_get(X, 0, 0));
  
   ERROR_IND = malloc(sizeof(int));
   ERROR_IND[0]= 0;
   
   alpha = dmatrix(1, hmm1.M, 1, hmm1.N);
   beta = dmatrix(1, hmm1.M, 1, hmm1.N);
   gamma = dmatrix(1, hmm1.M, 1, hmm1.N);  

   /* Design matrix */
   gsl_vector_set_all(X_col, 1);
   gsl_matrix_set_col(design_X, 0, X_col);
   
   for( i = 0; i < X-> size2; i++)
     {
       gsl_matrix_get_col(X_col,  X, i);
       gsl_matrix_set_col(design_X, i+1, X_col );
     }
   
 
   /* Self-correction : 1. Logistic regression allows d=1 only 
                        2. The values in response variables must be 1 or 0
   */

   if(!strcmp(hmm1.modelname, "singleMVN_logistic") | !strcmp(hmm1.modelname, "single_logistic")){
     if(Y->size2 > 1){
       printf(" The number of variables in Y is larger than 1. This program allows a single variable in Y at this point. Proceeding the program using the first variable in Y. \n");
       Y_col = gsl_vector_alloc(Y->size1);
       gsl_matrix_get_col(Y_col, Y, 0);
       Y = gsl_matrix_alloc(Y_col->size, 1);
       for(i = 0; i< Y->size1; i++){
	 tmp = gsl_vector_get(Y_col, i);
	 gsl_matrix_set(Y, i, 0, tmp);
       }
     }
     for(i = 0; i< Y->size1; i++){
       tmp = gsl_matrix_get(Y, i, 0);
       if((tmp!=(double) 0) & (tmp!= (double) 1)){
	 printf(" Y must be either 0 or 1. \n");
	 exit(1);
       }
     }
     
   }


   /* Run Baum-Welch: 8 models */

   if(!strcmp(hmm1.modelname, "singleMVN")){
         
       total_rep = 0; 
       ERROR_IND[0]=0;
       T = ini.rowX;  

       /* Begin iteration */
       for(itr = 1 ; itr <= ini.itr; itr++){  

       	      /* Initial values for mu and Sigma */
	     InitialA(&hmm1, &ini);
	     InitialParm(&parm_hmm1, X, &ini);
	     if(parm_hmm1.mu_user_para){
	       for(i=1; i<= hmm1.N;i++){
		 for(j=1; j<=ini.colX;j++){
		   parm_hmm1.mu[i][j] = ini.mu[i][j];
		 }
	       }
	     }
  
	     for( i = 1; i <=hmm1.N ; i++){
	       mvdnorm_matrix(X, parm_hmm1.mu[i], parm_hmm1.Sigma[i], i, &hmm1, ERROR_IND);
	     }
	       
	     
	     LogMaxScaleB(&hmm1);
	     
	   
	   /* Baum - Welch Algorithm */   
  
	     BaumWelch_itr(itr, &hmm1, &parm_hmm1, &ini,  X, Y, T,  alpha, beta, gamma,  &niter,  MaxItr,  Tolerance,  logprobprevItr, ERROR_IND);
	     if(ERROR_IND[0]) itr = itr-1;
	     if(total_rep > ini.itr * 10){
	       printf(" The total number of trials reached %d.\n", ini.itr *10);
	       exit(1);
	     }
	 
       }

   }


   if(!strcmp(hmm1.modelname, "multiMVN"))
    {
       /* Allocate Memgory */
	parm_hmm1.c = dmatrix(1, hmm1.N, 1, hmm1.mix_comp);
	hmm1.B1 =  AllocArray(hmm1.N, hmm1.M, hmm1.mix_comp);
	    
	total_rep = 0; 
	ERROR_IND[0]=0;
	T = ini.rowX;  

       for(itr = 1 ; itr <= ini.itr; itr++){  
	 total_rep = total_rep + 1;

	   InitialA(&hmm1, &ini);
	   InitialC(&parm_hmm1);
	   InitialMultiMVN(X, &parm_hmm1, &hmm1, &ini);
	   LogMaxScaleB1(&hmm1, ERROR_IND);
	   CalculateB_from_B1(&hmm1, &parm_hmm1);
	 	 
	        
	 BaumWelch_itr(itr, &hmm1, &parm_hmm1, &ini,  X, Y, T,  alpha, beta, gamma,  &niter,  MaxItr,  Tolerance, logprobprevItr, ERROR_IND);
	 if(ERROR_IND[0]) itr = itr-1;

	 if(total_rep > ini.itr * 10){
	   printf(" The total number of trails reached %d.\n", ini.itr *10);
	   exit(1);
	 }
       }
	 
    }


   if(!strcmp(hmm1.modelname, "singleMVN_regression")){

       /* Initialize Weighted mean and variance */   
  
       parm_hmm1.reg_coef = dmatrix(1,  hmm1.N *parm_hmm1.d, 1, ini.colX + 1);
       parm_hmm1.reg_cov = AllocXi(hmm1.N, ini.colY);
       parm_hmm1.c = dmatrix(1, hmm1.N, 1, hmm1.mix_comp);
       
       

       /* Begin Iteration */
       total_rep = 0;
       T = ini.rowX;  
       for (itr = 1 ; itr <= ini.itr; itr++){  
	 total_rep = total_rep + 1;
	 if(total_rep > ini.itr * 10){
	   printf(" The total number of trials reached %d.\n", ini.itr *10);
	   exit(1);
	 }
	 
	   /* Initial Values */
	   InitialA(&hmm1, &ini);
	   InitialParm(&parm_hmm1, X, &ini);
	
	   
	   if(parm_hmm1.mu_user_para){
	     for(i=1; i<= hmm1.N;i++){
	       for(j=1; j<=ini.colX;j++){
		 parm_hmm1.mu[i][j] = ini.mu[i][j];
	       }
	     }
	   }
	     for( i = 1; i <=hmm1.N; i++){
	       mvdnorm_matrix(X, parm_hmm1.mu[i], parm_hmm1.Sigma[i], i, &hmm1, ERROR_IND);
	     }
	   
	   
	   InitialsingleRegression(X, Y, &parm_hmm1, &hmm1, &ini);
	
	   LogMaxScaleB(&hmm1);

	
	 
	 /* Baum - Welch Algorithm */   
  
	 BaumWelch_itr(itr, &hmm1, &parm_hmm1, &ini,  X, Y, T,  alpha, beta, gamma,  &niter,  MaxItr,  Tolerance, logprobprevItr, ERROR_IND);
	 if(ERROR_IND[0]) itr = itr-1;
	 if(total_rep > ini.itr * 10){
	   printf(" The total number of trails reached %d.\n", ini.itr *10);
	   exit(1);
	 }

       }
   
   }

   if(!strcmp(hmm1.modelname, "multiMVN_regression_state")){
     /* Allocate Memory */
     parm_hmm1.reg_coef = dmatrix(1, hmm1.N * parm_hmm1.d, 1, ini.colX + 1);
     parm_hmm1.reg_cov = AllocXi(hmm1.N, ini.colY);
     hmm1.B1 =  AllocArray(hmm1.N, hmm1.M, hmm1.mix_comp);
     parm_hmm1.c = dmatrix(1, hmm1.N, 1, hmm1.mix_comp);
     
     total_rep = 0; 
     ERROR_IND[0]=0; 
     T = ini.rowX;  
   
      for (itr = 1 ; itr <= ini.itr; itr++){  
	ERROR_IND[0]=0;
	total_rep = total_rep + 1;
	InitialA(&hmm1, &ini);
	InitialC(&parm_hmm1);
	InitialMultiMVN(X, &parm_hmm1, &hmm1, &ini); 
	InitialsingleRegression(X, Y, &parm_hmm1, &hmm1, &ini);
	   
	LogMaxScaleB1(&hmm1, ERROR_IND);
	CalculateB_from_B1(&hmm1, &parm_hmm1);
	
	/* Baum - Welch Algorithm */   
	
	BaumWelch_itr(itr, &hmm1, &parm_hmm1, &ini,  X, Y, T,  alpha, beta, gamma,  &niter,  MaxItr,  Tolerance,logprobprevItr, ERROR_IND);

	
       if(ERROR_IND[0]) itr = itr-1;

       if(total_rep > ini.itr * 10){
	 printf(" The total number of trails reached %d.\n", ini.itr *10);
	 exit(1);
       }
      }
     
   }
   

   if(!strcmp(hmm1.modelname, "singleMVN_logistic")){

       /* Initialize Weighted mean and variance */   
       parm_hmm1.reg_coef = dmatrix(1,  hmm1.N *parm_hmm1.d, 1, ini.colX + 1);
       parm_hmm1.c = dmatrix(1, hmm1.N, 1, hmm1.mix_comp);
              

       /* Begin Iteration */
       total_rep = 0;
       ERROR_IND[0]=0;
       T = ini.rowX;  

       for (itr = 1 ; itr <= ini.itr; itr++){  
	 total_rep = total_rep + 1;
	
	 /* Initial Values */
	 InitialA(&hmm1, &ini);
	 InitialParm(&parm_hmm1, X, &ini);
	 if(parm_hmm1.mu_user_para){
	   for(i=1; i<= hmm1.N;i++){
	     for(j=1; j<=ini.colX;j++){
	       parm_hmm1.mu[i][j] = ini.mu[i][j];
	     }
	   }
	 }
	 
	   gsl_matrix_set_all(weight_m, 1);
 

       for( i = 1; i <=hmm1.N; i++){
	 mvdnorm_matrix(X, parm_hmm1.mu[i], parm_hmm1.Sigma[i], i, &hmm1, ERROR_IND);
       }
       
       Initial_logistic(X, Y, &parm_hmm1, &hmm1, &ini); 
      

       /* Baum - Welch Algorithm */   
  
       BaumWelch_itr(itr, &hmm1, &parm_hmm1, &ini,  X, Y, T,  alpha, beta, gamma,  &niter,  MaxItr,  Tolerance, logprobprevItr, ERROR_IND);
       if(ERROR_IND[0]) itr = itr-1;

       if(total_rep > ini.itr * 10){
	 printf(" The total number of trails reached %d.\n", ini.itr *10);
	 exit(1);
       }

	 
       }
   }


   if(!strcmp(hmm1.modelname, "multiMVN_regression"))
     {

       /* Allocate Memory */
       parm_hmm1.reg_coef = dmatrix(1, hmm1.N *hmm1.mix_comp*parm_hmm1.d, 1, (ini.colX + 1));
       parm_hmm1.reg_cov = AllocXi((hmm1.N * hmm1.mix_comp), ini.colY);
       hmm1.B1 =  AllocArray(hmm1.N, hmm1.M, hmm1.mix_comp);
       parm_hmm1.c = dmatrix(1, hmm1.N, 1, hmm1.mix_comp);
      
      total_rep = 0; 
      ERROR_IND[0]=0;
      T = ini.rowX;  
   
      for (itr = 1 ; itr <= ini.itr; itr++){  
	total_rep = total_rep + 1;
	InitialA(&hmm1, &ini);
	InitialC(&parm_hmm1);
	InitialMultiMVN(X, &parm_hmm1, &hmm1, &ini); 
      
	InitialsingleRegression(X, Y, &parm_hmm1, &hmm1, &ini);

 
      	   
	LogMaxScaleB1(&hmm1, ERROR_IND);
	CalculateB_from_B1(&hmm1, &parm_hmm1);


	if(0)printf("AFTER InitialMultiMVN_regression in main()\n");

	/* Baum - Welch Algorithm */   
	
	BaumWelch_itr(itr, &hmm1, &parm_hmm1, &ini,  X, Y, T,  alpha, beta, gamma,  &niter,  MaxItr,  Tolerance, logprobprevItr, ERROR_IND);
       if(ERROR_IND[0]) itr = itr-1;

       if(total_rep > ini.itr * 10){
	 printf(" The total number of trials reached %d.\n", ini.itr *10);
	 exit(1);
       }
      }
   

     }

   
 if(!strcmp(hmm1.modelname, "multiMVN_logistic_state")){
   printf("I am afraid to tell you that the model is nor ready yet.\n");
   exit(1);
 }



   if(!strcmp(hmm1.modelname, "single_regression")){
     
     /* Allocate Memory */

       parm_hmm1.reg_coef = dmatrix(1, hmm1.N * parm_hmm1.d, 1, ini.colX + 1);
       parm_hmm1.reg_cov = AllocXi(hmm1.N, ini.colY);
     
     if(parm_hmm1.reg_coef_user_num > 0){
       parm_hmm1.reg_coef_fixed_parm = imatrix(1, hmm1.N * parm_hmm1.reg_coef_user_num, 1, ini.colX +1);
       k=0;
       for(i=1; i<= parm_hmm1.reg_coef_user_num; i++){
	 for(j = 1; j<= (parm_hmm1.p+1); j++){
	   parm_hmm1.reg_coef_fixed_parm[i][j]=ini.reg_coef_fixed_parm[k];
	   if(ini.reg_coef_fixed_parm[i][j]!=0){
	     parm_hmm1.reg_coef[i][j]= ini.reg_coef[i][j];
	     /* printf("%f \t", parm_hmm1.reg_coef[i][j]); */
	   }
	   k++;
	 }
       }
     }
     
     
     
     
     total_rep = 0; 
     ERROR_IND[0]=0; 
     T = ini.rowX;  
     
      for (itr = 1 ; itr <= ini.itr; itr++){  
	total_rep = total_rep + 1;
	InitialA(&hmm1, &ini);
     	InitialsingleRegression(X, Y, &parm_hmm1, &hmm1, &ini); 

   
	LogMaxScaleB(&hmm1);
	
	/* Baum - Welch Algorithm */   
	ERROR_IND[0]=0;
	BaumWelch_itr(itr, &hmm1, &parm_hmm1, &ini,  X, Y, T,  alpha, beta, gamma,  &niter,  MaxItr,  Tolerance, logprobprevItr,  ERROR_IND);
	
       if(ERROR_IND[0]) itr = itr-1;

       if(total_rep > ini.itr * 10){
	 printf(" The total number of trails reached %d.\n", ini.itr *10);
	 exit(1);
       }
      }
     
   }
   
   if(!strcmp(hmm1.modelname, "single_logistic")){
     /* Allocate Memory */
     parm_hmm1.reg_coef = dmatrix(1, hmm1.N * parm_hmm1.d, 1, ini.colX + 1);

     
     total_rep = 0; 
     ERROR_IND[0]=0; 
     T = ini.rowX  ;

      for (itr = 1 ; itr <= ini.itr; itr++){  
	total_rep = total_rep + 1;
	InitialA(&hmm1, &ini);
     	InitialsingleLogistic(X, Y, &parm_hmm1, &hmm1, &ini); 

	/* Baum - Welch Algorithm */   
	ERROR_IND[0]=0;
	BaumWelch_itr(itr, &hmm1, &parm_hmm1, &ini,  X, Y, T,  alpha, beta, gamma,  &niter,  MaxItr,  Tolerance, logprobprevItr, ERROR_IND);

	
       if(ERROR_IND[0]) itr = itr-1;

       if(total_rep > ini.itr * 10){
	 printf(" The total number of trials reached %d.\n", ini.itr *10);
	 exit(1);
       }
      }
     
   }
   


   gsl_matrix_free(X);
   gsl_matrix_free(Y);
   gsl_matrix_free(design_X);

   free_dmatrix(alpha,1, hmm1.M, 1, hmm1.N);
   free_dmatrix(beta, 1, hmm1.M, 1, hmm1.N);
   free_dmatrix(gamma, 1, hmm1.M, 1, hmm1.N);  
     

 return(0);

}

double myround(double value, int prec)
{
  double x, temp;  
  temp = value * pow(10, prec);
  x = round(temp)/ pow(10, prec);
  return(x);
}
