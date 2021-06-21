/* This file includes functions: 
   Functions marked with * are not used in MRHMMs.

   read_initial : Read input file
   PrintInitial : Print initial values
   WeightedMean: Calculate weighted mean
   WeightedMean_gsl: Calculate weighted mean (inputs are gsl vector)
   WeightedCov_gsl: Calculate covariance of two gsl vectors
   WeightedVar_gsl: Calculate variance and covariance of gsl matrix 
   InitialA: Generate random intitial trnasition probability matrix A
   InitialParm: Generate initial mean and variance at random
   WeightedMean_gsl_matrix: Cacluate weighted mean of 
                            gsl_matrix -weight is matrix
   WeightedVar_gsl_matrix: Calculate weighted var of gsl_matrix 
                          - weight is matrix
   CalculateB_from_B1: Calculate emission probability of mixture model
   CalculateB_from_B1_Log: log(Calculate emission probability of mixture model)
   gsl_vector_sum: Calculate sum(gsl_vector)
   gsl_vector_print: Print gsl_vector
   gsl_matrix_print: Print gsl_matrix
   dvector_to_gsl_vector: Change format from dvector to gsl_vector
   dmatrix_to_gsl_matrix: Change format from dmatrix to gsl_matrix
   gsl_vector_to_dvector: Change format from gsl_vector to dvector (double)
   gsl_vector_to_ivector: Change format from gsl_vector to ivector (integer)


   
    
   *dReadMatrix : Read file (double) matrix 
   *dReadVector : Read file (double) vector
   *InitialSeq: Generate random initial sequence
   *RandomSeq: Generate random sequence
   *InitialWeight_gsl_m: hard weight (1 or 0) for a given state q
   *Initial_Uniform: assigns the observation probabilities in B at random
   *InitialParmSampleX: Alternative way to generate initial values
   *WeightedSum_B_multiMVN: Calculate emission probability of mixture model
   * ReadInitialParm: Read user-provided initial values for larger dimension
 
*/


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>


#include "nrutil.h"
#include "HMM.Read.Matrix.h"
#include "hmm.h"

int read_initial(char *int_location, INI *ini,  HMM *hmm1, PARM_HMM *hmm1_para)
{
  FILE *int_file;
  FILE *ini_state_file;
  char *variable;
  char *variable1;
  char *variable2;
  char *variable3;
  char num_state_string[100];
  char *pbs_string;
  char *temp_char;
  int i,j, k=1, count_k=1,  ini_count=0, ini_count_check[6], PBS_ind = 0;
  int check[14];
  char **check_list, **ini_check_list;

  check_list = malloc(14*sizeof(char *));
  check_list[0] = "Data.X.file.location";
  check_list[1] = "Data.Y.file.location";
  check_list[2] = "num.data.sets";
  check_list[3] = "dim.data.sets";
  check_list[4] = "num.state";
  check_list[5] = "model.name";
  check_list[6] = "mix.comp";
  check_list[7] = "Obs.loc";
  check_list[8] = "output.filename";
  check_list[9] = "Rep";
  check_list[10] = "Max.Itr";
  check_list[11] = "Tolerance";
  check_list[12] = "INI.VALUES";
  check_list[13] = "USER.PARAMETERS";


  ini_check_list = malloc(6 * sizeof(char *));
  ini_check_list[0] = "ini.pi";
  ini_check_list[1] = "ini.A";
  ini_check_list[2] = "ini.mu";
  ini_check_list[3] = "ini.Sigma";
  ini_check_list[4] = "ini.reg.coef";
  ini_check_list[5] = "ini.reg.sd";
  /* ini_check_list[6] = "ini.state";*/

  hmm1_para-> reg_coef_user_num = 0;
  hmm1_para-> reg_cov_user_para = 0;
  hmm1_para-> mu_user_para = 0;
  hmm1_para-> Sigma_user_para = 0;
  
  for(i=0; i<13; i++){
    /* printf("%s\n", check_list[i]); */
    check[i]=0;
  }
  /* No need for num.data.sets and dim.data.sets Jan10.2013*/
  check[2]=1;
  check[3]=1;

  variable = malloc(100 * sizeof(char));
  variable1 = malloc(100 * sizeof(char));
  variable2 = malloc(100 * sizeof(char));
  variable3 = malloc(100 * sizeof(char));
  temp_char = malloc(1 *sizeof(char));
  ini->num_data_sets = 0;
    
  if((int_file=fopen(int_location, "r"))==NULL){
    printf("ERROR: Cannot open initial value file.\n");
    exit(1);
  }
  while(!feof(int_file)){
    fscanf(int_file, "%s", variable);
    
    if(!strcmp(variable, "Data.X.file.location"))
      { 
	fscanf(int_file, "%s", ini-> loc_Data_X);
	fscanf(int_file, "%d", & (ini -> rowX));
        fscanf(int_file, "%d", & (ini -> colX));
	hmm1-> M  = ini -> rowX;
	check[0]=1;
      }
   
    if(!strcmp(variable, "Data.Y.file.location"))
      { 
	fscanf(int_file, "%s", ini-> loc_Data_Y);
	fscanf(int_file, "%d", &(ini -> rowY));
        fscanf(int_file, "%d", &(ini -> colY));
	check[1]=1;
      }
    
    if(!strcmp(variable, "num.data.sets"))
      { 

	fscanf(int_file, "%d", &(ini -> num_data_sets));
	check[2]=1;
      }	
      
    /* Not used in MRHMMs */
    if(!strcmp(variable, "dim.data.sets"))
      { 
	check[3]=1;
	if(ini -> num_data_sets == 0){
	  printf("Invalid number of data sets: 0.\n");
	  printf("Move num_data_sets to above the dim.data.sets in the initial file. \n");
	}else {
	  for(i = 0; i< (ini->num_data_sets) ; i++)
	    {
	      fscanf(int_file, "%d", &ini->dim_data_sets[i]);
	    }
	}
      }

    /* Not used in MRHMMs */
    if(!strcmp(variable, "num.state"))
      {
	check[4]=1;
	fscanf(int_file, "%s", num_state_string);
	if(!strcmp("PBS", num_state_string)){
	  PBS_ind = 1;
	  if((pbs_string = getenv("PBS_ARRAYID")))
	    {
	      printf("The number of state is %s \n", pbs_string );
	      hmm1->N = atoi(pbs_string);
	    }else 
	    {
	      printf("PBS_ARRAYID is not defined.\n");
	      exit(1);
	    }
	} else {
	  hmm1->N = atoi(num_state_string);
	} 
      }
    
   
    if(!strcmp(variable, "model.name")) 
      {
	check[5]=1;
	fscanf(int_file, "%s", hmm1-> modelname);
	
	/* Check if the model name is valid */
	
	if(strcmp(hmm1->modelname, "singleMVN") & strcmp(hmm1->modelname, "multiMVN") & strcmp(hmm1->modelname, "singleMVN_regression") & strcmp(hmm1->modelname, "singleMVN_logistic") & strcmp(hmm1->modelname, "multiMVN_regression") & strcmp(hmm1->modelname, "multiMVN_regression_state")& strcmp(hmm1->modelname, "multiMVN_logistic_state") & strcmp(hmm1->modelname, "single_regression") & strcmp(hmm1->modelname, "single_logistic")  ){
	  printf(" The model name is not valid. \n");
	  exit(1);
	}
	
      }
      
  
  
    if(!strcmp(variable, "mix.comp"))
      {
	check[6]=1;
	fscanf(int_file, "%s", num_state_string);
	if(!strcmp("PBS", num_state_string)){
	  PBS_ind = 1;
	  if((pbs_string = getenv("PBS_ARRAYID")))
	    {
	      printf("The number of mixture components is %s \n", pbs_string );
	      hmm1-> mix_comp = atoi(pbs_string);
	    }else 
	    {
	      printf("PBS_ARRAYID is not defined.\n");
	      exit(1);
	    }
	} else {
	  hmm1-> mix_comp = atoi(num_state_string);
	} 

	if(!strcmp(hmm1->modelname, "singleMVN") | !strcmp(hmm1->modelname, "singleMVN_regression") | !strcmp(hmm1->modelname, "single_regression") | !strcmp(hmm1->modelname, "single_logistic")| !strcmp(hmm1->modelname, "singleMVN_logistic")){
	  if(hmm1->mix_comp > 1) 
	    printf("The number of mixture components is set to 1 based on the model name you provided.\n");
	  hmm1->mix_comp = 1;

	  
	}
	
      }
    /* Note : mix_comp AFTER model.name */

    if(!strcmp(variable, "Obs.loc")) /* Window.dist -> Obs.loc */
      {
	check[7]=1;
	fscanf(int_file, "%s", temp_char);
	if(!strcmp(temp_char,"Y"))
	  {
	    hmm1->window_dist = 1;
	    fscanf(int_file, "%d", &( hmm1->D1) );
	    fscanf(int_file, "%s", ini->loc_window_dist);
	  } else
	  { 
	    hmm1->window_dist =0;
	  }

	if(hmm1->D1 <0){
	  printf("Error: Invalid value: D must be nonnegative(Line Obs.loc)\n");
	  exit(1);
	}
      }
    if(!strcmp(variable, "USER.PARAMETERS"))
      {
	check[13]=1;
	fscanf(int_file, "%s", temp_char);
	if(!strcmp(temp_char,"Y"))
	  {
	    hmm1_para-> user_para = 1;
	    while(strcmp(variable, "END.USER.PARAMETERS")){
	    
	      fscanf(int_file, "%s", variable);

	      if(!strcmp(variable, "reg.coef")){
		fscanf(int_file, "%s", temp_char);
		if(!strcmp(temp_char,"Y")){
		  fscanf(int_file, "%d", &( hmm1_para-> reg_coef_user_num) );
		  ini-> reg_coef_fixed_parm = 
		    imatrix(1, hmm1_para->reg_coef_user_num, 1, ini->colX +1);
		  hmm1_para-> reg_coef_fixed_parm = 
		    imatrix(1, hmm1_para->reg_coef_user_num, 1, ini->colX +1);
		  
		  ini-> reg_coef = 
		    dmatrix(1, hmm1_para->reg_coef_user_num, 1, ini->colX + 1);
		  k=0;
		  for(i=1; i<= hmm1_para-> reg_coef_user_num;i++){
		    for(j=1; j<= (ini-> colX +1);j++){
		      fscanf(int_file, "%s", variable);
		      if(!strcmp(variable,"u")){
			ini-> reg_coef_fixed_parm[i][j]=0;
		      } else {
			ini-> reg_coef_fixed_parm[i][j]=1;
			ini-> reg_coef[i][j]=atof(variable);
		      }
		      k++;
		    }
		  }
		} else hmm1_para-> reg_coef_user_num = 0;
	      }
	      
	      if(!strcmp(variable, "mu")){
		fscanf(int_file, "%s", temp_char);
		if(!strcmp(temp_char,"Y")){
		  hmm1_para-> mu_user_para =1;
		  ini-> mu = 
		    dmatrix(1,hmm1_para->N * hmm1_para->mix_comp, 1, ini->colX);
		  k=0;
		  for(i=1; i<=hmm1_para->N * hmm1_para->mix_comp;i++){
		    for(j=1; j<=ini->colX;j++){
		      fscanf(int_file, "%lf", &(ini->mu[i][j]) );
		    }
		  }
		}
	      }	      

	      
	    } /* End of while */
	  } /* End of "Y" to USER.PARAMETERS */
	else
	  { 
	    hmm1_para->user_para = 0;
	  }

      }
   
    if(!strcmp(variable, "output.filename") & check[8]==0)
      {
	check[8]=1;
	fscanf(int_file, "%s", ini-> output_filename);
	if(hmm1-> mix_comp >1){
	sprintf(ini->output_filename, "%s%s%d%s%d", ini->output_filename, "_state", hmm1->N, "mix", hmm1->mix_comp);
	}
	if(hmm1-> mix_comp == 1){
	  sprintf(ini->output_filename, "%s%s%d", ini->output_filename, "_state", hmm1->N);
	}

   
      }
   

    if(!strcmp(variable, "Rep"))
      {
	check[9]=1;
	fscanf(int_file, "%d", &(ini -> itr));
      }
    if(!strcmp(variable, "Max.Itr"))
      {
	check[10]=1;
	fscanf(int_file, "%d", &(ini -> MaxItr));
      }

    if(!strcmp(variable, "Tolerance"))
      {
	check[11]=1;
	fscanf(int_file, "%lf", &(ini -> Tolerance));
      }
    
    
    /*
      if(!strcmp(variable, "ini.seq"))
	{
	  fscanf(int_file, "%s", ini-> initial_q_loc);
	  if(1){ if(PBS_ind){
	    StringPaste(ini->initial_q_loc, ini->initial_q_loc,
			getenv("PBS_ARRAYID"));
	    }
	  }
	}
    */
      if(!strcmp(variable, "INI.VALUES")){
	check[12]=1;
	fscanf(int_file, "%s", variable1);
	for(i = 0; i< 8; i++){
	  ini-> fixed_ini_indicator[i]= 0;
	}
	if(!strcmp(variable1, "Y")){

	  /* HMM Initial values */
	  ini_count=0;

	  ini-> fixed_ini_indicator[7]= 0; /* ??? */
	  while(strcmp(variable,"END.INI.VALUES")){
	  
	    fscanf(int_file, "%s", variable);
	    /* printf("variable = %s\n", variable);*/
	    if(!strcmp(variable, "ini.pi"))
	      {
		ini_count++;
		ini_count_check[0]=1;
	
		fscanf(int_file, "%s", temp_char);
		if(!strcmp(temp_char,"Y")){
		  ini->fixed_ini_indicator[0]=1;
		  ini -> ini_pi = (double *) dvector(1, hmm1->N);
		  for (i = 1; i <= hmm1->N; i++) 
		  fscanf(int_file, "%lf", &(ini->ini_pi[i])); 
		} else   hmm1->pi = dvector(1, hmm1->N);
	
	      }
	    
	    
	  
	    if(!strcmp(variable, "ini.A"))
	      {
		ini_count++;
		ini_count_check[1]=1;
		
		fscanf(int_file, "%s", temp_char);
		if(!strcmp(temp_char,"Y")){
		  ini->fixed_ini_indicator[1]=1;
		  ini -> ini_A = (double **) dmatrix(1, hmm1->N, 1, hmm1->N);
		  for (i = 1; i <= hmm1-> N; i++) { 
		    for (j = 1; j <= hmm1-> N; j++) {
		      fscanf(int_file, "%lf", &(ini -> ini_A[i][j])); 
		    }
		    /* fscanf(int_file,"\n");*/
		  }
		} else hmm1->A = dmatrix(1, hmm1->N, 1, hmm1->N);

	      }
	    
	    if(!strcmp(variable, "ini.mu"))
	      {
		ini_count++;
		ini_count_check[2]=1;
		
		fscanf(int_file, "%s", temp_char);
		if(!strcmp(temp_char,"Y")){
		  ini->fixed_ini_indicator[2]=1;
		  fscanf(int_file, "\n");
		  ini->ini_mu = (double **) dmatrix(1, hmm1->N * hmm1->mix_comp , 1, ini->colX);
		  for (i = 1; i <= (hmm1-> N * hmm1->mix_comp); i++) { 
		    for (j = 1; j <= (ini->colX); j++) {
		      fscanf(int_file, "%lf", &(ini->ini_mu[i][j])); 
		    }
		    fscanf(int_file,"\n");
		  }
		}
	      }
	    
	    

	    if(!strcmp(variable, "ini.Sigma"))
	      {
		ini_count++;
		ini_count_check[3]=1;

	
		fscanf(int_file, "%s", temp_char);
		if(!strcmp(temp_char,"Y")){
		  ini->fixed_ini_indicator[3]=1;
		  fscanf(int_file, "\n");
		  ini->ini_Sigma = AllocXi(hmm1->M * hmm1 -> mix_comp , ini->colX);
		  for (i = 1; i <= (hmm1-> N * hmm1->mix_comp); i++) { 
		    for (j = 1; j <= (ini->colX); j++) {
		      for(k = 1; k <= (ini -> colX); k++){
			fscanf(int_file, "%lf", &(ini->ini_Sigma[i][j][k])); 
		      }
		    }
		    fscanf(int_file,"\n");
		  }
		}
	      }
	    
	    if(!strcmp(variable, "ini.reg.coef"))
	      {
		
		ini_count++;
		ini_count_check[4]=1;
	
		fscanf(int_file, "%s", temp_char);
		if(!strcmp(temp_char,"Y")){
		  ini->fixed_ini_indicator[4]=1;
		  fscanf(int_file, "\n");
		  if(strcmp(hmm1->modelname, "multiMVN_regression_state")){
		      ini->ini_reg_coef = dmatrix(1, hmm1->N * hmm1->mix_comp*ini->colY, 1, ini->colX);
		      for (i = 1; i <= (hmm1-> N * hmm1->mix_comp * ini->colY ); i++) { 			
			for (j = 1; j <= (ini->colX +1); j++) {
			  fscanf(int_file, "%lf", &(ini->ini_reg_coef[i][j])); 
			}
			fscanf(int_file,"\n");
		      }
		  } else{
		    ini->ini_reg_coef = dmatrix(1, hmm1->N * ini-> colY, 1, ini->colX + 1);
		    for (i = 1; i <= (hmm1-> N * ini->colY); i++) { 
		      for (j = 1; j <= (ini->colX +1); j++) {
			fscanf(int_file, "%lf", &(ini->ini_reg_coef[i][j])); 
		      }
		      fscanf(int_file,"\n");
		    }
		  }
		}
	      }
 
	    if(!strcmp(variable, "ini.reg.sd"))
	      {
		ini_count++;
		ini_count_check[5]=1;

	
		fscanf(int_file, "%s", temp_char);
		if(!strcmp(temp_char,"Y")){
		  ini->fixed_ini_indicator[5]=1;
		  fscanf(int_file, "\n");
		  if(strcmp(hmm1->modelname, "multiMVN_regression_state")){
		    ini->ini_reg_cov = AllocXi(hmm1->N * hmm1->mix_comp, ini->colY); 
		      for (i = 1; i <= (hmm1-> N * hmm1->mix_comp); i++) { 
			for (j = 1; j <= (ini->colY); j++) {
			  for(k = 1; k <= (ini-> colY); k++){
			    fscanf(int_file, "%lf", &(ini->ini_reg_cov[i][j][k])); 
			  }
			}
			 fscanf(int_file,"\n");
		      }
		  } else{
		    ini->ini_reg_cov = AllocXi(hmm1->N, ini->colY); 
		    for (i = 1; i <= (hmm1-> N); i++) { 
		      for (j = 1; j <= (ini->colY); j++) {
			for(k = 1; k <= (ini-> colY); k++){
			  fscanf(int_file, "%lf", &(ini->ini_reg_cov[i][j][k])); 			  
			}
		      }
		       fscanf(int_file,"\n"); 
		    }
		  }
		}
	      }
	    /* 
	    if(!strcmp(variable, "ini.state"))
	      {
		ini_count++;
		ini_count_check[6]=1;

	
		fscanf(int_file, "%s", temp_char);  
		if(!strcmp(temp_char,"Y")){
		  ini->fixed_ini_indicator[6]=1;
		  fscanf(int_file, "%s", variable);
		  if((ini_state_file  = fopen(variable,"r"))==NULL){
		    printf("ERROR: Cannot open ini.seq file.\n");
		    exit(1);
		  }
		  ini-> ini_state = malloc((hmm1->M+1) * sizeof(int));
		  for( i = 1; i<= (hmm1->M); i++)
		    {
		      fscanf(ini_state_file, "%d", &(ini->ini_state[i])); 
		    }
		  fclose(ini_state_file);
		}
		
	      }
	    */
	  } /* End of while */
	  


	    /* Check if there is no initial parameters not provided */
	    if(ini_count < 6){
	      printf("ini_count = %d", ini_count);
	      for(i=0; i< 6; i++){
		if(ini_count_check[i]!=1){
		  printf("ERROR: %s in INI.VALUES is not provided. \n", ini_check_list[i]);
		}
	      }
	      exit(1);
	    }
	         
	} /* End of USER */
	

	if(!strcmp(variable1, "N")){
	  
	  hmm1->A = dmatrix(1, hmm1->N, 1, hmm1->N);
	  hmm1->pi = dvector(1, hmm1->N);
	  
	}
      }/********End of INI.VALUES *******/
      
    
  }
  fclose(int_file);


  for( i = 0; i<=13; i++){
    count_k *=count_k*check[i];
  }

  if(count_k!=1){
    for(i = 0; i<=13;i++){
      if(check[i]!=1){
	printf("ERROR: %s is not provided.\n", check_list[i]);
      }
    }
    exit(1);
  }

  

  free(variable);
  free(variable1);
  free(variable2);
  free(variable3);
    
  return(0);
  
}


void PrintInitial(INI *ini, HMM *hmm1)
{
   /* Summary of the model from user input */
  printf("X:  %s \t %d\t by\t %d \n", ini->loc_Data_X, ini->rowX, ini->colX);
  printf("Y:  %s \t %d\t by\t %d \n", ini->loc_Data_Y, ini->rowY, ini->colY);
  printf("Sample size: %d \n",  hmm1->M);
  
  printf("The model name is %s \n", hmm1->modelname);
  printf("Number of states: %d \n", hmm1->N);
  

  if( (!strcmp(hmm1->modelname, "multiMVN")) & ( hmm1->mix_comp <=1)){
    printf("\n");
    printf("Error!: The number of mixture components should be larger than or equal to 2.\n");
    exit(1);
  }
 if( (!strcmp(hmm1->modelname, "multiMVN_logistic")) & ( hmm1->mix_comp <=1)){
    printf("\n");
    printf("Error!: The number of mixture components should be larger than or equal to 2.\n");
    exit(1);
  }
  if( ((!strcmp(hmm1->modelname, "singleMVN"))) & ( hmm1->mix_comp > 1)){
    printf("\n");
    printf("Warning!: The number of mixture components should be 1. The value is set to 1. \n");
    hmm1->mix_comp = 1;
  }
  
  if( (!strcmp(hmm1->modelname, "singleMVN_logistic")) &(  hmm1->mix_comp > 1)){
    printf("\n");
    printf("Warning!: The number of mixture components should be 1. The value is set to 1. \n");
    hmm1->mix_comp = 1; 
  }

  printf("Number of mixture components is: %d \n", hmm1->mix_comp);
  printf("Total number of repetition = %d\n\n", ini->itr);

}


void dReadMatrix(char *file_location, int num_row, int num_col, double **mtx)
{
  FILE *matrix_file;
  char *data_value;
  int i, j;
 
  data_value=malloc(100*sizeof(char));
  printf("...Reading a matrix from is %s \n", file_location);
  if((matrix_file =fopen(file_location,"r"))==NULL){
    printf("ERROR: Cannot open matrix file. \n");
    exit(1);
  }
  
   
  for(i=1; i <= num_row ; i++){
    for(j=1 ; j <= num_col; j++){
      fscanf(matrix_file, "%lf", &mtx[i][j]);
    }
  }
  
  fclose(matrix_file);
  free(data_value);
}

void dReadVector(char *file_location, int num, double *vec)
{
  FILE *vector_file;
  char *data_value;
  int i;
 
  data_value=malloc(100*sizeof(char));
  printf("...Reading a vector from is %s \n", file_location);
  if((vector_file =fopen(file_location,"r"))==NULL){
    printf("ERROR: Cannot open matrix file. \n");
    exit(1);
  }
  
   
  for(i=1; i <= num ; i++){
    fscanf(vector_file, "%lf", &vec[i]);
  }
  
  fclose(vector_file);
  free(data_value);
}

void InitialSeq(HMM *hmm1, int *q)
{
  double val, accum = 0;
  int i , j;
  int r;
  int print_ind = 0;
  
  srand ( (unsigned)time ( NULL ) );
  r = rand();
  val = (double) r / ((double) RAND_MAX +1);
  if(print_ind) printf("val = %lf\n", val);

  
  
  for( i = 1; i <= hmm1-> N; i++)  
    {
      if ( val < hmm1->pi[i] + accum)
	{
	  q[1] = i;
	  break;
	}
      else accum += hmm1->pi[i];
    }
  
  for( i = 2; i <= hmm1 -> M; i++)
    {
      accum = 0;
      r = rand();
      val = (double) r / ((double) RAND_MAX +1);

      if(print_ind) printf("val = %lf \n", val);
      for( j = 1; j <= hmm1 -> N; j++){
	if( val < (hmm1 -> A[q[i-1]][j] + accum))
	  {
	    q[i]= j;
	    break;
	  }
	else accum +=hmm1 ->A[q[i-1]][j];
      }
    }
}

void RandomSeq(double *p, int len_p, int *q, int len_q)
{
  double val, accum = 0;
  int i, j;
  int r;
  int print_ind = 0;
 
  srand ( (unsigned)time ( NULL ) );
    for( i = 1; i <= len_q; i++)
    {
      accum = 0;
      r = rand();
      val = (double) r / ((double) RAND_MAX +1);

      if(print_ind) printf("val = %lf \n", val);
      for( j = 1; j <= len_p; j++){
	if( val < p[j] + accum)
	  {
	    q[i]= j;
	    break;
	  }
	else accum +=p[j];
      }
    }
}

double WeightedMean(double *data, double *weight, int len, int *ERROR_IND)
{
  int i;
  double sum_all=0, sum_weight=0;
  
  if(len <= 0)
    {
      printf("Error! \n The length is nonpositive (WeighedMean)\n");
      exit(1);
    }
  
  for (i=0 ; i< len; i++)
    {
      sum_all += data[i]*weight[i];
      sum_weight += weight[i];
    }
  
  if (sum_weight == 0)
    {
      printf("Error! \n The weights are all 0 (WeightedMean), \n");
      ERROR_IND[0]=1;
    }

  return(sum_all/sum_weight);
}

double WeightedMean_gsl(gsl_vector *data, gsl_vector *weight, int len, int *ERROR_IND)
{
  int i;
  double sum_all=0.0, sum_weight=0.0, weight_temp;
  
  
  if(len <= 0)
    {
      printf("Error! \n The length is not positive (WeighedMean_gsl)\n");
      exit(1);
    }
  for (i=0 ; i< len; i++){
    weight_temp = gsl_vector_get(weight, i);
    sum_all += gsl_vector_get(data, i) * weight_temp;
    sum_weight += weight_temp;
    
    if(weight_temp < 0){
      ERROR_IND[0] = 1;
      printf("Weight at time %d is negative.\n", i+1);
      return(0);
    } 
  }

 if(sum_weight == 0){
    printf("Error! \n The weights are all 0 (WeightedMean_gsl), \n");
    ERROR_IND[0]=1;
    return(0);
  }
 
   return(sum_all/sum_weight);
}



double WeightedCov_gsl(gsl_vector *data1, gsl_vector *data2, double mean1, double mean2,  gsl_vector *weight, int len, int *ERROR_IND)
{
  int i;
  double sum_all = 0, sum_weight = 0;

  for (i=0; i < len; i++){
    sum_all += (gsl_vector_get(data1, i) - mean1)* (gsl_vector_get(data2, i)-mean2) * gsl_vector_get(weight, i);
    sum_weight += gsl_vector_get(weight, i);
  }
  
 if(sum_weight == 0){
    printf("Warning: The weights are all 0 (WeightedCov_gsl), \n");
    ERROR_IND[0] = 1; 
    return(0);
  }

  return(sum_all/sum_weight);
}


void WeightedVar_gsl(gsl_matrix *Data, double *mean1,  gsl_vector *weight, int len, double num_col_data, double **Var, int *ERROR_IND)

{
  int i,j;
  gsl_vector *col_i = gsl_vector_alloc (len); 
  gsl_vector *col_j = gsl_vector_alloc (len);
  double min_var = .0001;
  
  for ( i=0; i < num_col_data; i++){
    gsl_matrix_get_col(col_i, Data, i);
    for ( j=0; j<=i; j++){
      gsl_matrix_get_col(col_j, Data, j );
      Var[i+1][j+1] = WeightedCov_gsl(col_i, col_j, mean1[i+1], mean1[j+1], weight, len, ERROR_IND);
      if(i==j){
	if(Var[i+1][i+1] < .0001)
	  Var[i+1][j+1] += min_var;
      }
    }
  }

  for( i = 0; i < num_col_data; i++){
    for( j = i+1; j < num_col_data; j++){
      Var[i+1][j+1] = Var[j+1][i+1];
    }
  }

  gsl_vector_free(col_i);
  gsl_vector_free(col_j);
}


void InitialWeight_gsl_m(int *q, gsl_matrix *weight_m, int print_ind)
{
  /* Initial Weight ( N by M) : 
     weight_m[t, j] = 1 if the q[t] = j and 0 otherwise. */ 

  int i, j;
  int M, N;

  M = weight_m -> size1;
  N = weight_m -> size2;
   for( i = 0; i < M ; i++)
     {
       for ( j = 0; j < N ; j++)
	 {
	   gsl_matrix_set (weight_m, i, j, (j+1) == q[i+1]? 1:0);
	   if(print_ind) printf("%lf \t", gsl_matrix_get(weight_m, i, j));
	 }
       if(print_ind) printf("\n");
     }
}


void Initial_Uniform(HMM *hmm1)
{
  /* The function assigns the observation probabilities in B randomly 
     from a Uniform dsitribution */

  int i, j, k, r;
  int print_ind = 0;
  
  hmm1->B= dmatrix(1, hmm1->M, 1, hmm1->N);
   srand ( (unsigned)time ( NULL ) );
  for( i = 0; i < hmm1->M; i++)
    {
      for( j = 0; j < hmm1->N; j++)
	{
	  r = rand();
	  hmm1->B[i+1][j+1] = (double) r/ ( (double) RAND_MAX + 1);
	  if(print_ind) printf("%lf \t", hmm1->B[i+1][j+1]);
	}
      if(print_ind) printf("\n");
    }
  
  if(!strcmp(hmm1-> modelname, "multiMVN")){
    hmm1-> B1 = AllocArray(hmm1-> N, hmm1-> M, hmm1->mix_comp);
      for( i = 0; i < hmm1-> N; i++){
	  for( j = 0; j < hmm1-> M; j++){
	      for( k = 0; k < hmm1-> mix_comp; k++){
		r = rand();
		hmm1->B1[i+1][j+1][k+1] =(double) r/ ( (double) RAND_MAX + 1);
	      }
	  }  
      }  
  }
  
}
 
void InitialA(HMM *hmm1, INI *ini)
{
  int i, j, r;
  double sum, sum_pi;

 

  srand ( (unsigned)time ( NULL ) );
  sum_pi = 0;

  if(ini-> fixed_ini_indicator[0]!=1){ 
    for( i = 0; i < hmm1-> N; i++){
      r = rand();
      hmm1->pi[i+1] = (double) r/ ( (double) RAND_MAX + 1);
      sum_pi += hmm1-> pi[i+1];
    }
    for( i = 0; i < hmm1 -> N; i++) {
      hmm1->pi[i+1]/=sum_pi;
    }
  } else{
    hmm1-> pi = dvector(1, hmm1->N);
    for( i = 0 ; i < hmm1 -> N; i++) {
      hmm1->pi[i+1] = ini->ini_pi[i+1];
    }
  }
 
  if(ini-> fixed_ini_indicator[1]!=1){ 
    for( i = 0; i < hmm1-> N; i++){
      sum = 0;
      for( j = 0; j < hmm1-> N; j++)
	{
	  r = rand();
	  hmm1->A[i+1][j+1] =(double) r/ ( (double) RAND_MAX + 1);
	  
	  sum += hmm1->A[i+1][j+1];
	}
      for( j = 0; j < hmm1-> N; j++)
	{
	  hmm1-> A[i+1][j+1] /= sum;
	}
      
    }
  } else {
    hmm1->A = dmatrix(1, hmm1->N, 1, hmm1->N);
    for( i = 0; i < hmm1-> N; i++){
      for( j = 0; j < hmm1-> N; j++){
	hmm1->A[i+1][j+1] = ini->ini_A[i+1][j+1];
      }
    }
  }

    
   
}


  
  

   
void InitialC(PARM_HMM *parm_hmm1)
{
  int i, j, r;
  double sum;
  
  /* Random  
  for( i = 0; i < parm_hmm1-> N; i++){
    sum=0;
          for( j = 0; j < parm_hmm1-> mix_comp; j++)
	    {
		r = rand();
		parm_hmm1->c[i+1][j+1] =(double) r/ ( (double) RAND_MAX + 1);
		sum += parm_hmm1->c[i+1][j+1];
	    }
	  for( j = 0; j < parm_hmm1-> mix_comp; j++)
	    {
	      parm_hmm1->c[i+1][j+1] /= sum;
	    }
	 
  } 
  */
  /* Fixed */
   for( i = 0; i < parm_hmm1-> N; i++){
     for( j = 0; j < parm_hmm1-> mix_comp; j++)
       parm_hmm1->c[i+1][j+1] = (double) 1/ (parm_hmm1->mix_comp);
   }

}  
  
void InitialParm(PARM_HMM *parm_hmm1, gsl_matrix *X, INI *ini)
{
  /* Find the random mean and the sample variance */
  int i,j,k;
  gsl_vector *w = gsl_vector_alloc(X->size1);
  gsl_vector *x_col = gsl_vector_alloc(X->size1);
  double *m, **var, sigma, temp, **minmax, min, max;
  int ERROR_IND[1]={0};

  gsl_rng *r;
 
  srand ( (unsigned)time ( NULL ) );
 
  r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r,rand());
  
  gsl_vector_set_all(w, 1);
  m = dvector(1, X->size2);
  var = dmatrix(1, X->size2, 1, X->size2);
  minmax = dmatrix(1, X->size2, 1, 2);

  /* Find Mean and Variance of X */
  
  for(i = 0; i < X->size2; i++){
    gsl_matrix_get_col(x_col, X, i);
    gsl_vector_minmax(x_col, &min, &max);
    minmax[i+1][1]=min;
    minmax[i+1][2]=max;
    m[i+1] = WeightedMean_gsl(x_col, w, X->size1, ERROR_IND);
  }
  
  WeightedVar_gsl(X, m, w, X->size1, X->size2, var, ERROR_IND);

  /* Genearate mean  mu_(state)_i  from a univariate Normal mean m_i 
     and variance var */

  for( i  = 1; i <= (parm_hmm1->N) * (parm_hmm1 -> mix_comp) ; i++){
    for( j = 1; j <= parm_hmm1-> p; j++){
      sigma  = pow(var[j][j], .5);
      do{
	parm_hmm1->mu[i][j] =  gsl_ran_gaussian_ratio_method(r, sigma) + m[j] ; 
	temp = parm_hmm1->mu[i][j];
      } while( temp > minmax[j][2] || temp < minmax[j][1]);
      for(k = 1; k<=parm_hmm1->p ; k++){
	parm_hmm1->Sigma[i][j][k] = var[j][k];
      }
    }
  }

  /* Fixed Initial Values*/
  if(ini->fixed_ini_indicator[2]==1){
    for( i  = 1; i <= (parm_hmm1->N) * (parm_hmm1 -> mix_comp) ; i++){
      for( j = 1; j <= parm_hmm1-> p; j++){
    	parm_hmm1->mu[i][j] = ini->ini_mu[i][j];
      }
    }
  }
  if(ini->fixed_ini_indicator[3]==1){
    for( i  = 1; i <= (parm_hmm1->N) * (parm_hmm1 -> mix_comp) ; i++){
      for( j = 1; j <= parm_hmm1-> p; j++){    
	for(k = 1; k<= parm_hmm1->p; k++){
	  parm_hmm1->Sigma[i][j][k] = ini->ini_Sigma[i][j][k];
	}
      }
    }
  }
  
  
if(0) dmatrix_print(parm_hmm1->mu, parm_hmm1->N, parm_hmm1->p);

  gsl_vector_free(w);
  gsl_vector_free(x_col);
  gsl_rng_free(r);
  free_dvector(m, 1, X->size2);
  free_dmatrix(var, 1, X->size2, 1, X->size2);
  free_dmatrix(minmax, 1, X->size2, 1, 2);

}
  
 
void InitialParmSampleX(PARM_HMM *parm_hmm1, gsl_matrix *X)
{
  int i,j,k;
  gsl_vector *w = gsl_vector_alloc(X->size1);
  gsl_vector *x_col = gsl_vector_alloc(X->size1);
  double *m, **var, sigma, temp, **minmax, min, max;
  int ERROR_IND[1]={0};
  gsl_vector *rand_num = gsl_vector_alloc(X->size1);
  gsl_rng *r;
  int temp_rand, row_X, f1;
  
  srand ( (unsigned)time ( NULL ) );
 
  r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r,rand());
  
  gsl_vector_set_all(w, 1);
  m = dvector(1, X->size2);
  var = dmatrix(1, X->size2, 1, X->size2);
  minmax = dmatrix(1, X->size2, 1, 2);

  /* Find Mean and Variance of X */
  
  for(i = 0; i < X->size2; i++){
    gsl_matrix_get_col(x_col, X, i);
    gsl_vector_minmax(x_col, &min, &max);
    minmax[i+1][1]=min;
    minmax[i+1][2]=max;
    m[i+1] = WeightedMean_gsl(x_col, w, X->size1, ERROR_IND);
  }
  
  WeightedVar_gsl(X, m, w, X->size1, X->size2, var, ERROR_IND);

 
  for( i  = 1; i <= (parm_hmm1->N) * (parm_hmm1 -> mix_comp) ; i++){
    for( j = 1; j <= parm_hmm1-> p; j++){
      for(k = 1; k<=parm_hmm1->p ; k++){
	parm_hmm1->Sigma[i][j][k] = var[j][k];
      }
    }
  }
  
 /* Sample non-repeated  (parm_hmm1->N) * (parm_hmm1 -> mix_comp) 
observations from X*/

  
  for( i  = 1; i <= (parm_hmm1->N) * (parm_hmm1 -> mix_comp) ; i++){
    f1 = 1;
    while(f1){
      temp_rand = (int)((double)  rand()/((double)RAND_MAX +1) * X->size1);
      gsl_vector_set(rand_num, i-1, temp_rand );
      f1=0;
      for(j = 1 ;j < i; j++){
	if(gsl_vector_get(rand_num, j-1) == gsl_vector_get(rand_num, i-1)){
	  f1=1; 
	  j = i+1;
	}
      }
    }
  }
   
  for( i  = 1; i <= (parm_hmm1->N) * (parm_hmm1 -> mix_comp) ; i++){
    row_X = gsl_vector_get(rand_num, i-1);
    for(j = 1; j <= X->size2; j++){
      parm_hmm1->mu[i][j] = gsl_matrix_get(X, row_X  , j-1);
    }
  }

  if(0) dmatrix_print(parm_hmm1->mu, parm_hmm1->N, parm_hmm1->p);

  gsl_vector_free(w);
  gsl_vector_free(x_col);
  gsl_rng_free(r);
  gsl_vector_free(rand_num);
  free_dvector(m, 1, X->size2);
  free_dmatrix(var, 1, X->size2, 1, X->size2);
  free_dmatrix(minmax, 1, X->size2, 1, 2);

}
  
   
void WeightedMean_gsl_matrix(gsl_matrix *X, gsl_matrix *weight_m, PARM_HMM *parm_hmm1, int *ERROR_IND)
{
  /* hmm1 passes the number of states and the length of observation */
  
  gsl_vector *X_col, *weight_col;
  int i, j;
  int M, N, p;

  M = (int) weight_m -> size1;
  N = (int) weight_m -> size2;
  p = (int) X->size2;
  weight_col = gsl_vector_alloc(X->size1);
  X_col = gsl_vector_alloc(X->size1);
  for( i = 0; i< N ; i++)
    {
      gsl_matrix_get_col( weight_col, weight_m, i);
      for( j=0; j< p; j++)
	{
	  gsl_matrix_get_col( X_col, X, j);
	  parm_hmm1->mu[i+1][j+1] = (double) WeightedMean_gsl(X_col, weight_col, (int) M, ERROR_IND);


	  
	}
    }

  gsl_vector_free(X_col);
  gsl_vector_free(weight_col);
}

void WeightedVar_gsl_matrix(gsl_matrix *X, gsl_matrix *weight_m, PARM_HMM *parm_hmm1, int *ERROR_IND)
{
  gsl_vector *weight_col;
  int i, j, k;
  int print_ind = 0;
  int M, N;

  M = weight_m -> size1;
  N = weight_m -> size2;

  weight_col = gsl_vector_alloc(weight_m -> size1);
  
  for( i = 0; i< N ; i++)
    {
      gsl_matrix_get_col( weight_col, weight_m, (size_t) i);
      WeightedVar_gsl(X, parm_hmm1->mu[i+1], weight_col, M, X->size2, parm_hmm1->Sigma[i+1], ERROR_IND);
      if(ERROR_IND[0]) break;
     
    }

  if(print_ind){ 
    for( i = 0; i< N ; i++)
      {
	printf("Var[%d] \n", i);
	for( j=0; j < X->size2; j++)
	  {
	    for( k=0; k < X->size2; k++)
	      {
		
		printf(" %1.7lf \t", parm_hmm1->Sigma[i+1][j+1][k+1]);
	      }
	    printf("\n");
	  }
	printf("\n");
      }
  }
  gsl_vector_free(weight_col);
}

void WeightedSum_B_multiMVN(HMM *hmm1, PARM_HMM *parm_hmm1)
{ 
  int i,j,k;
  double sum;
  
  for( i = 1 ; i <= hmm1-> N; i++){
    for (j = 1; j <= hmm1-> M ; j++){
      sum = 0;
      for ( k =1; k <= hmm1 -> mix_comp; k++){
	sum += (hmm1->B1[i][j][k]) * (parm_hmm1->c[i][k]);
      }
      hmm1-> B[i][j] = sum;
    }
  }
}

void CalculateB_from_B1(HMM *phmm, PARM_HMM *parm_hmm)
{
  int i, j, k, T;
  double sum;

  T = phmm-> M;
  for( i = 0 ; i < phmm->N ; i++){
    for ( j = 0; j < T ; j++){
      sum = 0;
      for( k  =0 ; k <  phmm-> mix_comp ; k++){	
	sum += phmm->B1[i + 1][j + 1][ k+1 ] * parm_hmm->c[i+1][k+1];
      }
      phmm->B[j+1][i+1] =  sum;
    }
  }
}

void CalculateB_from_B1_Log (HMM *phmm, PARM_HMM *parm_hmm)
{
  int i, j, k, T;
  double sum;

  T = phmm-> M;
  for( i = 0 ; i < phmm->N ; i++){
    for ( j = 0; j < T ; j++){
      sum = 0;
      for( k  =0 ; k <  phmm-> mix_comp ; k++){	
	sum += exp(phmm->B1[i + 1][j + 1][ k+1 ]) * parm_hmm->c[i+1][k+1];
      }
      phmm->B[j+1][i+1] =  log(sum);
    }
  }
}



void ReadInitialParm(HMM *hmm, PARM_HMM *parm_hmm, INI *ini)
{

  int i,j,k;
  FILE *parm;
  gsl_matrix *mean = gsl_matrix_alloc(hmm->N * hmm->mix_comp, parm_hmm->p);
  gsl_matrix *var = gsl_matrix_alloc(hmm->N * hmm->mix_comp * parm_hmm->p, parm_hmm->p);

  
  parm = fopen("parm.mean", "rb");
  gsl_matrix_fscanf(parm, mean);
  fclose(parm);

  gsl_matrix_to_dmatrix(mean, parm_hmm->mu);
  
  parm = fopen("parm.var", "rb");
  gsl_matrix_fscanf(parm, var);
  fclose(parm);
  
  printf("mean \n");
  dmatrix_print(parm_hmm->mu, hmm->mix_comp*hmm->N, parm_hmm->p);
  
  for( i = 0; i < hmm->N*hmm->mix_comp; i++){
    for(j = 0; j < parm_hmm->p; j++){
      for( k = 0; k < parm_hmm->p ; k++){
	
	parm_hmm-> Sigma[i+1][j+1][k+1] = gsl_matrix_get(var,i*parm_hmm->p + j ,k);
      }
    }
  }
  if(1){
    for(  i = 0; i < hmm->N*hmm->mix_comp; i++){
      printf("i = %d \n", i);
      dmatrix_print(parm_hmm->Sigma[i+1], parm_hmm->p, parm_hmm->p);
    } 
  }
  
  gsl_matrix_free(mean);
  gsl_matrix_free(var);

}


/* GSL vector and matrix calculation */
double gsl_vector_sum(gsl_vector *a)
{
  int i;
  double sum;
  
  sum = 0;

  for(i = 0; i < a->size; i++){
    sum += (double) gsl_vector_get(a, i);
  }
  return(sum);
}


/* Print  and fprint */

void gsl_vector_print(gsl_vector *v)
{
  int i;
  for( i = 0; i < v->size ; i++)
    {
      printf("%f \n", gsl_vector_get(v, i));
    }
}

void gsl_matrix_print(gsl_matrix *X)
{
  int i, j;
  
  for( i= 0; i< X->size1  ; i++){
    for( j = 0; j < X->size2 ; j++){
      printf("%f   ", gsl_matrix_get(X, i, j));
    }
    printf("\n");
  }
}
void dvector_to_gsl_vector(double *vector, gsl_vector *V)
{
  int i;

  for( i = 0; i < V->size; i++){
    gsl_vector_set(V, i, vector[i+1]);
  }
}

void dmatrix_to_gsl_matrix(double **matrix, gsl_matrix *X)
{
  int i,j;

  for( i = 0; i< X->size1 ; i++)
    {
      for( j = 0; j< X->size2; j++)
	{
	  gsl_matrix_set(X, i, j, matrix[i+1][j+1]);
	}
    }
}

void gsl_matrix_to_dmatrix(gsl_matrix *X, double **matrix)
{
  int i,j;
  
  for( i = 0; i<X->size1; i++){
    for( j = 0; j< X->size2; j++){
      matrix[i+1][j+1]= gsl_matrix_get(X, i, j);
    }
  }
}


void gsl_vector_to_dvector(gsl_vector *v, double *dvector)
{
  int i;
  
  for( i = 0 ; i < v->size; i++){
    dvector[i+1] = gsl_vector_get(v, i);
  }
}

void gsl_vector_to_ivector(gsl_vector *v, int *ivector)
{
  int i;
  
  for( i = 0 ; i < v->size; i++){
    ivector[i+1] = (int) gsl_vector_get(v, i);
  }
}
