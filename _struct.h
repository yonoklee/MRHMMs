#ifndef  HMM_struct
#define HMM_struct
  
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

typedef struct {
        char modelname[100];
        int N;          /* number of states;  Q={1,2,...,N} */
        int M;          /* number of observation symbols; V={1,2,...,M}*/
        double  **A;    /* A[1..N][1..N]. a[i][j] is the transition prob
                           of going from state i at time t to state j
                           at time t+1 */
        double ***A_t;   /* A_t[1...N][1..N] Transition probability at time t */
        double  **B;    /* B[1..M][1..N]. b[j][k] is the probability of
                           of observing symbol k in state j */
        double **Braw;
        double  *pi;    /* pi[1..N] pi[i] is the initial state distribution. */

        double  ***B1;   /* B1[1..N][1..M][1..m], where m is the number of 
			   multivarate normal mixtures */
        int mix_comp;    /* number of mixture component */
        int window_dist;     /* 0 for homogeneous transition probabilty */
        int num_chr;   /* number of independent data sets */
        int *new_chr;  /* Location of the independent data start */
        double B_scale;  /* The scaled emission probability */
        int D1;  /* Parameter for nonhomogeneous transition probability */
        double *B_scale_multi;   /* The scaled emission probability 
				    for each independent data set*/
  
            
} HMM;

typedef struct {
  int N; /* Number of states */
  int p; /* The number of columns in X */
  int d; /* The number of columns in Y */
  int mix_comp;    /* number of mixture component */
  int user_para;  /*  Indicator 0/1 if User defiend parameters */
  int reg_coef_user_num; /* The number of fixed parameters defined 
			    Work with ini-> reg_coef_fixed_parm 
			 */
  int reg_cov_user_para;
  int mu_user_para;  /* 1 if user used fixed parameter in mu */
  int Sigma_user_para;
  int c_user_para;

  int **reg_coef_fixed_parm; /* Indicator 0 if need to be estimated
				      1 if user provides fixed value */

  double **mu;           /* Mean of MVN ( N*mix_comp) by p */
  double ***Sigma;       /* Variance of MVN (p by  p ) by  (N * mix_comp) */
  double **c ;           /* c[1..N][1..m] is the component probabilities, 
			   where N is the number of states and
			   m is the number of multivaraite normal mixtures. */
  double **reg_coef;     /* Regression coefficients (d * N * mix_comp) by p+1 */
  double ***reg_cov;     /* Regression standard error
			    (d by d) * (N * mix_comp) */
  
  
} PARM_HMM;


typedef struct { 
  char loc_Data_X[100];  /* Data file location X */
  char loc_Data_Y[100];  /* Data file location Y */
  char loc_window_dist[100]; /*Time file location */
  char output_filename[100] ; /* Output file name */
  int *ini_state;
  int fixed_ini_indicator[8]; /* 1 if a user fixes initial 
				 value for the corresponding location (0 - 5)*/
  int  rowX;  /* sample size */
  int  colX;  /* p */
  int  rowY;  /* sample size */
  int  colY;  /* d */
  int  itr;   /* Repetition */
  int MaxItr; /* The maximum iteration allowed */ 
  double Tolerance; /* Tolerance */
  int num_data_sets; 
  int dim_data_sets[100];

  /* For fixed initial values */
  double *ini_pi, **ini_A, **ini_mu, ***ini_Sigma, **ini_reg_coef, ***ini_reg_cov, **ini_c; 

  /* For fixed USER.PARAMETERS in emission density */
  double **mu;  
  double **reg_coef;
  int **reg_coef_fixed_parm;


  /* Not used in MRHMMs */
  double **c;
  double ***Sigma;
  double ***reg_cov;
  
} INI;



#endif
