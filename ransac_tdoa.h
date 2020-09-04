/*
 * @Author: Jiang Jiaming
 * @Contact: jiangjiaminghust@gmail.com
 * @Date: 2020-07-10 19:55:21
 * @Description: TDOA problem solved by method Chan, using ransac for robustness. The only interface method is tdoa_ransac().
 */ 

#ifndef RANSAC_LOCATOR_H
#define RANSAC_LOCATOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <assert.h>

#define CALCULABLE_LEAST_CNT 4
#define ENABLE_RANSAC 1
#define ENABLE_RANSAC_REFINE 1
#define ENABLE_POST_CHECK 1
#define RANSAC_MAX_ITERATION 10
#define EPS 1e-10
// comment the following line to disable logs.
#define __DEBUG__  
#ifdef __DEBUG__  
#define DEBUG(format,...) printf(format, ##__VA_ARGS__)  
#else  
#define DEBUG(format,...)  
#endif  

typedef struct {
   double x;
   double y;
} Position2D;


// matrix manipulate.
void MatrixDisplay(double **a,int m,int n){
	int i,j;
	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
			printf("%6.3f ",a[i][j]);
		printf("\n");
	}
	printf("--------------------------\n");
}

/*生成m行n列全矩阵h[][]*/
double ** zeros(int m,int n)
{
	int i,j;
	double **a;
	a=(double**)malloc(m*sizeof(double*));
	for(j=0;j<m;j++)
		a[j]=(double*)malloc(n*sizeof(double));
	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
			a[i][j]=0;
	}
	return a;
}


/*生成m行n列全矩阵h[][]*/
double ** ones(int m,int n)
{
	int i,j;
	double **a;
	a=(double**)malloc(m*sizeof(double*));
	for(j=0;j<m;j++)
		a[j]=(double*)malloc(n*sizeof(double));
	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
			a[i][j]=1;
	}
	return a;
}
//生成单位 
double ** eye(int m)
{
	int i,j;
	double **a;
	a=(double**)malloc(m*sizeof(double*));
	for(j=0;j<m;j++)
		a[j]=(double*)malloc(m*sizeof(double));
	for(i=0;i<m;i++)
	{
		for(j=0;j<m;j++){
			if(i==j)
				a[i][j]=1;
			else
				a[i][j]=0;
		}			
	}
	return a;
}
/*释放矩阵*/
void free_dmatrix(double **dm,int m)
{
	if (dm == NULL) {
		return;
	}
	int i;
	for(i=0;i<m;i++){
		free(dm[i]);
		dm[i] = NULL;
	}		
	free(dm);	
	dm=NULL;
}


/*********************************************************************
作用： 将矩阵转置 
参数： 
	a:       传入的矩阵，必须为二维矩阵 m*n 
	m：      矩阵为 m 行 
	n：      矩阵为 n 列 
返回值：
	返回结果矩阵  n*m
*********************************************************************/
double ** MatrixTranspose(double **a,int m,int n)
{    
	int i,j; 
	double **p;
	p=zeros(n,m);   
	for(i=0;i<m;i++)    
	{        
		for(j=0;j<n;j++)            
			p[j][i]=a[i][j];    
	}
	return p;
}


/*********************************************************************
作用： 两个矩阵相乘
参数： 
	a:       c传入的矩阵，必须为二维矩阵 Ra*La 
	b：      传入的矩阵， 必须为二维矩阵 La*Lb 
	Ra、La、Lb：表示传入的矩阵多少行多少列 
返回值：
	返回结果矩阵  Ra*Lb 
*********************************************************************/
double **MatrixMulti(double **a,double **b,int Ra,int La,int Lb)
{
	int i,j,k;
	double **p;
	p=zeros(Ra,Lb);
	for(i=0;i<Ra;i++)
   	{ 
  		 for(j=0;j<Lb;j++)
	       { 
	  		 for(k=0;k<La;k++)
	           p[i][j]+=a[i][k]*b[k][j];
   			}
    }
    return p;
}

void MatrixScale(double **a, int m, int n, double scale) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			a[i][j] *= scale;
		}
	}
}

/*********************************************************************
作用： 求矩阵的逆矩阵
参数： 
	a:       c传入的矩阵，必须为二维全矩阵 
	m：      矩阵的阶数 
返回值：
	若返回0，则失败（当中求逆失败）；返回1 ，表示成功 
*********************************************************************/
int MatrixInverse(double **a,int m)
{
	int i,j,k;
	double ** parallel;
	double temp;
	parallel = eye(m);

	for (i = 0; i < m; i++) {
		if (a[i][i] == 0) {
			for (k = i; k < m; k++) { // try find one non zero
				if (a[k][i] != 0) {
					for (j = 0; j < m; j++) {
						// swap matrix a
						temp = a[i][j];
						a[i][j] = a[k][j];
						a[k][j] = temp;
						// swap matrix parallel
						temp = parallel[i][j];
						parallel[i][j] = parallel[k][j];
						parallel[k][j] = temp;
					}
					break;
				}
			}
			if (k == m) {
				free_dmatrix(parallel, m);
				return 0;
			}
		}
		for (j = 0; j < m; j++) {
			parallel[i][j] /= a[i][i];
		}
		for (j = m - 1; j >= i; j--) {
			a[i][j] /= a[i][i];
		}
		for (k = 0; k < m; k++) {
			if (k != i) {
				temp = a[k][i];
				for (j = 0; j < m; j++) {
					a[k][j] -= temp * a[i][j];
					parallel[k][j] -= temp * parallel[i][j];
				}
			}
		}
	}

	for(i=0;i<m;i++)
	{
		for(j=0;j<m;j++)
		{
			a[i][j] = parallel[i][j];
		}
	}

	free_dmatrix(parallel, m);
	return 1;
}

// solve weighted least square problem.
// G*x=b, where W as weight matrix
double** solve(double** G, double** b, double** weight, int matrix_row_num, int coefficient_num, int* success_flag) {
    double ** result = NULL;
    double ** GT = NULL;
    double ** A1 = NULL;
    double ** A2 = NULL;
    double ** A3 = NULL;
    double ** A4 = NULL;
    GT = MatrixTranspose(G, matrix_row_num, coefficient_num);
    A1 = MatrixMulti(GT, weight, coefficient_num, matrix_row_num, matrix_row_num);
    A2 = MatrixMulti(A1, G, coefficient_num, matrix_row_num, coefficient_num);
    if (!MatrixInverse(A2, coefficient_num)) {
        result = NULL;
        *success_flag = 0;
    } else {
        A3 = MatrixMulti(A2, GT, coefficient_num, coefficient_num, matrix_row_num);
        A4 = MatrixMulti(A3, weight, coefficient_num, matrix_row_num, matrix_row_num);
        result = MatrixMulti(A4, b, coefficient_num, matrix_row_num, 1);
        *success_flag = 1;
    }
    
    free_dmatrix(GT, coefficient_num);
    free_dmatrix(A1, coefficient_num);
    free_dmatrix(A2, coefficient_num);
    free_dmatrix(A3, coefficient_num);
    free_dmatrix(A4, coefficient_num);
    return result;
}

void calculate_distances_given_position(const Position2D landmarks[], int total_landmark_num, const Position2D* position, double distances_out[]) {
    for (int i = 0; i < total_landmark_num; i++) {
        distances_out[i] = sqrt(pow((landmarks[i].x - position->x), 2) + pow((landmarks[i].y - position->y), 2));
    }
}

int is_fit(const double difference_of_distances[], const double distances_estimated[], 
        const int sample_indexes[], int sample_size,
        int index, double threshold) {
    // DEBUG("threshold:%f, is_fit(%d), distances:", threshold, index);
    for (int i = 0; i < sample_size; i++) {
        double difference_of_difference_of_distances = fabs((difference_of_distances[sample_indexes[i]]-difference_of_distances[index]) - (distances_estimated[sample_indexes[i]]-distances_estimated[index]));
        // DEBUG("%f,", difference_of_difference_of_distances);
        if (difference_of_difference_of_distances > threshold) {
            // DEBUG(", this failed.\n");
            return 0;
        }
    }
    // DEBUG("\n");
    return 1;
}

int check_fit_all_samples(const double difference_of_distances[], const double distances_estimated[], 
        const int sample_indexes[], int sample_size, double threshold) {
    for (int i = 0; i < sample_size; i++) {
        if (!is_fit(difference_of_distances, distances_estimated, sample_indexes, sample_size, sample_indexes[i], threshold)) {
            return 0;
        }
    }
    return 1;
}

int is_landmarks_in_one_line(const Position2D landmarks[], const int effective_landmark_indexes[], int effective_landmark_num) {
    assert(effective_landmark_num > 2);
    double angle_0 = atan2(landmarks[effective_landmark_indexes[0]].y - landmarks[effective_landmark_indexes[effective_landmark_num-1]].y, 
            landmarks[effective_landmark_indexes[0]].x - landmarks[effective_landmark_indexes[effective_landmark_num-1]].x);
    
    for (int i = 1; i < effective_landmark_num - 1; i++) {
        double angle_i = atan2(landmarks[effective_landmark_indexes[i]].y - landmarks[effective_landmark_indexes[effective_landmark_num-1]].y, 
                landmarks[effective_landmark_indexes[i]].x - landmarks[effective_landmark_indexes[effective_landmark_num-1]].x);
        
        int is_this_in_line = (fabs(angle_i - angle_0) < EPS) || (fabs(fabs(angle_i - angle_0) - M_PI) < EPS);

        if (!is_this_in_line) {
            return 0;
        }
    }
    return 1;
}

int post_check_result_pos(const Position2D landmarks[], const double difference_of_distances[], int total_landmark_num, 
        const int effective_landmark_indexes[], int effective_landmark_num,
        const Position2D* result_pos, double threshold) {
    double* distances_estimated = (double*)malloc(sizeof(double) * total_landmark_num);
    memset(distances_estimated, 0, sizeof(double) * total_landmark_num);
    calculate_distances_given_position(landmarks, total_landmark_num, result_pos, distances_estimated);
    int ret = check_fit_all_samples(difference_of_distances, distances_estimated, effective_landmark_indexes, effective_landmark_num, threshold);
    free(distances_estimated);
    return ret;
}

int tdoa_chan(const Position2D landmarks[], const double difference_of_distances[], int total_landmark_num, 
        const int effective_landmark_indexes[], int effective_landmark_num, double threshold,
        Position2D* result_pos) {
    int success_flag = 0;
    assert(effective_landmark_num >= CALCULABLE_LEAST_CNT);
    assert(total_landmark_num >= effective_landmark_num);
    if (is_landmarks_in_one_line(landmarks, effective_landmark_indexes, effective_landmark_num)) {
        DEBUG("all landmarks in one line, cannot locate the position.\n");
        return 0;
    }
    // matrices for input arguments
    double **Q = NULL;
    double **Ga = NULL;
    double **h = NULL;
	double **GT = NULL;
    // 1st least square matrices
	double **Za0 = NULL;
    // 2nd weighted least square matrices
	double **B = NULL;
    double **B0 = NULL;
    double **FI = NULL;
	double **Za1 = NULL;
    // 3rd weighted least square matrices
	double **sB = NULL;
    double **sFI = NULL;
    double **sGa = NULL;
    double **sh = NULL;
	double **C1 = NULL;
    double **CovZa = NULL;
    double **D1 = NULL;
	double sGa1[3][2]={1,0,0,1,1,1};
	double **Za2 = NULL;
    
    int matrix_row_num = effective_landmark_num - 1;
    int coefficient_num = 3;
    Q = eye(matrix_row_num);

    // step 0. prepare data
    Ga = zeros(matrix_row_num, 3);
    h = zeros(matrix_row_num, 1);
    int index_0  = effective_landmark_indexes[0];
    int k_0 =landmarks[index_0].x * landmarks[index_0].x + landmarks[index_0].y * landmarks[index_0].y;
    for (int i = 0; i < matrix_row_num; i++) {
        int index_ip1 = effective_landmark_indexes[i+1];
        double r_i_0 = difference_of_distances[index_ip1] - difference_of_distances[index_0];
        Ga[i][0] = -(landmarks[index_ip1].x - landmarks[index_0].x);
        Ga[i][1] = -(landmarks[index_ip1].y - landmarks[index_0].y);
        Ga[i][2] = -r_i_0;
        double k_i = landmarks[index_ip1].x * landmarks[index_ip1].x + landmarks[index_ip1].y * landmarks[index_ip1].y;
        h[i][0] = 0.5*(r_i_0 * r_i_0 - k_i + k_0);
    }
    // step 1. the 1st least square, to get a raw position.
    // Za0 = inv(Ga'*inv(Q)*Ga)*Ga'*inv(Q)*h'
    int solve_res = 0;
    Za0 = solve(Ga, h, Q, matrix_row_num, coefficient_num, &solve_res);
    if (!(solve_res && Za0[0][0] > -1e10 && Za0[0][0] < 1e10 && Za0[1][0] > -1e10 && Za0[1][0] < 1e10)) {
        success_flag = 0;
        DEBUG("failed when solve Za0.\n"); // not solvable, or ill-condition.
        goto LABEL_FREE_BEFORE_EXIT;
    }
    double Mx0 = Za0[0][0];
    double My0 = Za0[1][0];
    DEBUG("step1 result:%f, %f\n", Mx0, My0);

    // step 2. the 2nd weighted least square, to get a better position by introducing weights calculated from Za0
    // Za1=inv(Ga'*inv(FI)*Ga)*Ga'*inv(FI)*h'   in which  FI=B*Q*B  .
    // Tip: the position result of step 2 will be the same as of step 1 when effective_landmark_num is 4, 
    //         in which case the least square problem here will by positive definite rather than overdetermined.
    //         But we still need it to calculate Za1[2][0].
    B = eye(matrix_row_num);
    for (int i = 0; i < matrix_row_num; i++) {
        int index_ip1 = effective_landmark_indexes[i+1];
        B[i][i] = sqrt((landmarks[index_ip1].x - Mx0) * (landmarks[index_ip1].x - Mx0) + (landmarks[index_ip1].y - My0) * (landmarks[index_ip1].y - My0));
    }
    B0 = MatrixMulti(B, Q, matrix_row_num, matrix_row_num, matrix_row_num);
    FI = MatrixMulti(B0, B, matrix_row_num, matrix_row_num, matrix_row_num); 
    if(!MatrixInverse(FI, matrix_row_num)) {
        success_flag = 0;
        DEBUG("failed when invert FI.\n");
        goto LABEL_FREE_BEFORE_EXIT;
    }
    solve_res = 0;
    Za1 = solve(Ga, h, FI, matrix_row_num, coefficient_num, &solve_res);
    if (!(solve_res && Za1[0][0] > -1e10 && Za1[0][0] < 1e10 && Za1[1][0] > -1e10 && Za1[1][0] < 1e10)) {
        success_flag = 0;
        DEBUG("failed when solve Za1.\n");
        goto LABEL_FREE_BEFORE_EXIT;
    }
    double Mx1 = Za1[0][0];
    double My1 = Za1[1][0];
    DEBUG("step2 result:%f, %f\n", Mx1, My1);

    // Step 3. the 3rd weighted least square, using a new equation which has no correlated arguments in it(unlike the 2 least square problems above).
    // Za2=inv(sGa'*inv(sFI)*sGa)*sGa'*inv(sFI)*sh  in which  sFI=4*sB*CovZa*sB  and  CovZa=inv(Ga'*inv(FI)*Ga) .
    GT = MatrixTranspose(Ga, matrix_row_num, coefficient_num);
    C1 = MatrixMulti(GT, FI, coefficient_num, matrix_row_num, matrix_row_num);
    CovZa = MatrixMulti(C1, Ga, coefficient_num, matrix_row_num, coefficient_num);
    if (!MatrixInverse(CovZa, coefficient_num)) {
        success_flag = 0;
        DEBUG("failed when invert CovZa.\n");
        goto LABEL_FREE_BEFORE_EXIT;
    }
    sB = eye(3);
    sB[0][0] = Mx1 - landmarks[index_0].x;
    sB[1][1] = My1 - landmarks[index_0].y;
    sB[2][2] = Za1[2][0];
    D1 = MatrixMulti(sB, CovZa, 3, 3, 3);
    sFI = MatrixMulti(D1, sB, 3, 3, 3);
    MatrixScale(sFI, 3, 3, 4.0);

    if (!MatrixInverse(sFI, 3)) {
        success_flag = 0;
        DEBUG("failed when invert sFI.\n");
        goto LABEL_FREE_BEFORE_EXIT;
    }
    
    sGa = zeros(3, 2);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            sGa[i][j] = sGa1[i][j];
        }
    }
    sh = zeros(3, 1);
    sh[0][0] = (Mx1 - landmarks[index_0].x) * (Mx1 - landmarks[index_0].x);
    sh[1][0] = (My1 - landmarks[index_0].y) * (My1 - landmarks[index_0].y);
    sh[2][0] = Za1[2][0] * Za1[2][0];
    solve_res = 0;
    Za2 = solve(sGa, sh, sFI, 3, 2, &solve_res);
    if (!(solve_res && Za2[0][0] > -1e10 && Za2[0][0] < 1e10 && Za2[1][0] > -1e10 && Za2[1][0] < 1e10)) {
        success_flag = 0;
        DEBUG("failed when solve Za2.\n");
        goto LABEL_FREE_BEFORE_EXIT;
    }
    if (Za2[0][0] < 0 || Za2[1][0] < 0) {
        DEBUG("WARNING: Za2 results unexpected.\n");
    }
    double delta_x_0_abs = Za2[0][0] > 0 ? sqrt(Za2[0][0]) : 0;
    double delta_y_0_abs = Za2[1][0] > 0 ? sqrt(Za2[1][0]) : 0;
    DEBUG("delta_x_0_abs:%f, delta_y_0_abs:%f.\n", delta_x_0_abs, delta_y_0_abs);
    double popssible_poses[4][2] = {
        {landmarks[index_0].x - delta_x_0_abs, landmarks[index_0].y - delta_y_0_abs}, 
        {landmarks[index_0].x + delta_x_0_abs, landmarks[index_0].y - delta_y_0_abs},
        {landmarks[index_0].x - delta_x_0_abs, landmarks[index_0].y + delta_y_0_abs},
        {landmarks[index_0].x + delta_x_0_abs, landmarks[index_0].y + delta_y_0_abs}
    };
    // choose the one based on pre-aquired position.
    int best_index = -1;
    double smallest_distance_2_za1 = DBL_MAX;
    for (int i = 0; i < 4; i++) {
        double square_distance_2_za1 = (Mx1 - popssible_poses[i][0]) * (Mx1 - popssible_poses[i][0]) + (My1 - popssible_poses[i][1]) * (My1 - popssible_poses[i][1]);
        if (square_distance_2_za1 < smallest_distance_2_za1) {
            best_index = i;
            smallest_distance_2_za1 = square_distance_2_za1;
        }
    }
    result_pos->x = popssible_poses[best_index][0];
    result_pos->y = popssible_poses[best_index][1];
    success_flag = 1;

    if (ENABLE_POST_CHECK && effective_landmark_num == CALCULABLE_LEAST_CNT) {
        success_flag = post_check_result_pos(landmarks, difference_of_distances, total_landmark_num, effective_landmark_indexes, effective_landmark_num,
            result_pos, threshold);
        if (success_flag == 0) {
            DEBUG("post check failed.\n");
        }
    }
LABEL_FREE_BEFORE_EXIT:
    free_dmatrix(Q, matrix_row_num);
    free_dmatrix(Ga, matrix_row_num);
    free_dmatrix(h, matrix_row_num);
    free_dmatrix(GT, coefficient_num);
    free_dmatrix(Za0, coefficient_num);
    free_dmatrix(B, matrix_row_num);
    free_dmatrix(B0, matrix_row_num);
    free_dmatrix(FI, matrix_row_num);
    free_dmatrix(Za1, coefficient_num);
    free_dmatrix(sB, 3);
    free_dmatrix(sFI, 3);
    free_dmatrix(sGa, 3);
    free_dmatrix(sh, 3);
    free_dmatrix(C1, coefficient_num);
    free_dmatrix(CovZa, coefficient_num);
    free_dmatrix(D1, 3);
    free_dmatrix(Za2, 2);
    return success_flag;
}

void getAllSampleIndexes(const int sample_from_indexes[], int index_size, int sample_size, int max_num_of_samples, int sample_output_indexes[][CALCULABLE_LEAST_CNT]) {
    assert(index_size >= sample_size);
    int success_cnt = 0;
    
    int * shuffled_indices = (int*) malloc(sizeof(int) * index_size);
    for (int i = 0; i < index_size; i++) {
        shuffled_indices[i] = i;
    }
    while(success_cnt < max_num_of_samples) {
        // shuffle
        for (int i = 0; i < index_size - 1; i++) {
            // swap i and random index afterwards(including 'i' itself)
            int rand_index = i + (rand() % (index_size - i));
            int tmp = shuffled_indices[i];
            shuffled_indices[i] = shuffled_indices[rand_index];
            shuffled_indices[rand_index] = tmp;
        }
        
        for (int i = 0; i < index_size; i++) {
            for (int j = 0; j < sample_size; j++) {
                sample_output_indexes[success_cnt][j] = sample_from_indexes[shuffled_indices[(i + j) % index_size]];
            }
            success_cnt++;
            if (success_cnt >= max_num_of_samples) {
                break;
            }
        }
    }
    free(shuffled_indices);
    return;
}

double calc_total_residual(const double difference_of_distances[], const double distances_estimated[], 
        const int sample_indexes[], int sample_size, const int all_fit_indexes[], int all_fit_cnt) {
    double residual = 0;
    for (int i = 0; i < all_fit_cnt; i++) {
        for (int j = 0; j < sample_size; j++) {
            double difference_of_difference_of_distances = (difference_of_distances[sample_indexes[j]]-difference_of_distances[all_fit_indexes[i]]) 
                - (distances_estimated[sample_indexes[j]]-distances_estimated[all_fit_indexes[i]]);
            residual += difference_of_difference_of_distances * difference_of_difference_of_distances;
        }
    }
    return residual;
}

int get_all_fit_indexes(const Position2D landmarks[], const double difference_of_distances[], int total_landmark_num, 
        const int effective_landmark_indexes[], int effective_landmark_num, 
        const int sample_indexes[], int sample_size, const Position2D* position, double threshold, 
        int all_fit_indexes_out[], double* total_residual_out) {
    int* is_fit_vec = (int*)malloc(sizeof(int) * total_landmark_num);
    memset(is_fit_vec, 0, sizeof(int) * total_landmark_num);
    double* residuals = (double*)malloc(sizeof(double) * total_landmark_num);
    memset(residuals, 0, sizeof(double) * total_landmark_num);

    double* distances_estimated = (double*)malloc(sizeof(double) * total_landmark_num);
    memset(distances_estimated, 0, sizeof(double) * total_landmark_num);
    calculate_distances_given_position(landmarks, total_landmark_num, position, distances_estimated);
    
    // add samples
    for (int i = 0; i < sample_size; i++) {
        int index_sample_i = sample_indexes[i];
        is_fit_vec[index_sample_i] = 2; // set 2 rather than 1 as other landmarks, in case of being added twice.
    }
    for (int i = 0; i < effective_landmark_num; i++) {
        int index_effective_i = effective_landmark_indexes[i];
        if (is_fit_vec[index_effective_i] == 0) { // not including samples.
            if (is_fit(difference_of_distances, distances_estimated, sample_indexes, sample_size, index_effective_i, threshold)) {
                is_fit_vec[index_effective_i] = 1;
            }
        }
    }
    int j = 0;
    int cnt = 0;
    
    // add sample index first
    for (int i = 0; i < sample_size; i++) {
        int index_sample_i = sample_indexes[i];
        all_fit_indexes_out[j++] = index_sample_i;
        cnt++;
    }
    for (int i = 0; i < total_landmark_num; i++) {
        if (is_fit_vec[i] == 1) {
            all_fit_indexes_out[j++] = i;
            cnt++;
        }
    }
    *total_residual_out = calc_total_residual(difference_of_distances, distances_estimated, sample_indexes, sample_size, all_fit_indexes_out, cnt);

    free(is_fit_vec);
    free(residuals);
    free(distances_estimated);
    return cnt;
}

/***
 * @description: TDOA problem solved by method Chan, using ransac for robustness. 
 *              Details for method Chan: A simple and Efficient Estimator for Hyperbolic Location.
 * @input:
 *      landmarks: containing all landmarks' positions in the map.
 *      distances: relative distances from landmark to current position for TDOA problem. 
 *                 Only the distance of the detected(effective) landmarks need to be set.
 *      effective_landmark_mask: mask that indicates which landmarks are detected(effective). 
 *      total_landmark_num: The length of landmarks, distances, and effective_landmark_mask are all total_landmark_num.
 *      std_dev: the standard deviation of measurements, this value is used to filter noise data.
 * @output:
 *      result_pos: the result position.
 * @return: success or not. The method may fail when the effective landmarks are too less, or they are in one line, 
 *          or some matrix operations failed.
 ***/
int tdoa_ransac(const Position2D landmarks[], const double distances[], 
        const int effective_landmark_mask[], int total_landmark_num, double std_dev,
        Position2D* result_pos) {
    int success_flag = 0;
    double threshold = std_dev * 3;
    int* effective_landmark_indexes = (int*)malloc(sizeof(int) * total_landmark_num);
    int effective_landmark_num = 0;
    int j = 0;
    for (int i = 0; i < total_landmark_num; i++) {
        if(effective_landmark_mask[i]) {
            effective_landmark_indexes[j++] = i;
            effective_landmark_num++;
        }
    }
    if (effective_landmark_num < CALCULABLE_LEAST_CNT) {
        success_flag = 0;
    } else if (effective_landmark_num == CALCULABLE_LEAST_CNT || !ENABLE_RANSAC) {
        success_flag = tdoa_chan(landmarks, distances, total_landmark_num, 
            effective_landmark_indexes, total_landmark_num, threshold, result_pos);
    } else { // ransac
        double residual_best = DBL_MAX;
        int fit_cnt_best = -1;

        int sample_output_indexes[RANSAC_MAX_ITERATION][CALCULABLE_LEAST_CNT];
        getAllSampleIndexes(effective_landmark_indexes, effective_landmark_num, CALCULABLE_LEAST_CNT, RANSAC_MAX_ITERATION, sample_output_indexes);

        int sample_indexes_best[CALCULABLE_LEAST_CNT];
        int* all_fit_indexes = (int*)malloc(sizeof(int) * effective_landmark_num);
        int* all_fit_indexes_best = (int*)malloc(sizeof(int) * effective_landmark_num);
        Position2D tmp_pos;
        for (int i = 0; i < RANSAC_MAX_ITERATION; i++) {
            DEBUG("========iteration:%d===========\nchosen indexes:", i);
            int* sample_indexes = sample_output_indexes[i];
            for (int j = 0; j < CALCULABLE_LEAST_CNT; j++) {
                DEBUG("%d,", sample_indexes[j]);
            }
            DEBUG("\n");

            int success_once = tdoa_chan(landmarks, distances, total_landmark_num, 
                sample_indexes, CALCULABLE_LEAST_CNT, threshold, &tmp_pos);
            if (success_once) {
                DEBUG("result x:%f, y:%f\n", tmp_pos.x, tmp_pos.y);

                double total_residual = DBL_MAX;
                int n = get_all_fit_indexes(landmarks, distances, total_landmark_num, effective_landmark_indexes, effective_landmark_num, 
                    sample_indexes, CALCULABLE_LEAST_CNT, &tmp_pos, threshold, 
                    all_fit_indexes, &total_residual);

                DEBUG("fit cnt:%d, residual:%f, all_fit_indexes:", n, total_residual);
                for (int i = 0; i < n; i++) {
                    DEBUG("%d,", all_fit_indexes[i]);
                }
                DEBUG("\n");
                if (n > fit_cnt_best || (n == fit_cnt_best && total_residual < residual_best)) {
                    residual_best = total_residual;
                    fit_cnt_best = n;
                    for (int j = 0; j < n; j++) {
                        all_fit_indexes_best[j] = all_fit_indexes[j];
                    }
                    for (int j = 0; j < CALCULABLE_LEAST_CNT; j++) {
                        sample_indexes_best[j] = sample_indexes[j];
                    }
                    result_pos->x = tmp_pos.x;
                    result_pos->y = tmp_pos.y;
                    success_flag = 1;
                }
            }
        }
        // refine model
        if (success_flag) {
            DEBUG("\n===============\nbefore refine[best]. best sample_indexes:");
            for (int i = 0; i < CALCULABLE_LEAST_CNT; i++) {
                DEBUG("%d,", sample_indexes_best[i]);
            }
            DEBUG(" result:%f, %f, residual:%f.\n", result_pos->x, result_pos->y, residual_best);
            if (ENABLE_RANSAC_REFINE && fit_cnt_best > CALCULABLE_LEAST_CNT) {
                Position2D pos_refined;
                DEBUG("fit_cnt_best:%d. refine indexes:", fit_cnt_best);
                for (int i = 0; i < fit_cnt_best; i++) {
                    DEBUG("%d,", all_fit_indexes_best[i]);
                }
                DEBUG("\n");
                int success_once = tdoa_chan(landmarks, distances, total_landmark_num, 
                    all_fit_indexes_best, fit_cnt_best, threshold, &pos_refined);
                if (success_once) {
                    double residual_refined = 0.0;
                    double* distances_estimated = (double*)malloc(sizeof(double) * total_landmark_num);
                    memset(distances_estimated, 0, sizeof(double) * total_landmark_num);
                    for (int i = 0; i < total_landmark_num; i++) {
                        distances_estimated[i] = sqrt(pow((landmarks[i].x - pos_refined.x), 2) + pow((landmarks[i].y - pos_refined.y), 2));
                    }
                    residual_refined = calc_total_residual(distances, distances_estimated, sample_indexes_best, CALCULABLE_LEAST_CNT, all_fit_indexes_best, fit_cnt_best);
                    DEBUG("after refine. result:%f, %f, residual:%f\n", pos_refined.x, pos_refined.y, residual_refined);
                    if (residual_refined < residual_best) {
                        result_pos->x = pos_refined.x;
                        result_pos->y = pos_refined.y;
                    }
                    free(distances_estimated);
                }
            }
        }
        free(all_fit_indexes);
        free(all_fit_indexes_best);
    }

    free(effective_landmark_indexes);
    
    return success_flag;
}

#endif // RANSAC_LOCATOR_H