/*******************************************************************
*《周志华 机器学习》C++代码
*
* htfeng
* 2018.09.28
*
* 第三章：线性模型
* 主要写线性回归
*******************************************************************/
#include <iostream>
#include <fstream>
#include "include/LinearRegression.h"
#include "include/Utils.h"


using namespace std;

//初始化
LinearRegression::LinearRegression(double x[], double y[], int m) {
	this->x = x;
	this->y = y;
	this->m = m;
}

//梯度下降
double *LinearRegression::gradient_descent(double x[], double y[], double alpha, int iters, double *J, int m) {
	double *theta = new double[2];
	theta[0] = 1;
	theta[1] = 1;
	for (int i = 0; i < iters; i++) {
		double *predictions = calculate_predictions(x, theta, m);
		double *diff = Utils::array_diff(predictions, y, m);
		double *error_x1 = diff;
		double *error_x2 = Utils::array_multiplication(diff, x, m);
		theta[0] = theta[0] - alpha * (1.0 / m) * Utils::array_sum(error_x1, m);
		theta[1] = theta[1] - alpha * (1.0 / m) * Utils::array_sum(error_x2, m);
		J[i] = compute_cost(x, y, theta, m);
	}
	return theta;
}

// 训练函数
void LinearRegression::train(double alpha, int iterations) {
	double *J = new double[iterations];//将J定义为一个包含了iterations个元素的数组
	this->theta = gradient_descent(x, y, alpha, iterations, J, m);
	cout << "J = ";
	for (int i = 0; i < iterations; ++i) {
		cout << J[i] << " ";
	}
	cout << endl << "Theta: " << theta[0] << " " << theta[1] << endl;
}

//预测
double LinearRegression::predict(double x){
	return h(x, theta);
}

//计算误差
double LinearRegression::compute_cost(double x[], double y[], double theta[], int m) {
	double *predictions = calculate_predictions(x, theta, m);
	double *diff = Utils::array_diff(predictions, y, m);
	double *sq_errors = Utils::array_pow(diff, m, 2);
	return (1.0 / (2 * m)) * Utils::array_sum(sq_errors, m);
}

//预测单个值
double LinearRegression::h(double x, double theta[]) {
	return theta[0] + theta[1] * x;
}

//预测
double *LinearRegression::calculate_predictions(double x[], double theta[], int m) {
	double * predictions = new double[m];
	for (int i = 0; i < m; i++) {
		predictions[i] = h(x[i], theta);
	}
	return predictions;
}

