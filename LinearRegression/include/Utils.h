/*******************************************************************
*《周志华 机器学习》C++代码
*
* htfeng
* 2018.09.28
*
* 第三章：线性模型
* 累加函数实现
*******************************************************************/
#ifndef __UTILS__
#define __UTILS__
#include<string>
#include<iostream>

using namespace std;


class Utils {
public:
	static double* array_diff(double predictions[], double y[], int m);
	static double* array_multiplication(double diff[], double x[], int m);
	static double* array_pow(double error[], int m, int n);
	static double array_sum(double error[], int m);
};

double* Utils::array_diff(double predictions[], double y[], int m) {
	double *diff = new double[m];
	for (int i = 0; i < m; i++) {
		diff[i] = predictions[i] - y[i];
	}

	return diff;
}

double* Utils::array_multiplication(double diff[], double x[], int m) {
	double *differror = new double[m];
	for (int i = 0; i < m; i++) {
		differror[i] = diff[i] * x[i];
	}

	return differror;
}

double Utils::array_sum(double error[], int m) {
	double sum = 0.0;
	for (int i = 0; i < m; i++) {
		sum += error[i];
	}

	return sum;
}

double* Utils::array_pow(double error[], int m, int n) {
	double *sq_errors = new double[m];
	for (int i = 0; i < m; i++) {
		sq_errors[i] = pow(error[i], n);
	}

	return sq_errors;
}
#endif
