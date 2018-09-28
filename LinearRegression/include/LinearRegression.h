/*******************************************************************
*《周志华 机器学习》C++代码
*
* htfeng
* 2018.09.28
*
* 第三章：线性模型
* 定义一个线性回归的类
*******************************************************************/
#ifndef ML_LINEARREGRESSION_H//如果这个宏没有被定义
#define ML_LINEARREGRESSION_H//则定义宏


class LinearRegression {
public:
	double *x;
	double *y;
	int m;
	double *theta;
	__declspec(dllexport) LinearRegression(double x[], double y[], int m);
	__declspec(dllexport) void train(double alpha, int iterations);
	__declspec(dllexport) double predict(double x);
private:
	//计算模型损失
	__declspec(dllexport) static double compute_cost(double x[], double y[], double theta[], int m);
	//计算单个预测值
	__declspec(dllexport) static double h(double x, double theta[]);
	//预测
	__declspec(dllexport) static double *calculate_predictions(double x[], double theta[], int m);
	//梯度下降
	__declspec(dllexport) static double *gradient_descent(double x[], double y[], double alpha, int iter, double *J, int m);
};

#endif
