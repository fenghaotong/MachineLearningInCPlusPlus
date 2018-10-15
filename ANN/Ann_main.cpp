//main.cpp
#include "include/Ann.h"


int main()
{
	const int hidnodes = 8; //单层隐藏层的结点数
	const int inNodes = 10;   //输入层结点数
	const int outNodes = 5;  //输出层结点数

	const int trainClass = 5; //5个类别
	const int numPerClass = 30;  //每个类别30个样本点

	int sampleN = trainClass * numPerClass;     //每类训练样本数为30，5个类别，总的样本数为150
	float **trainMat = new float*[sampleN];                         //生成训练样本
	for (int k = 0; k < trainClass; ++k) {
		for (int i = k * numPerClass; i < (k + 1) * numPerClass; ++i) {
			trainMat[i] = new float[inNodes];
			for (int j = 0; j < inNodes; ++j) {
				trainMat[i][j] = rand() % 1000 / 10000.0 + 0.1*(2 * k + 1);

			}
		}
	}

	int **labelMat = new int*[sampleN]; //生成标签矩阵
	for (int k = 0; k < trainClass; ++k) {
		for (int i = k * numPerClass; i < (k + 1) * numPerClass; ++i) {
			labelMat[i] = new int[outNodes];
			for (int j = 0; j < trainClass; ++j) {
				if (j == k)
					labelMat[i][j] = 1;
				else
					labelMat[i][j] = 0;
			}

		}
	}

	Ann ann_classify(sampleN, inNodes, outNodes, hidnodes, 0.12);  //输入层为10个结点，输出层5个结点，单层隐藏层
	ann_classify.train(sampleN, trainMat, labelMat);


	for (int i = 0; i < 30; ++i) {
		ann_classify.predict(trainMat[i + 120], NULL);
		std::cout << std::endl;
	}


	//释放内存
	for (int i = 0; i < sampleN; ++i)
		delete[] trainMat[i];
	delete[] trainMat;

	for (int i = 0; i < sampleN; ++i)
		delete[] labelMat[i];
	delete[] labelMat;

	system("pause");
	return 0;
}