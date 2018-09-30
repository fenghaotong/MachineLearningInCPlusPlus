/*******************************************************************
*《周志华 机器学习》C++代码
*
* htfeng
* 2018.09.30
*
* 第三章：线性模型
* 定义一个决策树的类
*******************************************************************/
#ifndef ML_DECISION_H//如果这个宏没有被定义
#define ML_DECISION_H//则定义宏

#include "DecisionTreeStruct.h"
#include <string> 
#define MAXLEN 6 //输入每行的数据个数  

class DecisionTree {
public:
	__declspec(dllexport) DecisionTree(vector<vector<string>> state, int tree_size);
	__declspec(dllexport) void ComputeMapFrom2DVector();
	__declspec(dllexport) double ComputeEntropy(vector <vector <string> > remain_state, string attribute, string value, bool ifparent);
	__declspec(dllexport) double ComputeGain(vector <vector <string> > remain_state, string attribute);
	__declspec(dllexport) int FindAttriNumByName(string attri);
	__declspec(dllexport) string MostCommonLabel(vector <vector <string> > remain_state);
	__declspec(dllexport) bool AllTheSameLabel(vector <vector <string> > remain_state, string label);
	__declspec(dllexport) Node * BulidDecisionTreeDFS(Node * p, vector <vector <string> > remain_state, vector <string> remain_attribute);
	__declspec(dllexport) void Input();
	__declspec(dllexport) void PrintTree(Node *p, int depth);
	__declspec(dllexport) void FreeTree(Node *p);

private:
	vector <vector <string> > state;//实例集  
	vector <string> item{ MAXLEN };//对应一行实例集  
	vector <string> attribute_row;//保存首行即属性行数据  
	string end = "end";//输入结束  
	string yes = "yes";
	string no = "no";
	string blank = "";
	map<string, vector < string > > map_attribute_values;//存储属性对应的所有的值  
	int tree_size = 0;
};
#endif