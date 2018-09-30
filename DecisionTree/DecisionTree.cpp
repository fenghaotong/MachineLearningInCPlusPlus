/*******************************************************************
*《周志华 机器学习》C++代码
*
* htfeng
* 2018.09.30
*
* 第三章：线性模型
* 决策树
*******************************************************************/
#include "include/DecisionTree.h"

//根据数据实例计算属性与值组成的map  
void DecisionTree::ComputeMapFrom2DVector() {
	unsigned int i, j, k;
	bool exited = false;
	vector<string> values;
	for (i = 1; i < MAXLEN - 1; i++) {//按照列遍历  
		for (j = 1; j < state.size(); j++) {
			for (k = 0; k < values.size(); k++) {
				if (!values[k].compare(state[j][i])) exited = true;
			}
			if (!exited) {
				values.push_back(state[j][i]);//注意Vector的插入都是从前面插入的，注意更新it，始终指向vector头  
			}
			exited = false;
		}
		map_attribute_values[state[0][i]] = values;
		values.erase(values.begin(), values.end());
	}
}

//根据具体属性和值来计算熵
double DecisionTree::ComputeEntropy(vector <vector <string> > remain_state, 
	string attribute, string value, bool ifparent) {
	vector<int> count(2, 0);
	unsigned int i, j;
	bool done_flag = false;
	for (j = 1; j < MAXLEN; j++) {
		if (done_flag) break;
		if (!attribute_row[j].compare(attribute)) {
			for (i = 1; i < remain_state.size(); i++) {
				if ((!ifparent && !remain_state[i][j].compare(value)) || ifparent) {
					if (!remain_state[i][MAXLEN - 1].compare(yes)) {
						count[0] ++;
					}
					else count[1] ++;
				}
			}
			done_flag = true;
		}
	}

	// 全是正例或者全是反例
	if (count[0] == 0 || count[1] == 0)  return 0;

	double sum = count[0] + count[1];
	double entropy = -count[0] / sum * log(count[0] / sum) / log(2.0) - count[1] / sum * log(count[1] / sum) / log(2.0);
	return entropy;
}

// 计算按照属性attribute划分当前剩余实例的信息增益
double DecisionTree::ComputeGain(vector <vector <string> > remain_state, string attribute) {
	unsigned int j, k, m;
	double parent_entropy = ComputeEntropy(remain_state, attribute, blank, true);
	double children_entropy = 0;

	vector<string> values = map_attribute_values[attribute];	
	vector<double> ratio;	
	vector<int> count_values;	
	int tempint;	
	for (m = 0; m < values.size(); m++) { 
		tempint = 0;		
		for (k = 1; k < MAXLEN - 1; k++) { 
			if (!attribute_row[k].compare(attribute)) {
				for (j = 1; j < remain_state.size(); j++) {
					if (!remain_state[j][k].compare(values[m])) {
						tempint++; 
					} 
				}
			} 
		}		
		count_values.push_back(tempint); 
	}

	for (j = 0; j < values.size(); j++) {
		ratio.push_back((double)count_values[j] / (double)(remain_state.size() - 1));
	}
	double temp_entropy;
	for (j = 0; j < values.size(); j++) {
		temp_entropy = ComputeEntropy(remain_state, attribute, values[j], false);
		children_entropy += ratio[j] * temp_entropy;
	}
	return (parent_entropy - children_entropy);
}

int DecisionTree::FindAttriNumByName(string attri) {
	for (int i = 0; i < MAXLEN; i++) {
		if (!state[0][i].compare(attri)) return i;
	}

	cerr << "can't find the numth of attribute" << endl;
	return 0;
}

//找出样例中占多数的正/负性  
string DecisionTree::MostCommonLabel(vector <vector <string> > remain_state) {
	int p = 0, n = 0;
	for (unsigned i = 0; i < remain_state.size(); i++) {
		if (!remain_state[i][MAXLEN - 1].compare(yes)) p++;
		else n++;
	}
	if (p >= n) return yes;
	else return no;
}

//判断样例是否正负性都为label  
bool DecisionTree::AllTheSameLabel(vector <vector <string> > remain_state, string label) {
	int count = 0;
	for (unsigned int i = 0; i < remain_state.size(); i++) {
		if (!remain_state[i][MAXLEN - 1].compare(label)) count++;
	}
	if (count == remain_state.size() - 1) return true;
	else return false;
}

//计算信息增益，DFS构建决策树
Node * DecisionTree::BulidDecisionTreeDFS(Node * p, vector <vector <string> > remain_state,
	vector <string> remain_attribute) {
	if (p == NULL)
		p = new Node();

	if (AllTheSameLabel(remain_state, yes)) {
		p->attribute = yes;
		return p;
	}

	if (AllTheSameLabel(remain_state, no)) {
		p->attribute = no;
		return p;
	}

	double max_gain = 0, temp_gain;
	vector<string>::iterator max_it = remain_attribute.begin();
	vector<string>::iterator it1;
	for (it1 = remain_attribute.begin(); it1 < remain_attribute.end(); it1++) {
		temp_gain = ComputeGain(remain_state, (*it1));
		if (temp_gain > max_gain) {
			max_gain = temp_gain;
			max_it = it1;  // 表示最大增益属性
		}
	}

	// 下面根据max_it指向的属性来划分当前样例，更新例集和属性集
	vector<string> new_attribute;
	vector<vector<string>> new_state;
	for (vector<string>::iterator it2 = remain_attribute.begin();
		it2 < remain_attribute.end(); it2++) {
		if ((*it2).compare(*max_it)) new_attribute.push_back(*it2);
	}

	p->attribute = *max_it;
	vector<string> values = map_attribute_values[*max_it];
	int attribute_num = FindAttriNumByName(*max_it);
	new_state.push_back(attribute_row);
	for (vector<string>::iterator it3 = values.begin(); it3 < values.end(); it3++) {
		for (unsigned int i = 1; i < remain_state.size(); i++) {
			if (!remain_state[i][attribute_num].compare(*it3)) {
				new_state.push_back(remain_state[i]);
			}
		}
		Node * new_node = new Node();
		new_node->arrived_value = *it3;
		if (new_state.size() == 0) {
			new_node->arrived_value = MostCommonLabel(remain_state);
		}
		else
			BulidDecisionTreeDFS(new_node, new_state, new_attribute);

		p->childs.push_back(new_node);
		new_state.erase(new_state.begin() + 1, new_state.end());
	}

	return p;
}

void DecisionTree::Input() {
	string s;
	while (cin >> s, s.compare("end") != 0) {//-1为输入结束  
		item[0] = s;
		for (int i = 1; i < MAXLEN; i++) {
			cin >> item[i];
		}
		state.push_back(item);//注意首行信息也输入进去，即属性  
	}
	for (int j = 0; j < MAXLEN; j++) {
		attribute_row.push_back(state[0][j]);
	}
}

void DecisionTree::PrintTree(Node *p, int depth) {
	for (int i = 0; i < depth; i++) cout << '\t';//按照树的深度先输出tab  
	if (!p->arrived_value.empty()) {
		cout << p->arrived_value << endl;
		for (int i = 0; i < depth + 1; i++) cout << '\t';//按照树的深度先输出tab  
	}
	cout << p->attribute << endl;
	for (vector<Node*>::iterator it = p->childs.begin(); it != p->childs.end(); it++) {
		PrintTree(*it, depth + 1);
	}
}

void DecisionTree::FreeTree(Node *p) {
	if (p == NULL)
		return;
	for (vector<Node*>::iterator it = p->childs.begin(); it != p->childs.end(); it++) {
		FreeTree(*it);
	}
	delete p;
	tree_size++;
}