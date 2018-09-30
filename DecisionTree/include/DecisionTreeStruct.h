#include <iostream>  
#include <string>  
#include <vector>  
#include <map>  
#include <algorithm>  
#include <cmath>  
using namespace std;


struct Node {//决策树节点  
	string attribute;//属性值  
	string arrived_value;//到达的属性值  
	vector<Node *> childs;//所有的孩子  
	Node() {
		attribute = "";
		arrived_value = "";
	}
};
Node * root;