#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>
#include "obstacle.h"
 

using namespace std;

obstacle::obstacle(vector <sommet> T)
{
	Listsommet=T;
	ns=T.size();
for(int i=0;i<ns;i++)
{	if(i<ns-1)
{

	Tab.push_back(segment(T[i],T[i+1]));
}
	if(i==ns-1)
	{
		Tab.push_back(segment(T[i],T[0]));
	}
};
};
