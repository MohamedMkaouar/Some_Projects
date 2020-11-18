#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>

#include "segment.cpp"
 

using namespace std;


class obstacle{
	public :
		int ns;
		vector<sommet> Listsommet;
		vector<segment> Tab;
		obstacle();
		obstacle(vector <sommet>);
		void confirm(){
			int j;
			for(j=0;j<ns;j++)
			{
				cout <<"sommet "<<j<<":";
				cout<<Listsommet[j]<<endl;
			}
			for(j=0;j<ns;j++)
			{
				cout <<"segment "<<j+1<<":";
				cout<<Tab[j]<<endl;
			}
		}
};
