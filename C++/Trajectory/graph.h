#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>

 #include "arc.cpp"

 

using namespace std;

class graph{
	public :
	
		vector <arc> A ;
		graph();
		graph(vector <sommet>);
		float traj(sommet,sommet);
		void virtual ShortestPath(vector<sommet>,int,int);
		vector<vector<float>> ComputeMatrix(vector<sommet>);
		void confirmGraph();
		
};
