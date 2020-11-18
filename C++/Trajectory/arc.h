#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>

 #include "obstacle.cpp"

using namespace std;
class arc{
	public :
		sommet start;
		sommet end;
		segment seg;
		float  cost;
		arc();
		arc(sommet,sommet);
		friend ostream& operator<<(ostream& , const arc& );
};
