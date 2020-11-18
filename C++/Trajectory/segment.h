#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>
#include "sommet.h"
 

using namespace std;



class segment{
	public :
		sommet start , end; //left-right
		segment();
		segment(sommet A , sommet B){
			start.abs = A.abs;
			start.ord = A.ord;
			end.abs=B.abs;
			end.ord =B.ord;
			
		}
	friend bool operator!= (const segment&,const segment&);
	friend bool operator== (const segment&,const segment&);
	float length(); 
	bool intersect(segment);
	friend ostream& operator<<(ostream& , const segment& );
	
};
