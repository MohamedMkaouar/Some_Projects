#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>

#include "segment.h" 
const double pi = 3.14159;
using namespace std;

bool operator!=(const segment& d1,const segment& d2){
    if(d1.start!=d2.start){return true;}
    if(d1.end!=d2.end){return true;}
   
    return false;
}
bool operator==(const segment& d1,const segment& d2){
    if(d1.start==d2.start)
    {
    	if(d1.end==d2.end)
    	{
    		return true;
		}
	}
	if(d1.start==d2.end){
		if(d1.end==d2.start)
		{
			return true;
		}
	}
    
   
    return false;
}
ostream& operator<<(ostream& flux, const segment& seg){
    flux<<"start :"<< seg.start <<"/ end :"<< seg.end <<endl;
    return flux;
}
float segment:: length(){
	float l;
	l=sqrt(pow((start.abs-end.abs),2)+pow((start.ord-end.ord),2));
	return l;
};
bool segment::intersect(segment s2)
{	
	float x1=end.abs-start.abs;
	float y1=end.ord-start.ord;
		float x2=s2.end.abs-s2.start.abs;
	float y2=s2.end.ord-s2.start.ord;
	 double determinant = x1*y2-y1*x2; 
  
    if (determinant == 0) 
    { 
        // The lines are parallel. This is simplified 
        // by returning a pair of FLT_MAX
		 
        return 0;
    } 
    else
    { 
   
        return 1;
    } 
} ;
