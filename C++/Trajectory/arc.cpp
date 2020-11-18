#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>

 #include "arc.h"

using namespace std;



ostream& operator<<(ostream& flux, const arc& a){
    flux<<"sommet de depart de l'arc:"<< a.start <<"\n"<<"sommet d'arrivee de l'arc :"<< a.end<<"\n" <<"arc de longeur :"<<a.cost<<endl;
    return flux;
}
arc::arc(sommet A , sommet B)
{
	start = A;
	end = B;
	segment s(A,B);
	seg=s;
	cost=s.length();
};

