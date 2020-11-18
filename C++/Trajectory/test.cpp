#include <iostream> 
#include <iterator> 
#include <map> 
#include "functions.h"

#include <cmath>
#define _USE_MATH_DEFINES
#include <limits.h> 
#include <fstream>
#include <chrono> 
#include<omp.h>
#include <string>
sommet::sommet(){
};
obstacle::obstacle(){
};
segment::segment()
{
};
graph::graph()
{
};
arc::arc(){
};
using namespace std;
using namespace std::chrono; 
int main()
{
	 auto start = high_resolution_clock::now(); 
	const double pi = 3.14159;
	
cout<<"*********************************************************************************************"<<"\n";
cout<<"*                                                                                             *"<<"\n";
cout<<"*	 ******    *        ******   *      *   *******  ******     ******   *********        *"<<"\n";
cout<<"*	 *    *    *        *    *   * *    *      *     *    *     *    *   *                *"<<"\n";
cout<<"*	 ******    *        ******   *  *   *      *     ******     ******   *                *"<<"\n";
cout<<"*	 *         *        *    *   *   *  *      *     *     *    *    *   *                *"<<"\n";
cout<<"*	 *         *        *    *   *    * *      *     *      *   *    *   *                *"<<"\n";
cout<<"*	 *         *******  *    *   *      *      *     *       *  *    *   **********       *"<<"\n";
cout<<"*********************************************************************************************"<<"\n";
cout<<"                   By : Bechir Trablesi , Martin Rouy , Aya slimen "<<endl;
int nnn;
cout<<"\n";
cout<<"\n";
cout<<"Welcome to PlanTrac V1.0 our trajectory planning app !!! "<<endl;
cout<<"press any number to continue"<<endl;
cin>>nnn;
//sommet A(1,1) ,B(2,1),C(2,2),D(1,2),A1(1.5,1.5),A2(0,1),E(0,1.5),G(-1,-2),F(-4,2),H(3,1.5),I(5,1.5),J(6,2.5),K(5,3.5),L(4,3.5),N(3,2.5),St(-2,-4),En(7,7);


//sommet A(1,0) ,B(3,0),C(3,3),D(1,3),A1(2,2),A2(0,1),E(4,4),F(5,4),G(4.5,5),H(2,2),St(-2,-4),En(7,7);


string str,str1,str2,str3;
cout<<"enter the folder path in which you want to store the adjency matrix and the X Y coordinates : please use this format => C:\\\\Path\\\\path ( double \\)"<<endl;
cin>>str;
str1=str+"\\Matrice.txt";
str2=str+"\\Xcoord.txt";
str3=str+"\\Ycoord.txt";
//vector <sommet> T{St,A,B,C,A1,D,E,G,F,H,I,J,K,L,N,En};
//ector <sommet> T{St,A,B,C,D,E,G,F,H,En};
system("cls");
vector <sommet> T;
vector<vector<sommet>> Ti;
vector<sommet> Temp;
int nob;
int nv;
cout<<"enter number of obstacles \n";
cin>>nob;
system("cls");
sommet auxil;
obstacle obaux;
int m,l;
vector<int> sizeob;
cout<<"enter the coordinates of the start point "<<endl;
	cout<<"enter abs"<<"\n";
		cin>>auxil.abs;
		cout<<"enter ord"<<"\n";
		cin>>auxil.ord;
T.push_back(auxil);

while(1)

{	system("cls");
cout<<"Constructing obstacle :"<<endl;
while(1)
{
	
	cout<<"enter vertex:"<<endl;
	cout<<"enter abs"<<"\n";
		cin>>auxil.abs;
		cout<<"enter ord"<<"\n";
		cin>>auxil.ord;
	Temp.push_back(auxil);
	T.push_back(auxil);
	cout<<"============> press 1 if next vertex is in the same obstacle else press 0 <============="<<endl;
	cin>>m;
	if(m==0)
	{
		Ti.push_back(Temp);
		Temp.clear();
		m=9999;
		break;
	}
}
cout<<"--------------------moving to next obstacle : press 0 if no more obstacles are present------------------"<<endl;
cin>>l;
if(l==0)
{
	break;
}
}

/*for(int i=0;i<nob;i++)
{
	cout<<"enter number of verticies for obstacle "<<i+1<<"\n";
	cin>>nv;
	for(int j=0;j<nv;j++)
	{
		cout<<"vertex "<<j+1<<" for obstacle "<<i+1<<"\n";
		cout<<"enter abs"<<"\n";
		cin>>auxil.abs;
		cout<<"enter ord"<<"\n";
		cin>>auxil.ord;
		sizeob.push_back(nv);
		T.push_back(auxil);
	}
	
}*/
cout<<"enter the coordinates of the arrival point "<<endl;
	cout<<"enter abs"<<"\n";
		cin>>auxil.abs;
		cout<<"enter ord"<<"\n";
		cin>>auxil.ord;
T.push_back(auxil);		
//vector<sommet> Li;
vector <obstacle> O;


vector<sommet> St{T[0]};
obstacle a(St);
O.push_back(a);

for(int i=0;i<Ti.size();i++)
{	

	obstacle aux(Ti[i]);
	O.push_back(aux);
	
}
vector<sommet> En{T[T.size()-1]};
obstacle b(En);
O.push_back(b);


/*
vector<sommet> T1{A,B,C,A1,D};
vector<sommet> T2{E,F,G};
vector<sommet> T3{H,I,J,K,L,N};
vector<sommet> T4{St};
vector<sommet> T5{En};
obstacle o1(T1);
obstacle o2(T2);
obstacle o3(T3);
obstacle o4(T4);
obstacle o5(T5);
vector<obstacle> O{o1,o2,o3,o4,o5};
*/


system("cls");

graph g(T);
graph g_new;

g_new=CleanGraph(g,T,O);


vector<vector<float>> M;
//T.push_back(St);
//T.push_back(En);
M=g_new.ComputeMatrix(T);



vector<float> X;
vector<float> Y;
for(int i=0;i<T.size();i++)
{
	X.push_back(T[i].abs);
	Y.push_back(T[i].ord);
}

ofstream fileX(str2);
for(int i=0;i<X.size();i++)
{
	fileX<<X[i]<<" ";
}

fileX.close();


ofstream fileY(str3);
for(int i=0;i<Y.size();i++)
{
	fileY<<Y[i]<<" ";
}

fileY.close();



ofstream file(str1);
for(int i=0;i<M.size();i++)
{
	for(int j=0;j<M.size();j++)
	{	if(M[i][j]!=0)
	{
	
		file<<M[i][j]<<" ";
	}
	else{
	
		file<<"0"<<" ";
		
	}
	}
	file<<" \n";
}

file.close();

g_new.ShortestPath(T,0,0);
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start); 
  
// To get the value of duration use the count() 
// member function on the duration object 


cout << "\n duration = "<<duration.count() <<" microseconds" << endl; 
cout<<" =================================>press any key to exist<=========================  "<<endl;
cin>>l;
system("cls");
cout<<" ===============================> Stay home Stay Safe <================================"<<endl; 



return 1;	
}
