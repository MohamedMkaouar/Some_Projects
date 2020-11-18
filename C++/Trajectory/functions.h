#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>
#include <algorithm>
#include "graph.cpp"

 

using namespace std;

sommet determinIntersect(segment S1,segment S2)
{
float slope1=(S1.start.ord-S1.end.ord)/(S1.start.abs-S1.end.abs);
float	slope2=(S2.start.ord-S2.end.ord)/(S2.start.abs-S2.end.abs);
float	intercept1=S1.start.ord-slope1*S1.start.abs;
float  intercept2=S2.start.ord-slope2*S2.start.abs;
float x,y;
if(abs(slope1)>10000)
{
	y=slope2*S1.start.abs+intercept2;
}



if(abs(slope2)>10000)
{

	y=slope1*S2.start.abs+intercept1;
}


if(abs(slope1)>10000)
{
	x=S1.start.abs;
}
else if(abs(slope2)>1000)
{
	x=S2.start.abs;
}
else
{

x=(intercept1-intercept2)/(slope2-slope1);
}

if(x<0.001)
{x=0;
}
if(y<0.001)
{y=0;
}
sommet s(x,y);
	return s;

	
}
bool testin(segment S1,sommet s)
{

	float	ps=((s.abs-S1.start.abs)*(-s.abs+S1.end.abs)+((s.ord-S1.start.ord)*(-s.ord+S1.end.ord)));

float norm=sqrt(((s.abs-S1.start.abs)*(s.abs-S1.start.abs)+(s.ord-S1.start.ord)*(s.ord-S1.start.ord))*((-s.abs+S1.end.abs)*(-s.abs+S1.end.abs)+(-s.ord+S1.end.ord)*(-s.ord+S1.end.ord)));

if (ps==norm)
{
	return 1;
}
return 0;
}

bool sinlist(sommet s, vector<segment> L){
	int m=0;
	for(int i=0;i<L.size();i++)
	{
		if(testin(L[i],s)==1)
		{
			m++;
		}
	}
	if(m==0)
	{
		return false;
	}
	return true;
}
sommet barycentre(vector<sommet> S)
{	sommet b;
sommet bary;
	for(int i=0;i<S.size();i++)
	{
		b.abs+=S[i].abs;
		b.ord+=S[i].ord;
		
	}
	bary.abs=b.abs/S.size();
	bary.ord=b.ord/S.size();
	return bary;
	
}

double vectcos(segment s1,segment s2){
	double x1,x2,y1,y2,ps,norm,cosinus;
	x1=s1.end.abs-s1.start.abs;
	x2=s2.end.abs-s2.start.abs;
	y1=s1.end.ord-s1.start.ord;
	y2=s2.end.ord-s2.start.ord;
	ps=x1*x2+y1*y2;
	norm=sqrt((x1*x1+y1*y1)*(x2*x2+y2*y2));
	cosinus=ps/norm;
	return cosinus;
}



double sumofangles(obstacle ob1, sommet mid){
	segment l1,l2;
double c;
for(int i=0;i<ob1.Listsommet.size();i++)
{
	l1.start=mid;
	//cout<<l1.start;
	l1.end=ob1.Listsommet[i];
	//cout<<l1.end;
	l2.start=mid;
	//cout<<l2.start;
	l2.end=ob1.Listsommet[i+1];
	
	if(i==ob1.Listsommet.size()-1)
	{
		l2.end=ob1.Listsommet[0];
	}
	//cout<<l2.end;
	c+=acos(vectcos(l1,l2));
	
	
	
}
return c;
}


sommet midpoint(segment S1)
{
	sommet mid;
mid.abs=(S1.start.abs+S1.end.abs)/2;
mid.ord=(S1.start.ord+S1.end.ord)/2;
return mid;
}

segment inverse(segment s)
{
	segment aux;
	aux.start=s.end;
	aux.end=s.start;
	return aux;
}
bool upgradedintersection(segment S1 , segment S2)
{	segment S11,S22;
	S11=inverse(S2);
	S22=inverse(S2);
	
	
	if(S1.start.ord!=S2.start.ord){

if((((S1.start.ord>S2.start.ord)&&(S1.end.ord<S2.end.ord))||((S1.start.ord<S2.start.ord)&&(S1.end.ord>S2.end.ord)))&&(S1.start!=S2.end)&&(S1.end!=S2.start) )
{
	return 1;
}

else 
{
	return 0 ;
}

}
else
{
	if((((S1.start.abs>S2.start.abs)&&(S1.end.abs<S2.end.abs))||((S1.start.abs<S2.start.abs)&&(S1.end.abs>S2.end.abs)))&&(S1.start!=S2.end)&&(S1.end!=S2.start))
{
	return 1;
}
else 
{
	return 0;
}

}
}



bool seginlist(segment s,vector <segment> L)
{
	int m=0;
	for(int i=0;i<L.size();i++)
	{
		if(s==L[i])
		{
			m++;
		}
	}
	if(m==0)
	{
		return false;
	}
	return true;
}

bool sinlistsom(sommet s,vector <sommet> L)
{
	int m=0;
	for(int i=0;i<L.size();i++)
	{
		if(s==L[i])
		{
			m++;
		}
	}
	if(m==0)
	{
		return false;
	}
	return true;
}


bool isdiag(obstacle ob,segment s)
{
	sommet m;
	int n=ob.Listsommet.size();
	m=midpoint(s);
	if(((sumofangles(ob,m)-(n-2)*pi)<0.001)&&(seginlist(s,ob.Tab)==0)&&(sinlistsom(s.start,ob.Listsommet)==1)&&(sinlistsom(s.end,ob.Listsommet)==1))
	{
		return 1;
	}
	return 0;
}



bool newInter(segment S1,segment S2)
 {	
	float I1[2]={min(S1.start.abs,S1.end.abs),max(S1.start.abs,S1.end.abs)};
	float I2[2]={min(S2.start.abs,S2.end.abs),max(S2.start.abs,S2.end.abs)};
	if((I1[1]<=I2[0])||(I2[1]<=I1[0]))
	{	
			return 0;
		
	}
	float I3[2]={min(S1.start.ord,S1.end.ord),max(S1.start.ord,S1.end.ord)};
	float I4[2]={min(S2.start.ord,S2.end.ord),max(S2.start.ord,S2.end.ord)};
	if((I3[1]<=I4[0])||(I4[1]<=I3[0]))
	{
		return 0;
	}
	
if((S1.start==S2.end)||(S1.end==S2.start))
{ 

return 0;
 }
 if(S1.start==S2.start)
 {

 return 0;
}
 if(S1.end==S2.end)
 {

 return 0;
}


  return 1;
  
}


//##########################################################################################################################################
graph CleanGraph(graph g,vector<sommet> T, vector<obstacle> O){
	vector <segment> Listedges;
	
	#pragma omp parallel for
for(int i=0;i<(int)O.size();i++)
{
	for(int j=0;j<(int)O[i].Tab.size();j++)
	{
		Listedges.push_back(O[i].Tab[j]);
	}
}
cout<<"nbr of edges "<<Listedges.size()<<endl;
/*for(int i=0;i<ob2.Tab.size();i++)
{
	cout<<"@@@@@"<<ob2.Tab[i]<<endl;
}*/

sommet St(-2,-4);
sommet F(3,1);
segment S(St,F);
int m=0;
int n=0;
int indic=0;
sommet H(3,1.5);
sommet I(5,1.5);
segment S1(St,I);

vector <segment> treated;
cout<<"N arc :"<<g.A.size()<<endl;
#pragma omp parallel for
for(int j=0;j<(int)Listedges.size();j++)
{	
	vector<sommet> points;
	for(int i=0;i<(int)g.A.size();i++)
	{	
		if((newInter(g.A[i].seg,Listedges[j])==1)||(newInter(inverse(g.A[i].seg),Listedges[j])==1)||(newInter(g.A[i].seg,inverse(Listedges[j]))==1)||(newInter(inverse(g.A[i].seg),inverse(Listedges[j]))==1))
		{	if (g.A[i].seg==S1)
				{
				cout<<"here ==>"<<Listedges[j];}
			g.A.erase(g.A.begin()+i);
			i--;
				
}
	}
}
cout<<"N arc :"<<indic<<endl;
graph g_new=g;




graph g_final;
	sommet mid;
	vector <segment> list;
	int d=0;
	#pragma omp parallel for
for(int i=0;i<g_new.A.size();i++)
{		//cout<<g_new.A[i].seg;
		//cout<<"\n "<<isdiag(O[0],g_new.A[i].seg)<<"\n";
		for(int j=0;j<O.size();j++)
		{
		
		if((isdiag(O[j],g_new.A[i].seg)==1))
		
		{	//cout<<g_new.A[i];
		g_new.A.erase(g_new.A.begin()+i);
			i--;
			
			
		
		}
		}



	
}
	


vector<segment> Listsegments;
#pragma omp parallel for
for(int i=0;i<g_new.A.size();i++)
{
	Listsegments.push_back(g_new.A[i].seg);
};
#pragma omp parallel for
for(int i=0;i<Listedges.size();i++)

{
		if(seginlist(Listedges[i],Listsegments)==0)
		{	arc aux(Listedges[i].start,Listedges[i].end);
			g_new.A.push_back(aux);
		}
	
}
#pragma omp parallel for
for(int i=0;i<Listedges.size();i++)
{
	arc aux(Listedges[i].start,Listedges[i].end);
	g_new.A.push_back(aux);
}
#pragma omp parallel for
for(int i=0;i<(int)g_new.A.size();i++)
{
	for(int j=0;j<(int)Listedges.size();j++)
	{
			
			if((upgradedintersection(g_new.A[i].seg,Listedges[j])==1)&&(testin(g_new.A[i].seg,determinIntersect(g_new.A[i].seg,Listedges[j]))==1)&&(testin(Listedges[j],determinIntersect(g_new.A[i].seg,Listedges[j]))==1))
		{	cout<<g_new.A[i];
			g_new.A.erase(g_new.A.begin()+i);
			i--;
		}
	}
}








cout<<"final size ="<<g_new.A.size();

return g_new;

}

