#include <iostream> 
#include <iterator> 
#include <map> 
#include <cmath> 
#include <vector>


using namespace std;
/*#####################################################*/
class sommet{
public:
float abs;
float ord;
sommet(){};
sommet(float b,float a){ abs =b ; ord = a;
};
friend ostream& operator<<(ostream& , const sommet& );
friend bool operator!= (const sommet&,const sommet&);
};

bool operator==(const sommet& d1,const sommet& d2){
    if(d1.abs!=d2.abs){return false;}
    if(d1.ord!=d2.ord){return false;}
   
    return true;
}
ostream& operator<<(ostream& flux, const sommet& s){
    flux<<"ord:"<<s.ord<<"/ abs"<<s.abs<<endl;
    return flux;
}
/*#####################################################*/
class segment{
	public :
		sommet start , end;
		segment(){
		}
		
		segment(sommet A , sommet B){
			start.abs = A.abs;
			start.ord = A.ord;
			end.abs=B.abs;
			end.ord =B.ord;
		}
	float length(); 
	bool intersect(segment);
	friend ostream& operator<<(ostream& , const segment& );
	
};
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
{	if((start==s2.start)||(end==s2.end)||(start==s2.end)||(end==s2.start))
{
return 0;}
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
/*#####################################################*/

class obstacle{
	public :
		int ns;
		vector<sommet> Listsommet;
		vector<segment> Tab;
		obstacle(){
		};
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
obstacle::obstacle(vector <sommet> T)
{
	Listsommet=T;
	ns=T.size();
for(int i=0;i<ns;i++)
{
	Tab.push_back(segment(T[i],T[i+1]));
};
};

/*#####################################################*/
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
/*#####################################################*/
class graph{
	public :
		
		vector <arc> A ;
		graph();
		graph(vector <sommet>);
		
		void confirmGraph();
};
void graph::confirmGraph(){
	
	for(int i=0;i<A.size()-1;i++)
	{
		cout<<"arc numero"<<i+1<<"\n"<<A[i]<<endl;
	};
}
bool test_intersect(sommet S1,sommet S2,vector <sommet> A1){
	int n=A1.size();
	int j=0;
	segment s(S1,S2);
	for(int i=0;i<n;i++)
	{
		segment s2(A1[i],A1[i+1]);
		if(s.intersect(s2)==1)
		{
			return 1;
		};
	};
	return 0;
};
graph::graph(vector <sommet> V)
{
	int size=V.size();
	int i;
	for(i=0;i<size-1;i++)
	{
		
		
	for(int j=0;j<size-1;j++)
	{
			
		A.push_back(arc(V[i],V[j+1]));
	}
	}
/*	int k=0;
	for(i=0;i<size-1;i++)
	{
		for(int j=0;j<size-1;j++)
		{
			
			if(A[i].seg.intersect(A[j].seg))
			{
				k+=1;
			};
		};
		if(k>0)
		{
			A.erase(A.begin()+i);
		};
	}*/
}
/*#####################################################*/
graph cleanGraph(vector<obstacle> O,graph G)
{
	int k=0;
	for(int i=0;i<G.A.size();i++)
	{
		for(int j=0;j<G.A.size();j++)
		{
			for(int m=0;m<O[j].ns;m++)
			{
			
			if(G.A[i].seg.intersect(O[j].Tab[m]))
			{
				k+=1;
			};
		}
	}
		if(k>0)
		{
			G.A.erase(G.A.begin()+i);
		};
	}
}
/*#####################################################*/
sommet determinIntersect(segment S1,segment S2)
{
	float slope1 ,slope2 ,intercept1,intercept2;
	slope1=(S1.start.abs-S1.end.abs)/(S1.start.ord-S1.end.ord);
	slope2=(S2.start.abs-S2.end.abs)/(S2.start.ord-S2.end.ord);
	intercept1=S1.start.ord-slope1*S1.start.abs;
	intercept2=S2.start.ord-slope1*S2.start.abs;
	float x=(intercept1-intercept2)/(slope2-slope1);
	float y=slope1*x+intercept1;
	sommet s(x,y);
	return s;

	
}
/*#############*/
segment normale(segment seg)
{
    float a;
    float b;
    float c;
    float d;
    b= -(seg.end.abs-seg.start.abs)/seg.length();
    a=(seg.end.ord-seg.start.ord)/seg.length();
    c=(seg.end.abs+seg.start.abs)/2;
    d=(seg.end.ord+seg.start.ord)/2;
    sommet s1=sommet(c,d);
    sommet s2=sommet(c+a,d+b);
    segment s=segment(s1,s2);
    return(s);
}
obstacle padding(obstacle objet,obstacle obst)
{
    obstacle res;
    int n=obst.ns;
    res.ns=n;
    /* gestion du premier sommet à  part */
    sommet som_1=obst.Listsommet[0];
    segment nor1=normale(obst.Tab[0]);
    segment nor2=normale(obst.Tab[n-1]);
    float a=(nor1.end.abs-nor1.start.abs);
    float b=(nor1.end.ord-nor1.start.ord);
    float c=(nor2.end.abs-nor2.start.abs);
    float d=(nor2.end.ord-nor2.start.ord);
    int k=0;
    float min=objet.Listsommet[0].abs;
    float max=objet.Listsommet[0].abs;
    float abs=objet.Listsommet[0].abs;
    for(k=0;k<objet.ns;k++){
    	segment seg=objet.Tab[k];
        float x;
            
        x= -a*(seg.end.abs-seg.start.abs)+b*(seg.end.ord-seg.start.ord);
            
        abs=abs+x;
            
        if(abs<min)
			{
			min=abs;
			}
        if(abs>max)
			{
			max=abs;
			}
        }
    float taille1=(max-min);
    k=0;
    min=objet.Listsommet[0].abs;
    max=objet.Listsommet[0].abs;
    abs=objet.Listsommet[0].abs;
    for(k=0;k<objet.ns;k++){
        segment seg=objet.Tab[k];
        float x;
        x= -c*(seg.end.abs-seg.start.abs)+d*(seg.end.ord-seg.start.ord);
        abs=abs+x;
        if(abs<min)
		{
			min=abs;
		}
        if(abs>max)
		{
			max=abs;
		}
        }
    float taille2=(max-min);
    sommet nouveau_sommet=sommet(som_1.abs+a*taille1+c*taille2,som_1.ord+b*taille1+d*taille2);
    res.Listsommet.push_back(nouveau_sommet);
    /* reste des sommets */
    int j=1;
    for(j=1;j<obst.ns;j++){
        sommet som_1=obst.Listsommet[j];
        segment nor1=normale(obst.Tab[j]);
        segment nor2=normale(obst.Tab[(j-1)%n]);
        float a=(nor1.end.abs-nor1.start.abs);
        float b=(nor1.end.ord-nor1.start.ord);
        float c=(nor2.end.abs-nor2.start.abs);
        float d=(nor2.end.ord-nor2.start.ord);
        int k=0;
        float min=objet.Listsommet[0].abs;
        float max=objet.Listsommet[0].abs;
        float abs=objet.Listsommet[0].abs;
        for(k=0;k<objet.ns;k++){
            segment seg=objet.Tab[k];
            float x;
            
            x= -a*(seg.end.abs-seg.start.abs)+b*(seg.end.ord-seg.start.ord);
            
            abs=abs+x;
            
            if(abs<min)
			{
			min=abs;
			}
            if(abs>max)
			{
			max=abs;
			}
        }
        float taille1=(max-min);
        k=0;
        min=objet.Listsommet[0].abs;
        max=objet.Listsommet[0].abs;
        abs=objet.Listsommet[0].abs;
        for(k=0;k<objet.ns;k++){
            segment seg=objet.Tab[k];
            float x;
            x= -c*(seg.end.abs-seg.start.abs)+d*(seg.end.ord-seg.start.ord);
            abs=abs+x;
            if(abs<min)
			{
			min=abs;
			}
            if(abs>max)
			{
			max=abs;
			}
        }
        float taille2=(max-min);
        sommet nouveau_sommet=sommet(som_1.abs+a*taille1+c*taille2,som_1.ord+b*taille1+d*taille2);
         res.Listsommet.push_back(nouveau_sommet);
    }
    
    int i=0;
    for(i=0;i<res.ns;i++)
{

	res.Tab.push_back(segment(res.Listsommet[i],res.Listsommet[i+1]));
}
    return(res);
}
/*#####################################*/
int main(){
	vector<sommet> T1;
	T1.push_back(sommet(0,0));
	T1.push_back(sommet(1,0));
	T1.push_back(sommet(1,1));
	T1.push_back(sommet(0,1));
	obstacle test1=obstacle(T1);
	vector<sommet> T2;
	T2.push_back(sommet(0,0));
	T2.push_back(sommet(1,-1));
	T2.push_back(sommet(2,0));
	T2.push_back(sommet(1,1));
	obstacle test2=obstacle(T2);
	obstacle test3=padding(test1,test2);
	test3.confirm();
	return(0);
}
