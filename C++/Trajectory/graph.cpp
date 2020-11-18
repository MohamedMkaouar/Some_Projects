#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>

#include "graph.h" 

using namespace std;
void graph::confirmGraph(){
	
	for(int i=0;i<A.size();i++)
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
	for(i=0;i<size;i++)
	{
		
		
	for(int j=0;j<size;j++)
	{
		
		A.push_back(arc(V[i],V[j]));
	}

	}

};

float graph::traj(sommet C,sommet B){
	
		for(int i=0;i<A.size();i++)
		{
			if(((A[i].start==C)&&(A[i].end==B))||((A[i].start==B)&&(A[i].end==C)))
			{
				return A[i].cost;
			}
		}
		return 0;





}

vector<vector<float>> graph::ComputeMatrix(vector<sommet> S){
	vector<float>L;
	vector<vector<float>> M;
#pragma omp parallel for	
for(int i=0;i<S.size();i++)
{	
	for(int j=0;j<S.size();j++)
	{
			//cout<<T[i]<<T[j]<<endl;
			//cout<<traj(T[i],T[j],g_new);
			L.push_back(traj(S[i],S[j]));
			//cout<<"L="<<L[j]<<endl;
			
	}
	//cout<<"size of L"<<L.size()<<endl;
	M.push_back(L);
	L.clear();
}
return M;
}

void graph::ShortestPath(vector<sommet> T,int startnode,int n) {
	vector<vector<float>> G = ComputeMatrix(T);
	int max=G.size();
   float cost[max][max],distance[max],pred[max];
   int visited[max],count;
   float mindistance;
   int nextnode,i,j;
   for(i=0;i<max;i++)
    {
	    for(j=0;j<max;j++)
	    {
		
   			if(G[i][j]==0)
      			cost[i][j]=9999;
  					 else
      			cost[i][j]=G[i][j];
      		}}
   for(i=0;i<max;i++) {
      distance[i]=cost[startnode][i];
      pred[i]=startnode;
      visited[i]=0;
   }
   distance[startnode]=0;
   visited[startnode]=1;
   count=1;
   while(count<max-1) {
      mindistance=9999;
      for(i=0;i<max;i++)
         if(distance[i]<mindistance&&!visited[i]) {
         mindistance=distance[i];
         nextnode=i;
      }
      visited[nextnode]=1;
      for(i=0;i<max;i++)
         if(!visited[i])
      if(mindistance+cost[nextnode][i]<distance[i]) {
         distance[i]=mindistance+cost[nextnode][i];
         pred[i]=nextnode;
      }
      count++;
   }
   for(i=0;i<max;i++)
   if(i!=startnode) {
      cout<<"\nDistance of node"<<i<<"="<<distance[i];
      cout<<"\nPath="<<i;
      j=i;
      do {
         j=pred[j];
         cout<<"<-"<<j;
      }while(j!=startnode);
   }
}
