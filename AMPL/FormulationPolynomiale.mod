param n;
set T:=1..n;
param Amin;
param dep;
param ar;
param m;
param reg{t in T};
param R;
param l  >= 0;
param dis{t in T,j in T};
param cord{t in T, j in 1..m};
param c:=m-1;
param AIR{i in 1..m, j in 1..n}:= if (reg[j] == i) then 1 else 0 ;
var x{i in T,j in T} binary;
var u{i in T} integer >=0;
set T2:=T diff{dep};
set T3:= T diff{dep,ar};
minimize z: sum{i in 1..n, j in 1..n}(dis[i,j]*x[i,j]);
subject to 
init : u[1]=dep;
v_min {i in 1..n, j in 1..n}: dis[i,j]*x[i,j] <= R;

nbrMinAero: sum{i in 1..n,j in 1..n}(x[i,j]) >= Amin-1;

NodeToNode {j in T3} : sum{i in T}(x[i,j]-x[j,i])==0;

NoInnerLoop {i in 1..n} : x[i,i]==0;

depart : sum{j in 1..n} x[dep,j] ==1;

arrivee : sum{j in 1..n} x[j,ar] ==1;

depart2 : sum{j in 1..n} x[j,dep] ==0;

arrivee2 : sum{j in 1..n} x[ar,j] ==0;

RegionVist {k in 1..m}: sum{i in T , j in T } (AIR[k,i]*x[j,i] + AIR[k,j]*x[i,j]) >= 1 ; 

NoST{i in T2, j in T2} : u[j] >= u[i] + 1 - n*(1-x[i,j]);


