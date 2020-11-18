param n;
set T={1..n};
param R;
param m;
param cord{i in T, j in 1..2};
param reg{i in T};
param Amin;
param dis{i in T, j in T} := round(sqrt((cord[j,2]-cord[i,2])**2 + (cord[j,1]-cord[i,1])**2));
param dep;
param ar;
set s_t={dep,ar};
param P_prime integer >=0 ;
param a{1..P_prime, 1..n} integer >= 0;

#-----------------------------------------------------------
# Probleme maitre
#-----------------------------------------------------------

var x{1..n, 1..n} binary;
minimize z : sum{i in T, j in T} dis[i,j]*x[i,j];
subject to
NoInnerLoop: sum{i in T} x[i,i] = 0; 
v_min{i in T, j in T} : dis[i,j]*x[i,j]<=R;
nbrMinAero : sum{i in T, j in T} x[i,j] >= Amin - 1;
NodeToNode{j in T : j not in s_t} : sum{i in T} x[i,j] = sum{i in T} x[j,i];
RegionVist{k in 1..m} : sum{i in T, j in T : reg[i]=k} x[i,j] >= 1;

sous_tour{k in 1..P_prime} : sum{i in T, j in T} a[k,i]*a[k,j]*x[j,i] <= sum{j in T} a[k,j] - 1;

depart: sum{i in T} x[i,dep] = 0;
arrivee : sum{i in T} x[i,ar] = 1;
arrivee2 : sum{i in T} x[ar,i] = 0;
depart2: sum{i in T} x[dep,i] = 1;


#--------------------------------------------------------------
# Sous probleme
#--------------------------------------------------------------
param w {1..n, 1..n} default 0.0;
var y{1..n, 1..n} binary;
maximize st: sum{i in 1..n, j in 1..n} w[i,j]*y[i,j] - sum{i in T} y[i,i] + 1;
subject to
sub1{i in T, j in T} : y[i,j] <= y[i,i];
sub2{i in T, j in T} : y[i,j] = y[j,i];
sub3{i in T, j in T} : y[i,j] >= y[i,i] + y[j,j] - 1;
sub4: sum{i in T} y[i,i] >= 2;
sub5: sum{i in T} y[i,i] <= n-2;