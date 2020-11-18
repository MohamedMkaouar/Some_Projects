#include <iostream> 
#include <iterator> 
#include <map> 
 #include <cmath> 
 #include <vector>
 #include <limits.h> 
 #include <stdio.h>
 #include <fstream> 
 #include<omp.h>


 

using namespace std;
/*#####################################################*/
class sommet{
public:
float abs; //template
float ord;
sommet();
sommet(float b,float a){ abs =b ; ord = a;
};
friend ostream& operator<<(ostream& , const sommet& );
friend bool operator!= (const sommet&,const sommet&);
friend bool operator== (const sommet&,const sommet&);
};

bool operator!=(const sommet& d1,const sommet& d2){
    if(d1.abs!=d2.abs){return true;}
    if(d1.ord!=d2.ord){return true;}
   
    return false;
}
bool operator==(const sommet& d1,const sommet& d2){
    if(d1.abs!=d2.abs){return false;}
    if(d1.ord!=d2.ord){return false;}
   
    return true;
}
ostream& operator<<(ostream& flux, const sommet& s){
    flux<<"ord:"<<s.ord<<"/ abs"<<s.abs<<endl;
    return flux;
}
