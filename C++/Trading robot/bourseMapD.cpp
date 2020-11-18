#include"bourse.h"


//Partie bourseMapD
BourseMapD::BourseMapD():Bourse(){};
BourseMapD::BourseMapD(string str):Bourse(str){};
BourseMapD::~BourseMapD(){
    database.clear();
    cout<<" suppression avec réussite"<<endl;
}
bool BourseMapD::open(Date d)const{
    int c = database.count(d);
    if (c!=0){return true; }
    else return(false);
}
void BourseMapD::addDailyPrice(PrixJour& pj){
    database[pj.getDay()].push_back(pj);
}
void BourseMapD::removeDailyPrice(PrixJour& pj){
    vector<PrixJour> P = database[pj.getDay()];
    vector<PrixJour>::iterator i = P.begin();
    while(i!=P.end()){
        if((*i)==pj){i=database[pj.getDay()].erase(i);break;}
    }
}
vector<PrixJour> BourseMapD::getDayPrice(Date d)const{
    return(database.at(d));
}
vector<PrixJour> BourseMapD::getAllHistoric(Date d)const{
    vector<PrixJour>result ;
    map<Date,vector<PrixJour> >::const_iterator i = database.begin();
    while(i!=database.end()){
        if(i->first<=d){result.insert(result.end(),i->second.begin(),i->second.end());}
        i++;
    }
    return result;
}
vector<PrixJour> BourseMapD::getCompanyHistoric(string str ,Date d)const{
    vector<PrixJour>result ;
    map<Date,vector<PrixJour> >::const_iterator i = database.begin();
    vector<PrixJour>::const_iterator j;
    while(i!=database.end()){
        if(i->first<=d){
                j=i->second.begin();
                while(j!=i->second.end()){
                        if(j->getName()==str){result.push_back(*j);}
                        j++;
                }
        }
        ++i;
    }
    return result;
}
PrixJour BourseMapD::getDayCompanyPrice(string str,Date d)const{
    vector<PrixJour>::const_iterator i=database.at(d).begin();
    while(i!=database.at(d).end()){
        if (i->getName()==str) return(*i);
        i++;
    }
    PrixJour pj("empty",0,Date(0,0,0)) ;
    return(pj);
}
vector<string> BourseMapD::getCompanyNames(Date d) const {
    vector<PrixJour>::const_iterator pj =database.at(d).begin();
    vector<string> result;
    while(pj!=database.at(d).end()){
        result.push_back(pj->getName());
        ++pj;
    }
    return(result);

}
void BourseMapD::clear(){
    database.clear();
}
Date BourseMapD::closure()const{
    map<Date, vector<PrixJour> >::const_iterator i = database.begin();
    Date F= Date(0,0,0);
    while(i!=database.end()){
            if(F<i->first){F=i->first;}
            i++;
    }
    return(F);

}
Date BourseMapD::opening()const{
    map<Date, vector<PrixJour> >::const_iterator i = database.begin();
    Date F= i->first;
    while(i!=database.end()){
            if(F>i->first){F=i->first;}
            i++;
    }
    return(F);

}

