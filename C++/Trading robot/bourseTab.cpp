#include"bourse.h"



//parite bourseTab
BourseTab::BourseTab():Bourse(){};
BourseTab::BourseTab(string str):Bourse(str){};
BourseTab::~BourseTab(){
    database.clear();
    cout<<" suppression avec réussite"<<endl;
}
bool BourseTab::open(Date d)const{
    vector<PrixJour>::const_iterator i =database.begin();
    while(i!=database.end()){
        if(i->getDay()==d){return (true) ;}
        ++i;
    }
    return(false);
}
void BourseTab::addDailyPrice(PrixJour& pj) {
    database.push_back(pj);
}
void BourseTab::removeDailyPrice(PrixJour& pj){
    vector<PrixJour>::iterator i = database.begin();
    while(i!=database.end()){
        if((*i)==pj){i=database.erase(i);break;}
        ++i;
    }
}
vector<PrixJour> BourseTab::getAllHistoric(Date d)const{
    vector<PrixJour> result;
    vector<PrixJour>::const_iterator i =database.begin();
    while(i!=database.end()){
        if((*i).getDay()<=d){result.push_back(*i);}
        ++i;
    }
    return result;
}
vector<PrixJour> BourseTab::getCompanyHistoric(string str,Date d) const{
    vector<PrixJour> result;
    vector<PrixJour>::const_iterator i =database.begin();
    while(i!=database.end()){
        if(i->getName()==str&&i->getDay()<=d){result.push_back(*i);}
        ++i;
    }
    return result;
}
vector<PrixJour> BourseTab::getDayPrice(Date d)const{
    vector<PrixJour> result;
    vector<PrixJour>::const_iterator i =database.begin();
    while(i!=database.end()){
        if((*i).getDay()==d){result.push_back(*i);}
        ++i;
    }
    return result;
}
PrixJour BourseTab::getDayCompanyPrice(string str, Date d)const{
    vector<PrixJour>::const_iterator i =database.begin();
    while(i!=database.end()){
        if(i->getName()==str&&i->getDay()==d){break;}
        ++i;
    }
    return(*i);
}
vector<string> BourseTab::getCompanyNames(Date d) const{
    vector<string> result;
    vector<PrixJour>::const_iterator i =database.begin();
    while(i!=database.end()){
        if((*i).getDay()==d){result.push_back(i->getName());}
        ++i;
    }
    return result;
}
void BourseTab::clear(){
    database.clear();
}
Date BourseTab::closure()const{
    vector<PrixJour>::const_iterator i =database.begin();
    Date F=Date(0,0,0);
    while(i!=database.end()){
        if(F<i->getDay()){F=i->getDay();}
        i++;
    }
    return(F);
}
Date BourseTab::opening()const{
    vector<PrixJour>::const_iterator i =database.begin();
    Date F =i->getDay();
    while(i!=database.end()){
        if(F>i->getDay()){F=i->getDay();}
        i++;
    }
    return(F);
}


