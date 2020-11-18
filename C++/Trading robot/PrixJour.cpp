#include"PrixJour.h"
#include<iostream>
PrixJour::PrixJour(){};
PrixJour::PrixJour(string n,double p , Date d){
    name=n;
    price=p;
    day=d;
}
string PrixJour::getName ()const{return name ;}
double PrixJour::getPrice ()const{return price ;}
Date PrixJour::getDay ()const{return day;}
void PrixJour::setName (string str){ name=str;}
void PrixJour::setPrice(double d){price=d;}
void PrixJour::setDay(Date d){day=d;}
PrixJour PrixJour::operator=(const PrixJour& pj){
    name=pj.getName();
    price=pj.getPrice();
    day=pj.getDay();
    return(*this);
}
 bool operator< (const PrixJour& pj1,const PrixJour& pj2){
    if(pj1.getPrice()<pj2.getPrice()){return true;}
    else return false;
}
 bool operator<= (const PrixJour& pj1,const PrixJour& pj2){
    if(pj1.getPrice()<=pj2.getPrice()){return true;}
    else return false;
}
 bool operator== (const PrixJour& pj1,const PrixJour& pj2){
    if(pj1.getPrice()==pj2.getPrice()&&pj1.getName()==pj2.getName()&&pj1.getDay()==pj2.getDay()){return true;}
    else return false;
}
bool operator> (const PrixJour& pj1,const PrixJour& pj2){
    if(pj1.getPrice()>pj2.getPrice()){return true;}
    else return false;
}
 bool operator>= (const PrixJour& pj1,const PrixJour& pj2){
    if(pj1.getPrice()>=pj2.getPrice()){return true;}
    else return false;
}
 bool operator!= (const PrixJour& pj1,const PrixJour& pj2){
    if(pj1.getPrice()!=pj2.getPrice()&&pj1.getName()!=pj2.getName()&&pj1.getDay()!=pj2.getDay()){return true;}
    else return false;
}
ostream& operator<<(ostream& flux, const PrixJour p){
    flux<<"le prix de l'entreprise "<<p.getName()<<" le jour "<<p.getDay()<<" est "<<p.getPrice();
    return flux;
}
