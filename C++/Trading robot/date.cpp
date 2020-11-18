#include <iostream>
#include"date.h"
#include <string>
#include <sstream>
#include <stdexcept>
Date::Date(){};
Date::Date(int d, int m , int y){
    day=d;
    month=m;
    year=y;
}
Date::Date(const Date& d ){
    day=d.getDay();
    month=d.getMonth();
    year=d.getYear();
}
int Date::getDay()const{return(day);}
int Date::getMonth()const{return(month);}
int Date::getYear()const{return(year);}
void Date::setDay(int d){day=d;}
void Date::setMonth(int m){month = m;}
void Date::setYear(int y) {year = y;}
void Date::incrementDate(){
    day+=1;
    if (month==2){
        if (year%4!=0 &&day==29){day=1; month=3;}
    }
    switch (month){
        case 1: if (day==32){day=1;month+=1;}break;
        case 2: if (day==30){day=1;month+=1;}break;
        case 3: if (day==32){day=1;month+=1;}break;
        case 4: if (day==31){day=1;month+=1;}break;
        case 5: if (day==32){day=1;month+=1;}break;
        case 6: if (day==31){day=1;month+=1;}break;
        case 7: if (day==32){day=1;month+=1;}break;
        case 8: if (day==32){day=1;month+=1;}break;
        case 9: if (day==31){day=1;month+=1;}break;
        case 10: if (day==32){day=1;month+=1;}break;
        case 11: if (day==31){day=1;month+=1;}break;
        case 12: if (day==32){day=1;month=1;year+=1;}break;

}
}
void Date::decrementDate(){
    day-=1;
    if (month==3){
        if (year%4==0 &&day==0){day=29; month=2;}
        else if(year%4!=0&&day==0){day=28;month=2;}
    }
    switch (month){
        case 1: if (day==0){day=31;month=12,year-=1;}break;
        case 2: if (day==0){day=31;month-=1;}break;
        case 4: if (day==0){day=31;month-=1;}break;
        case 5: if (day==0){day=30;month-=1;}break;
        case 6: if (day==0){day=31;month-=1;}break;
        case 7: if (day==0){day=30;month-=1;}break;
        case 8: if (day==0){day=31;month-=1;}break;
        case 9: if (day==0){day=31;month-=1;}break;
        case 10: if (day==0){day=30;month-=1;}break;
        case 11: if (day==0){day=31;month-=1;}break;
        case 12: if (day==0){day=30;month=11;}break;

}
}
Date Date::operator=(const Date& d){
    day=d.getDay();
    month=d.getMonth();
    year=d.getYear();
    return(*this);
}
bool operator<(const Date& d1,const Date& d2){
    if (d1.getYear()<d2.getYear()){return true;}
    else if (d1.getYear()>d2.getYear()){return false;}
    else {
        if (d1.getMonth()<d2.getMonth()){return true;}
        else if (d1.getMonth()>d2.getMonth()){return false;}
        else {
            if (d1.getDay()<d2.getDay()){return true;}
            else if (d1.getDay()>d2.getDay()){return false;}
            else
                return false;
        }
    }
}
bool operator<=(const Date& d1,const Date& d2){
    if (d1.getYear()<d2.getYear()){return true;}
    else if (d1.getYear()>d2.getYear()){return false;}
    else {
        if (d1.getMonth()<d2.getMonth()){return true;}
        else if (d1.getMonth()>d2.getMonth()){return false;}
        else {
            if (d1.getDay()<d2.getDay()){return true;}
            else if (d1.getDay()>d2.getDay()){return false;}
            else
                return true;
        }
    }
}
bool operator>(const Date& d1,const Date& d2){
    if (d1.getYear()>d2.getYear()){return true;}
    else if (d1.getYear()<d2.getYear()){return false;}
    else {
        if (d1.getMonth()>d2.getMonth()){return true;}
        else if (d1.getMonth()<d2.getMonth()){return false;}
        else {
            if (d1.getDay()>d2.getDay()){return true;}
            else if (d1.getDay()<d2.getDay()){return false;}
            else
                return false;
        }
    }
}
bool operator>=(const Date& d1,const Date& d2){
    if (d1.getYear()>d2.getYear()){return true;}
    else if (d1.getYear()<d2.getYear()){return false;}
    else {
        if (d1.getMonth()>d2.getMonth()){return true;}
        else if (d1.getMonth()<d2.getMonth()){return false;}
        else {
            if (d1.getDay()>d2.getDay()){return true;}
            else if (d1.getDay()<d2.getDay()){return false;}
            else
                return true;
        }
    }
}
bool operator==(const Date& d1,const Date& d2){
    if(d1.getDay()!=d2.getDay()){return false;}
    if(d1.getMonth()!=d2.getMonth()){return false;}
    if(d1.getYear()!=d2.getYear()){return false;}
    return true;
}
bool operator!=(const Date& d1,const Date& d2){
    if(d1.getDay()!=d2.getDay()){return true;}
    if(d1.getMonth()!=d2.getMonth()){return true;}
    if(d1.getYear()!=d2.getYear()){return true;}
    return false;
}
ostream& operator<<(ostream& flux, const Date& dd){
    flux<<dd.day<<"/"<<dd.month<<"/"<<dd.year;
    return flux;
}
istream& operator>>(istream& flux ,Date& dd ){
    string t;
    int day , month , year ;
    getline(flux,t,'/');
    istringstream iss (t);
    iss>>day;
    getline(flux,t,'/');
    istringstream iss2 (t);
    iss2>>month;
    getline(flux,t);
    istringstream iss3 (t);
    iss3>>year;
    if(day<1||day>31){throw invalid_argument( " Invalide date " );}
    if(month<1||month>13){throw invalid_argument( " Invalide date " );}
    if (month==2){
        if (year%4!=0 &&day>28){throw invalid_argument( " Invalide date " );}
        else if(year%4==0 &&day>29){throw invalid_argument( " Invalide date 1" );}
    }
    switch (month){
        case 4: if (day==31){throw invalid_argument( " Invalide date " );}break;
        case 6: if (day==31){throw invalid_argument( " Invalide date " );}break;
        case 9: if (day==31){throw invalid_argument( " Invalide date " );}break;
        case 11: if (day==31){throw invalid_argument( " Invalide date " );}break;
        }
    dd.setDay(day);
    dd.setMonth(month);
    dd.setYear(year);
}
