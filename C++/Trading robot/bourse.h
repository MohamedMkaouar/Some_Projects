#ifndef BOURSE_H_INCLUDED
#define BOURSE_H_INCLUDED

#include<vector>
#include<map>
#include"PrixJour.h"
#include"date.h"

//SATTAR ALLAHHHH

using namespace std;
class Bourse{
    private:
        string name;
    public:
        Bourse();
        Bourse(string);
        ~Bourse();
        string getName()const;
        void setName(string);
        virtual bool open(Date)const=0;
        virtual void addDailyPrice(PrixJour&)=0;
        virtual void removeDailyPrice(PrixJour&)=0;
        virtual vector<PrixJour> getDayPrice(Date)const=0;
        virtual vector<PrixJour> getAllHistoric(Date)const=0;
        virtual vector<PrixJour> getCompanyHistoric(string,Date)const=0;
        virtual PrixJour getDayCompanyPrice(string,Date)const =0;
        virtual vector<string> getCompanyNames(Date) const=0;
        virtual void clear()=0;
        virtual Date closure()const=0;
        virtual Date opening()const=0;
};


class BourseTab : public Bourse{
    private:
        vector<PrixJour> database;
    public:
       BourseTab();
       BourseTab(string);
       ~BourseTab();
       void addDailyPrice(PrixJour&) ;
       void removeDailyPrice(PrixJour&);
       vector<PrixJour> getDayPrice(Date)const;
       vector<PrixJour> getAllHistoric(Date)const;
       vector<PrixJour> getCompanyHistoric(string,Date)const;
       PrixJour getDayCompanyPrice(string,Date)const;
       vector<string> getCompanyNames(Date) const;
       bool open(Date)const;
       void clear();
       Date closure()const;
       Date opening()const;
};

class BourseMapD : public Bourse{
    private:
        map<Date,vector<PrixJour> > database;
    public:
       BourseMapD();
       BourseMapD(string);
       ~BourseMapD();
       void addDailyPrice(PrixJour&);
       void removeDailyPrice(PrixJour&);
       vector<PrixJour> getDayPrice(Date)const;
       vector<PrixJour> getAllHistoric(Date)const;
       vector<PrixJour> getCompanyHistoric(string,Date)const;
       PrixJour getDayCompanyPrice(string ,Date)const;
       vector<string> getCompanyNames(Date) const;
       bool open(Date)const;
       void clear();
       Date closure()const;
       Date opening()const;
};

#endif // BOURSE_H_INCLUDED


