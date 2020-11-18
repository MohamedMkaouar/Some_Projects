#ifndef ROBOT_H_INCLUDED
#define ROBOT_H_INCLUDED

#include<iostream>
#include<time.h>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <cmath>
//#include <random>
#include"transaction.h"
#include"Portefeuille.h"
#include"bourse.h"

class Robot {
    protected:
        Portefeuille wallet;
        string name;
    public:
        Robot();
        Robot(string);
        string getName()const;
        vector<Titre> getActions()const;
        float getBalance()const;
        Portefeuille getWallet()const;
        void setBalance(float);
        virtual Transaction Decide(const Bourse*,const Date)=0;
        float fortune(Bourse*,Date); //permet d'avoir la fortune du robot
        bool possesAction(string,int); //vérifie l'exsitance d'un élément dans son portefeuille
        void sellAction(string,int);// à spécifier
        void buyAction(string,int);// à spécifier
        void payment(float);//fonction lors du payment
        void receipt(float);// ............et la vente.
        int getsize()const{return wallet.getSize();}
};


class RobotRandom: public Robot{
    public :
        RobotRandom();
        RobotRandom(string);
        Transaction Decide(const Bourse*,const Date);

};
class RobotSmart: public Robot{
    private :
        map<string, vector<float> > memory ;
        Date* dateMemo ;
    public:
        RobotSmart();
        RobotSmart(string);
        void memorySet(const Bourse*,const Date);
        Transaction Decide(const Bourse*,const Date);
};

#endif // ROBOT_H_INCLUDED
