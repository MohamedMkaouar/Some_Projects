#ifndef SIMULATION_H_INCLUDED
#define SIMULATION_H_INCLUDED
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include"date.h"
#include"robot.h"
#include"bourse.h"
#include"transaction.h"

using namespace std ;
class Simulation {
    private:
        Robot* robot;
        Bourse* bourse;
        Date today ;
        float intialSolde;
        Date* firstDay;
        Date* finalDay;


        void setRobot(int,string);
        void setBourse(int,string);
        void getDataFile(const string&);
        void setInitialSolde(float );
        bool validateTransaction(const Transaction);
        void effectTransaction(Transaction);
    public:
        Simulation();
        Simulation(int,string,int,string,Date);
        Simulation(int,string,int,string,Date,Date);
        string getRobotName()const;
        string getBourseName()const;
        Date getToday()const;
        float getIntialSolde()const;
        vector<Date> getSimualtionInterval()const;
        float getRobotFortune()const;
        float getRobotBalance()const;
        vector<Titre> getRobotActions()const;
        Portefeuille getRobotWallet()const;
        void setInitialvalues(float);
        void ExcuteSim();
        void setDatasource(int ,const string&);
        vector<PrixJour> getDayData(Date& )const;
        vector<PrixJour> getAllSimData()const;
        int getsize()const{return robot->getsize();}
};

#endif // SIMULATION_H_INCLUDED
