#include"robot.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <random>


RobotSmart::RobotSmart():Robot(){};
RobotSmart::RobotSmart(string str):Robot(str){dateMemo=NULL;};
void RobotSmart::memorySet(const Bourse* b, const Date D){
        vector<string> Cmp = b->getCompanyNames(D);
        for(int j=0;j<Cmp.size();j++)
            {
            Date date=D;
            vector<float> prices;
            int i =0;
             prices.push_back(b->getDayCompanyPrice(Cmp[j],date).getPrice());
             date.decrementDate();
             while(date>b->opening()&&i<7){
                    if(b->open(date)){
                    prices.push_back(b->getDayCompanyPrice(Cmp[j],date).getPrice());
                    }
                    i++;
                    date.decrementDate();
             }
             int taille=prices.size();
            float moy =0;
            for(int i=0;i<taille;i++)
            {
                moy+= prices[i];
            }
            float mean = moy/taille;
            float s=0;
            for(int i=0;i<taille;i++)
            {
                s+= pow(prices[i]-mean,2);
            }
            if(memory.count(Cmp[j])==0){
            memory[Cmp[j]].push_back(mean);
            memory[Cmp[j]].push_back(s);
            memory[Cmp[j]].push_back(0);
            }
            else {
            memory[Cmp[j]][0]=mean;
            memory[Cmp[j]][1]=s;
            memory[Cmp[j]][2]=0;
            }
            }
}
Transaction RobotSmart::Decide(const Bourse* b, const Date D){
    if(dateMemo==NULL){
        memorySet(b,D);
        dateMemo = new Date(D);
    }
    else if(*dateMemo!=D){
        memorySet(b,D);
        dateMemo = new Date(D);
    }
    Transaction Tr(none,"",0);
    map<string, vector<float> >::iterator i =memory.begin();
    string str;
    float sold =this->getBalance();
    float sigma,mean;
    int decision;
    while(i!=memory.end()){
        decision=i->second[2];
        if(decision==1){i++;continue;}
        mean=i->second[0];
        sigma=i->second[1];
        int q = round(sigma*mean);
        if(q>100){q=99;}
        if(q==0){q=1;}
        int Rdec=0;
        if(sigma < sqrt(mean)&&sold>b->getDayCompanyPrice(i->first,D).getPrice()*q&&q!=0)
        {
            Rdec=-1;
        }
          else if((sigma > 2.5*sqrt(mean))&&possesAction(i->first,q)&&q!=0)
        {
            Rdec=1;
        }
		float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        default_random_engine generator;
        bernoulli_distribution distribution(r);
            if ((distribution(generator)))
        {
            Rdec=Rdec*(-1);
        }

        if(Rdec==-1&&sold>b->getDayCompanyPrice(i->first,D).getPrice()*q&&q!=0)
        {
            Tr.setQuantity(q);
            Tr.setType(buy);
            Tr.setStock(i->first);
            i->second[2]=1;
            return Tr;
        }
        else if(Rdec==1&&possesAction(i->first,q)&&q!=0)
        {
            Tr.setQuantity(q);
            Tr.setType(sell);
            Tr.setStock(i->first);
            i->second[2]=1;
            return Tr;
        }
        else
        {
            i->second[2]=1;
        }
        i++;
    }
    return Tr;
}
