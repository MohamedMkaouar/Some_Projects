#include "robot.h"
#include <time.h>
#include <cstdlib>


RobotRandom::RobotRandom():Robot(){};
RobotRandom::RobotRandom(string str):Robot(str){};
Transaction RobotRandom::Decide(const Bourse* b, const Date D){
    float DEC=static_cast<float>(rand())/ static_cast<float> (RAND_MAX);
    Transaction Tr(none,"",0);
    if(DEC*100<70){
        vector<PrixJour> pj = b->getDayPrice(D);
        PrixJour pmin=pj[0];
        PrixJour pMax=pj[0];
        PrixJour pmoy=pj[0];
        int k=rand();
        for(int i=0;i<pj.size();i++){
            if(pj[i]<pmin){pmin=pj[i];}
            else if(pMax<pj[i]){pMax=pj[i];}
            else if (k%100>70) pmoy=pj[i];
        }
        int d =rand();
        PrixJour pm;
        if(d%100<=20){pm=pmin;}
        else if (d%100<=40){pm=pMax;}
        else pm=pmoy;
        if(this->getBalance()>pm.getPrice()){Tr.setType(buy);Tr.setQuantity(1);Tr.setStock(pm.getName());}
    }
    else if(DEC*100<=90) {
        vector<Titre> t = wallet.getActions();
        PrixJour pj ;
        int p;
        for(int i=0;i<t.size();i++){
            srand(time(NULL));
            p=rand();
            if(p%100<70){Tr.setType(sell);Tr.setQuantity(1);Tr.setStock(t[i].getName());break;}
        }
    }
    return Tr;
}
