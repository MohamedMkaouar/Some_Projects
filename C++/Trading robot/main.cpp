#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include"date.h"
#include"PrixJour.h"
#include"bourse.h"
#include"simulation.h"
using namespace std;

int main()
{
    int n,m,p,f;
    string str1,str2;
    Date d1,d2;
    string path="C:\\Users\\kais\\Documents\\scolaire\\2ATA\\C++\\prices.csv";
    Simulation sim;
    cout<<"           "<<"Bonjour                                    *"<<endl;
    cout<<"***********************************************************"<<endl;
    cout<<"*          "<<"Choisir le robot a utiliser                *"<<endl;
    cout<<"*          "<<"1-Robot aleatoire                          *"<<endl;
    cout<<"*          "<<"2-Robot intelligent                        *"<<endl;
    cout<<"***********************************************************"<<endl;
    do{
    cout<<"*          "<<"Entrer votre choix:                        *"<<endl;
    cin>>n;
    }while(n<1||n>2);
    cout<<"***********************************************************"<<endl;
    cout<<"*          "<<"Donner le nom de votre robot               *"<<endl;
    cin>>str1;
    cout<<"***********************************************************"<<endl;
    cout<<"*          "<<"Choisir la bourse a utiliser               *"<<endl;
    cout<<"*          "<<"1-Bourse Tableau                           *"<<endl;
    cout<<"*          "<<"2-Bourse Map                               *"<<endl;
    cout<<"***********************************************************"<<endl;
    do{
    cout<<"*          "<<"Entrer votre choix:                        *"<<endl;
    cin>>m;
    }while(m<1||m>2);
    cout<<"***********************************************************"<<endl;
    cout<<"*          "<<"Donner le nom de votre bourse              *"<<endl;
    cin>>str2;
    cout<<"***********************************************************"<<endl;
    cout<<"*          "<<"Specifier L'intervalle de simualtion        *"<<endl;
    cout<<"*          "<<"1-Une date de debut                        *"<<endl;
    cout<<"*          "<<"2-Une date de debut et une date de fin      *"<<endl;
    cout<<"***********************************************************"<<endl;
    do{
    cout<<"*          "<<"Entrer votre choix:                        *"<<endl;
    cin>>p;
    }while(p<1||p>2);
    cout<<"***********************************************************"<<endl;
    cout<<"*          "<<"Entrer la premiere date au format jj/mm/aaaa*"<<endl;
    cin>>d1;
    if(p==2){
        cout<<"***********************************************************"<<endl;
        cout<<"*          "<<"Entrer la deuxieme date au format jj/mm/aaaa*"<<endl;
        cin>>d2;
    }
    sim=Simulation(n,str1,m,str2,d1);
    if (p==1){ }
    else if (p==2){sim=Simulation(n,str1,m,str2,d1,d2);}

    cout<<"***********************************************************"<<endl;
    cout<<"*          "<<"choisissez la source de donnees            *"<<endl;
    cout<<"*          "<<"1-Un fichier                               *"<<endl;
    cout<<"***********************************************************"<<endl;
    do{
    cout<<"*          "<<"Entrer votre choix:                        *"<<endl;
    cin>>f;
    }while(f<1||f>2);
    if(path==""&&f==1){
            cout<<"***********************************************************"<<endl;
            cout<<"*          "<<"Donner le chemin absolu du fichier         *"<<endl;
            cin>>path;
    }
    else{
            cout<<"***********************************************************"<<endl;
            cout<<"*          "<<"Le chemin absolu du fichier est fixer      *"<<endl;
            cout<<path;

    }
    sim.setDatasource(f,path);
    cout<<"***********************************************************"<<endl;
    int s ;
    do{
    cout<<"*          "<<"choisissez le solde initial de votre robot*"<<endl;
    cin>>s;
    }while(s<0);
   sim.setInitialvalues(s);
   cout<<"*************"<<" La simualtion va commencer "<<"**************"<<endl;
    sim.ExcuteSim();
     cout<<"done sim"<<endl;
    cout<<sim.getIntialSolde()<<endl;
    cout<<"balance "<<sim.getRobotBalance()<<endl;
    cout<<"fortune "<<sim.getRobotFortune()<<endl;
    cout<<sim.getsize()<<endl;
    vector<Titre> tit=sim.getRobotActions();
    for(int i=0;i<tit.size();i++){cout<<tit[i]<<endl;}
    return 0;
}
