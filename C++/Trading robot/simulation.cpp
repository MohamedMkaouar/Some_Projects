#include"simulation.h"


Simulation::Simulation(){};
void Simulation::setBourse(int n ,string str){
    if(n==1){
        bourse=new BourseTab(str);
    }
    else if (n==2){
        bourse=new BourseMapD(str);
    }
}
void Simulation::setRobot(int m,string str){
    if(m==1){
        robot= new RobotRandom(str);
    }
    else if (m==2){
        robot= new RobotSmart(str);
    }
}
Simulation::Simulation(int n, string str1,int m ,string str2, Date d){
    setRobot(n,str1);
    setBourse(m,str2);
    firstDay= new Date (d);
    finalDay=NULL;
}
Simulation::Simulation(int n, string str1,int m ,string str2, Date d1, Date d2){
    setRobot(n,str1);
    setBourse(m,str2);
    firstDay= new Date (d1);
    finalDay= new Date (d2);
}

void Simulation::setInitialSolde(float f){
    intialSolde=f;
    robot->setBalance(f);
}
void Simulation::setInitialvalues(float f){
    setInitialSolde(f);
    today=*firstDay;
}

void Simulation::setDatasource(int n ,const string& str){
    if(n==1){this->getDataFile(str);}
}
void Simulation::getDataFile(const string& path){
    ifstream fichier(path.c_str(), ios::in);
    if(fichier)  // si l'ouverture a fonctionné
        {
            while ((!fichier.eof()) )
               {
                string contenu;  // déclaration d'une chaîne qui contiendra la ligne lue
                getline(fichier, contenu,'-');  // on met dans "contenu" la chaîne désirée
                int c1;
                istringstream iss (contenu);
                iss >> c1;
                getline(fichier, contenu,'-');
                int c2;
                istringstream iss2 (contenu);
                iss2 >> c2;
                getline(fichier, contenu,',');
                int c3;
                istringstream iss3 (contenu);
                iss3 >> c3;
                Date D(c3,c2,c1);
                getline(fichier, contenu,',');
                string Nom_action=contenu;
                getline(fichier, contenu,',');
                double p1;
                istringstream issp1 (contenu);
                issp1 >> p1;
                getline(fichier, contenu,',');
                double p2;
                istringstream issp2 (contenu);
                issp2 >> p2;
                getline(fichier, contenu,',');
                double p3;
                istringstream issp3 (contenu);
                issp3 >> p3;
                getline(fichier, contenu,',');
                double p4;
                istringstream issp4 (contenu);
                issp4 >> p4;
                double m=(p1+p2+p3+p4)/4;
                getline(fichier, contenu,'\n');
                PrixJour P(Nom_action,m,D);
                bourse->addDailyPrice(P);
        }}
        else
                cerr << "Impossible d'ouvrir le fichier !" << endl;
        fichier.close();
        cout<<"done"<<endl;
}

string Simulation::getRobotName()const{return robot->getName();}
string Simulation::getBourseName()const{return bourse->getName();}
Date Simulation::getToday()const{return today;}
float Simulation::getIntialSolde()const{return intialSolde;}
vector<Date> Simulation::getSimualtionInterval()const{
    vector<Date> d ;
    d.push_back(*firstDay);
  if(finalDay!=NULL){
                d.push_back(*finalDay);
   }
    return(d);
}
float Simulation::getRobotFortune()const{
    Date D ;
    if(finalDay!=NULL&&bourse->open(*finalDay)){D=*finalDay;}
    else if (finalDay!=NULL&&!(bourse->open(*finalDay))){
        D=*finalDay;
        while(!bourse->open(D)){
            D.decrementDate();
        }
    }
    else{D=bourse->closure();}

    return robot->fortune(bourse,D);
    }
float Simulation::getRobotBalance()const{return robot->getBalance();}
vector<Titre> Simulation::getRobotActions()const{return robot->getActions();}
vector<PrixJour> Simulation::getDayData(Date &d)const{
     return bourse->getDayPrice(d);
}
vector<PrixJour> Simulation::getAllSimData()const{
    Date arret;
    if(finalDay==NULL){arret=bourse->closure();}
    else arret=*finalDay;
    return bourse->getAllHistoric(arret);
}
Portefeuille Simulation::getRobotWallet()const{
    return robot->getWallet();
}
bool Simulation::validateTransaction(const Transaction Tr){
    bool result;
    if(Tr.getType()==buy){
        float balance=robot->getBalance();
        PrixJour pj=bourse->getDayCompanyPrice(Tr.getStock(),today);
        float payment = Tr.getQuantity()*pj.getPrice();
        if(payment>balance){result= false;}
        else if(Tr.getQuantity()<0){return false;}
        else {
            result= true;
        }
    }
    else if (Tr.getType()==sell){
        if(robot->possesAction(Tr.getStock(),Tr.getQuantity())){
                result= true;
        }
        else result = false ;
    }
    else if (Tr.getType()==none){ result =true ;}

    return result;
}
void Simulation::effectTransaction(Transaction Tr){
    if(Tr.getType()==buy){
         robot->buyAction(Tr.getStock(),Tr.getQuantity());
         PrixJour pj=bourse->getDayCompanyPrice(Tr.getStock(),today);
         robot->payment(pj.getPrice()*Tr.getQuantity());
    }
    else if(Tr.getType()==sell) {
        PrixJour pj=bourse->getDayCompanyPrice(Tr.getStock(),today);
        robot->sellAction(Tr.getStock(),Tr.getQuantity());
        robot->receipt(pj.getPrice()*Tr.getQuantity());
    }
}
void Simulation::ExcuteSim(){
    bool cond=true;
    Date arret;
    if(finalDay==NULL){arret=bourse->closure();}
    else arret=*finalDay;
    while(true){
        cout<<today<<" la bourse est "<<bourse->open(today)<<endl;
        if(bourse->open(today)){
            Transaction tr(buy,"",0);
            while(tr.getType()!=none){
                tr=robot->Decide(bourse,today);
                if(validateTransaction(tr)){effectTransaction(tr);cout<<tr<<endl;}
                else {
                    cout<<tr<<endl;
                    throw invalid_argument( " Transcation invalide " );
                }
            }
        }
        if(cond==false){break;}
        if(today==arret){break;}
        today.incrementDate();
        }
    cout<<"simualtion over   "<<today<<endl;
}
