#include"robot.h"


Robot::Robot(){
    wallet=Portefeuille();
}
Robot::Robot(string str){
            wallet=Portefeuille(str);
            name=str;
}
string Robot::getName()const{
    return(name);
}
float Robot::getBalance()const{
    return(wallet.getBalance());
}
vector<Titre> Robot::getActions()const{
   return(wallet.getActions());
}
Portefeuille Robot::getWallet()const{
    return wallet;
}
bool  Robot::possesAction(string str,int q){
    int i= wallet.search(str);
    if(i==-1){return(false);}
    if(wallet.getTabCase(i).getQuantity()<q){return false;}
    return(true);
}
void Robot::setBalance(float f){
    wallet.setBalance(f);
}
void Robot::buyAction(string str, int nbre){
    wallet.buyAction(str,nbre);
}
void Robot::payment(float f){
    wallet.changeBalance(-f);
}
void Robot::sellAction(string str , int nbre){
    wallet.sellAction(str,nbre);
}
void Robot::receipt(float f){
    wallet.changeBalance(f);
}
float Robot::fortune(Bourse* b , Date d){
    float fortune = wallet.getBalance();
    vector<Titre> T = wallet.getActions();
    PrixJour pj ;
    for (unsigned int i=0 ; i<T.size();i++){
        pj=b->getDayCompanyPrice(T[i].getName(),d);
        fortune+=pj.getPrice()*T[i].getQuantity();
    }
    cout<<wallet;
    return fortune;
}
