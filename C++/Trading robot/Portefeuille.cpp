
#include"Portefeuille.h"


Portefeuille::Portefeuille(){};
Portefeuille::Portefeuille(string s){
    owner=s;
    size=0;
}
Portefeuille::~Portefeuille(){
    tab.clear();
}
string Portefeuille::getOwner() const {
    return owner ;
}
int Portefeuille::getSize() const{
    return size;
}

float Portefeuille::getBalance() const {
    return balance ;
}

Titre Portefeuille::getTabCase(int i) const {
    return tab[i];
}
string   Portefeuille::getActionName(int i)const{
    return (tab[i].getName());
}
int  Portefeuille::getActionQuantity(int i)const{
    return (tab[i].getQuantity());
}
vector <Titre> Portefeuille::getActions()const{
    return tab;
};
void Portefeuille::setBalance(float b){
    balance=b;
}
int Portefeuille::search(string action){
   int i=0;
    while (i<size)
    {
        if ((tab[i].getName())== action){ return i ;}
        i++;
    }
    if (i>=size)
        i=-1;
    return i;
}
void Portefeuille::buyAction(string name, int q){
    int i=this->search(name);
    if (i>=0)
        tab[i].addQuantity(q);
    else
        {
        Titre t (name,q);
        tab.push_back(t);
        size+=1;
        }
}
void Portefeuille::sellAction(string name, int q){
    int i=search(name);
    if (i>=0)
        {if (tab[i].getQuantity()>q)
        tab[i].reduceQuantity(q);
        else
        deleteCase(i);
        }
}
void Portefeuille::deleteCase(int i){
    tab.erase(tab.begin()+i);
    size-=1;
}
void Portefeuille::changeBalance(float f){
    balance+=f;
}
ostream& operator<<(ostream& flux , const Portefeuille P ){
    flux<<"C'est le portefeuille de : "<<P.getOwner()<<"\n";
    flux<<"il possede le montant "<<P.getBalance()<<" et les actions suivants : "<<"\n";
    vector<Titre> tab= P.getActions();
    for(int i=0; i<P.getSize(); i++){flux<<tab[i]<<"\n";}
    return flux;
};
