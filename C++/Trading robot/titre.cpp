#include "titre.h"
#include<iostream>


Titre::Titre(){;}
Titre::Titre(string s , int n){
     name=s ;
     quantity=n;
}
void Titre::setName(string s){
    name = s;
}
string Titre::getName() const {
    return name;
}
int Titre::getQuantity() const {
    return quantity;
}
void Titre::addQuantity(int i)
{
    quantity+=i;
};
void Titre::reduceQuantity(int i)
{
    quantity-=i;
};
ostream& operator<<(ostream& flux, const Titre tt){
    flux<<"le titre "<<tt.getName()<<", le nombre d'action est :"<<tt.getQuantity();
    return flux;
}
