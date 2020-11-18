#ifndef TRANSACTION_H_INCLUDED
#define TRANSACTION_H_INCLUDED
#include<string>
#include<iostream>
using namespace std;


enum typeTrans {sell,buy,none};
class Transaction{
    private :
        typeTrans type;
        string stock;
        int quantity;
    public :
        Transaction(){};
        ~Transaction(){};
        Transaction(typeTrans t, string str,int q){type=t;stock=str;quantity=q;}
        typeTrans getType()const{return type;}
        int getQuantity()const{return quantity;}
        string getStock()const{return stock;}
        void setType(typeTrans tt){type=tt;}
        void setQuantity(int q){quantity=q;}
        void setStock(string str){stock=str;}
        friend ostream& operator<<(ostream& , const Transaction );
};
#endif // TRANSACTION_H_INCLUDED
