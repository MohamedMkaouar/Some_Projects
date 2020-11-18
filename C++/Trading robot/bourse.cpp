#include"bourse.h"


Bourse::Bourse(){};
Bourse::Bourse(string str){name=str;}
Bourse::~Bourse(){};
string Bourse::getName()const{return name;}
void Bourse::setName(string str){name=str;}


