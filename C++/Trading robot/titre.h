#ifndef TITRE_H_INCLUDED
#define TITRE_H_INCLUDED
#include <string>
using namespace std;
class Titre
{
	private :
		string name ;
		int quantity;
	public :
		Titre();
		Titre(string,int);
		void setName(string) ;
		string getName() const;
    	int getQuantity() const;
    	void addQuantity(int);
    	void reduceQuantity(int);
        friend ostream& operator<<(ostream& , const Titre );
};


#endif // TITRE_H_INCLUDED
