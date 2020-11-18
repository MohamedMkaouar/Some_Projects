#ifndef PRIXJOUR_H_INCLUDED
#define PRIXJOUR_H_INCLUDED

#include <string>
#include "date.h"

class PrixJour {
    private :
        string name ;
        double price ;
        Date day ;
        friend bool operator< (const PrixJour&,const PrixJour&);
        friend bool operator<= (const PrixJour&,const PrixJour&);
        friend bool operator== (const PrixJour&,const PrixJour&);
        friend bool operator!=(const PrixJour&, const PrixJour&);
        friend bool operator> (const PrixJour&,const PrixJour&);
        friend bool operator>= (const PrixJour&,const PrixJour&);
    public :
        PrixJour();
        PrixJour(string,double,Date);
        ~PrixJour(){};
        string getName() const;
        double getPrice() const;
        Date getDay() const;
        void setName(string);
        void setPrice(double);
        void setDay(Date);
        PrixJour operator=(const PrixJour&);
        friend ostream& operator<<(ostream& , const PrixJour );

};

#endif // PRIXJOUR_H_INCLUDED
