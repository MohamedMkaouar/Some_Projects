                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %%%%%%% MATLAB Script pour l'affichage du graph %%%%%%%
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                
%%veuillez changer l'emplacement des fichiers et mettre celui choisi en
%%éxécutant test.cpp


M=dlmread('C:\\Users\\Bechir\\Desktop\\SIM202\\Matrice.txt');
X=dlmread('C:\\Users\\Bechir\\Desktop\\SIM202\\Xcoord.txt');
Y=dlmread('C:\\Users\\Bechir\\Desktop\\SIM202\\Ycoord.txt');
g=graph(M);
plot(g,'k','Xdata',X,'Ydata',Y,'MarkerSize',5,'NodeLabel',{}); 