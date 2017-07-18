#include <RcppArmadillo.h>


typedef double distancetype;
typedef distancetype (*distancefuntype)(const arma::Col<double>&, const arma::Col<double>&);

class DenseCustomFactory;

