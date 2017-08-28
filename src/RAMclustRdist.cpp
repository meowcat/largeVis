#include "largeVis.h"
#include "largeVis_types.h"
#include "distance.h"
#include "denseneighbors.h"

//' @export
//[[Rcpp::export]]
distancetype ramclustDist(const arma::vec& i, const arma::vec& j, const double sr, const double st, const double maxt, const double maxdist) {
  const dimidxtype D = i.n_elem;

    // the first entry is the rt, which is used to form delta_rt^2 / 2st^2,
  // the second to last entries are the intensities to be used for correlation, 1-Cor^2/2sr^2 (the resulting value goes from 0 to 2)
  
  // I am not sure whether the following two versions make a difference in speed,
  // this depends on how arma::vec is implemented... I leave the potentially faster in now
  // Correlation of elements 2 to last
  
  // const arma::vec i_two = i.subvec(1, D-1);
  // const arma::vec j_two = j.subvec(1, D-1);
  // distancetype r = 1 - arma::as_scalar(arma::cor(
  //   i_two, j_two));
  
  distancetype r = 1 - arma::as_scalar(arma::cor(
    i.subvec(1, D-1),
    j.subvec(1, D-1)));
  // Note that we already convert the correlation to a distance measure:
  // the range of 1-cor(a,b) is zero to 2 since cor(a,b) is -1 to 1
  // Note that there is a better way of converting the distance (TODO: find citation), 
  // but this is the one used in RAMClustR
  
  // delta t, using the first element as retention time
  distancetype t = abs(i[0] - j[0]);
  if(t > maxt) return 1;
  
  // 1- e^(-dt^2/2st^2) * e^( -R^2/2sr^2 )
  distancetype dist = exp(-((t*t) / (2*(st*st)) )) * exp(- (r*r) / (2*(sr*sr)) );
  dist = 1-dist;
  if(dist > maxdist) return 1;  
  
  return dist;
}

class DenseRamclust : public DenseAnnoySearch {
protected:
  double _sr;
  double _st;
  double _maxt;
  double _maxdist;
  virtual distancetype distanceFunction(const Col<double>& x_i, const Col<double>& x_j) const {
    return ramclustDist(x_i, x_j, _sr, _st, _maxt, _maxdist);
  }
public:
  DenseRamclust(const Mat<double>& data, const kidxtype& K, Progress& p,
                const double sr, const double st, const double maxt, const double maxdist
  ) : DenseAnnoySearch(data, K, p), _sr(sr), _st(st), _maxt(maxt), _maxdist(maxdist) {}
};


class DenseRamclustProvider : public DenseAnnoySearchProvider {
protected:
  double _sr;
  double _st;
  double _maxt;
  double _maxdist;
public:
  DenseRamclustProvider(const double sr, const double st, const double maxt, const double maxdist) : 
    _sr(sr), _st(st), _maxt(maxt), _maxdist(maxdist) {}
  virtual DenseAnnoySearch * getAnnoySearch(const Mat<double>& data, const kidxtype& K, Progress& p)
  {
    return(new DenseRamclust(data, K, p, _sr, _st, _maxt, _maxdist));
  }
  virtual double distanceFunction(const Col<double>& x_i, const Col<double>& x_j) const
  {
    return ramclustDist(x_i, x_j, _sr, _st, _maxt, _maxdist);
  }
  DenseAnnoySearchProvider * getAnnoySearchProvider()
  {
    return(new DenseRamclustProvider(_sr, _st, _maxt, _maxdist));
  }
};

//' @export
//[[Rcpp::export]]
SEXP ramclustDistance(double sr, double st, double maxt, double maxdist)
{
  DenseAnnoySearchProvider * f = new DenseRamclustProvider(sr, st, maxt, maxdist);
  return(wrap(Rcpp::XPtr<DenseAnnoySearchProvider>(f,true)));
}



