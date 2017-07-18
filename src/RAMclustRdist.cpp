#include "largeVis.h"
#include "largeVis_types.h"
#include "distance.h"
#include "denseneighbors.h"

//[[Rcpp::export]]
distancetype ramclustDist(const arma::vec& i, const arma::vec& j, const double sr, const double st) {
  const dimidxtype D = i.n_elem;

    // the first entry is the rt, which is used to form delta_rt^2 / 2st^2,
  // the second to last entries are the intensities to be used for correlation, 1-Cor^2/2sr^2 (the resulting value goes from 0 to 2)
  const arma::vec i_two = i.subvec(1, D-1);
  const arma::vec j_two = j.subvec(1, D-1);
  distancetype r = 1 - arma::as_scalar(arma::cor(i_two, j_two));
  distancetype t = abs(i[0] - j[0]);

  distancetype dist = exp(-((t*t) / (2*(st*st)) )) * exp(- (r*r) / (2*(sr*sr)) );
  return 1-dist;
}

class DenseRamclust : public DenseCustom {
protected:
  double _sr;
  double _st;
  virtual distancetype distanceFunction(const Col<double>& x_i, const Col<double>& x_j) const {
    return ramclustDist(x_i, x_j, _sr, _st);
  }
public:
  DenseRamclust(const Mat<double>& data, const kidxtype& K, Progress& p,
                const double sr, const double st
  ) : DenseCustom(data, K, p, NULL), _sr(sr), _st(st) {}
};


class DenseRamclustFactory : public DenseCustomFactory {
protected:
  double _sr;
  double _st;
public:
  DenseRamclustFactory(const double sr, const double st) : _sr(sr), _st(st), DenseCustomFactory(NULL) {}
  virtual DenseRamclust * getDC(const Mat<double>& data, const kidxtype& K, Progress& p)
  {
    return(new DenseRamclust(data, K, p, _sr, _st));
  }
};

//[[Rcpp::export]]
Rcpp::XPtr<DenseCustomFactory> ramclustDistance(double sr, double st)
{
  DenseCustomFactory * f = new DenseRamclustFactory(sr, st);
  return(Rcpp::XPtr<DenseCustomFactory>(f,true));
}



