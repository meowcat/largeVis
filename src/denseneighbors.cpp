#include "neighbors.h"
#include "denseneighbors.h"
#include "distance.h"
#include "largeVis_types.h"

using namespace Rcpp;
using namespace std;
using namespace arma;

class DenseEuclidean : public DenseAnnoySearch {
public:
  virtual distancetype distanceFunction(const Col<double>& x_i, const Col<double>& x_j) const {
		return relDist(x_i, x_j);
	}
	DenseEuclidean(const Mat<double>& data, const kidxtype& K, Progress& p) : DenseAnnoySearch(data, K, p) {}
};


class DenseCosine : public DenseAnnoySearch {
public:
	virtual distancetype distanceFunction(const Col<double>& x_i, const Col<double>& x_j) const {
		return cosDist(x_i, x_j);
	}
	DenseCosine(const Mat<double>& data, const kidxtype& K, Progress& p) : DenseAnnoySearch(data, K, p) {}
};



//' Instantiate an AnnoySearch for the given distMethod
//' 
//' This instantiates a DenseAnnoySearch based on distMethod input.
//' distMethod is either a Rcpp::XPtr<DenseDistanceProvider> or a character
//' If this is a DenseDistanceProvider, use the provided getAnnoySearch to get an AnnoySearch instance,
//' if this is one of the predefined types (Cosine or Euclidean), instantiate it directly without going
//' past a DistanceProvider.
// DenseAnnoySearch * getAnnoySearch(const SEXP& distMethod, const Mat<double>& data, const kidxtype& K, Progress& p)
// {
//  
//   return(annoy);
// }

// [[Rcpp::export]]
arma::imat searchTrees(const int& threshold,
                       const int& n_trees,
                       const int& K,
                       const int& maxIter,
                       const arma::mat& data,
                       const SEXP distMethod,
                       Rcpp::Nullable< NumericVector > seed,
                       Rcpp::Nullable< NumericVector > threads,
                       bool verbose
                         ) {
#ifdef _OPENMP
	checkCRAN(threads);
#endif
  const vertexidxtype N = data.n_cols;

  Progress p((N * n_trees) + (3 * N) + (N * maxIter), verbose);


  DenseAnnoySearch* annoy = NULL;
  
  if(TYPEOF(distMethod) == STRSXP)
  {
    std::string distMethodStr = Rcpp::as<std::string>(distMethod);
    const mat dataMat = (distMethodStr.compare(string("Cosine")) == 0) ? normalise(data) : mat();
    
    if (distMethodStr.compare(string("Cosine")) == 0) {
      annoy = new DenseCosine(dataMat, K, p);
    } else {
      annoy = new DenseEuclidean(data, K, p);
    }
  }
  else // this assumes that it is an XPtr . How to check this? if(TYPEOF(distMethod) == ???)
  {
    Rcpp::XPtr<DenseAnnoySearchProvider> dc(distMethod);
    annoy = dc->getAnnoySearch(data, K, p);
  }
  
  annoy->setSeed(seed);
	annoy->trees(n_trees, threshold);
	annoy->reduce();
	annoy->exploreNeighborhood(maxIter);
	imat ret = annoy->sortAndReturn();
	delete annoy;
	return ret;
}
