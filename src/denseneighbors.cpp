#include "neighbors.h"
#include "denseneighbors.h"
#include "distance.h"
#include "largeVis_types.h"

using namespace Rcpp;
using namespace std;
using namespace arma;

class DenseEuclidean : public DenseAnnoySearch {
protected:
	virtual distancetype distanceFunction(const Col<double>& x_i, const Col<double>& x_j) const {
		return relDist(x_i, x_j);
	}
public:
	DenseEuclidean(const Mat<double>& data, const kidxtype& K, Progress& p) : DenseAnnoySearch(data, K, p) {}
};

class DenseCosine : public DenseAnnoySearch {
protected:
	virtual distancetype distanceFunction(const Col<double>& x_i, const Col<double>& x_j) const {
		return cosDist(x_i, x_j);
	}
public:
	DenseCosine(const Mat<double>& data, const kidxtype& K, Progress& p) : DenseAnnoySearch(data, K, p) {}
};




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


	DenseAnnoySearch* annoy;

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
	  Rcpp::XPtr<DenseCustomFactory> dc(distMethod);
	  annoy = dc->getDC(data, K, p);
	}

	annoy->setSeed(seed);
	annoy->trees(n_trees, threshold);
	annoy->reduce();
	annoy->exploreNeighborhood(maxIter);
	imat ret = annoy->sortAndReturn();
	delete annoy;
	return ret;
}
