#ifndef _LARGEVISDENSENEIGHBORS
#define _LARGEVISDENSENEIGHBORS
#include "largeVis_types.h"
#include "neighbors.h"
#include "distance.h"


class DenseAnnoySearch : public AnnoySearch<arma::Mat<double>, arma::Col<double>> {
protected:
  virtual vec hyperplane(const ivec& indices) {
    const vertexidxtype I = indices.n_elem;
    vec direction = vec(I);

    const vertexidxtype idx1 = sample(I);
    vertexidxtype idx2 = sample(I - 1);
    idx2 = (idx2 >= idx1) ? (idx2 + 1) % I : idx2;

    const vec x2 = data.col(indices[idx1]);
    const vec x1 = data.col(indices[idx2]);
    // Get hyperplane
    const vec m =  (x1 + x2) / 2; // Base point of hyperplane
    const vec d = x1 - x2;
    const vec v =  d / as_scalar(norm(d, 2)); // unit vector

    for (vertexidxtype i = 0; i != I; i++) {
      const vec X = data.col(indices[i]);
      direction[i] = dot((X - m), v);
    }
    return direction;
  }
public:
  DenseAnnoySearch(const mat& data, const kidxtype& K, Progress& p) : AnnoySearch(data, K, p) {}
};


class DenseAnnoySearchProvider : public AnnoySearchProvider<DenseAnnoySearch>
{
};


#endif
