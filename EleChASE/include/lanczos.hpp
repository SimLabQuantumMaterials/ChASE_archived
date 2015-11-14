/*! \file lanczos.hpp
 *      \brief Header file for Lanczos algorithm
 *      \details This file contains a single function, which is a partial implementation of the Lanczos algorithm,
 *      to the extent we neeed in our algorithm.
 *  */

/// \cond

#ifndef ELECHFSI_LANCZOS
#define ELECHFSI_LANCZOS

#include "El.hpp"
using namespace El;
using namespace std;

/// \endcond

typedef double Real;
typedef Complex<Real> C;

/** \fn lanczos(UpperOrLower uplo, const DistMatrix<F>& A, DistMatrix<F>& V,
             const int ub_iter, const int block, const int max_iter,
             DistMatrix<R,VR,STAR>& Lambda, R* const upper)
 * \brief Implementation of the Lanczos algorithm.
 * \details This function implements the Lancsos algorithms
 * up to an extent needed for our algorithm. Namely, a few iterations of the Lancsos algorithm are used in the first steps of ChASE
 * to give an upper estimate of the eigenvalue spectrum. The vectors with eigenvalues larger than this value, are to be suppressed by the filter.
 * This function also calls an Elemental direct tridiagonal eigensolver, in order to compute the eigenvalues of the the arrising tridiagonal
 * eigensystem, generated by the partial Lanczos procedure.
 * \param uplo          Elemental enum for specifying "LOWER" or "UPPER" part of the hermitian matrix is being used.
 * \param A             Elemental DistMatrix of template type F. Contains the input hermitian matrix, defining the eigenvalue problem.
 * \param ub_iter       Integer specifying the number of Lanczos iterations used to bound the spectral interval.
 * \param block         Integer specifying the number Ritz values to approximate for the Random version of ChASE, set to ub_iter if not needed.
 * \param max_iter      Integer specifying the number of Lanczos iterations that are to be run.
 * \param V             Elemental DistMatrix of template type F. Outputs the approximate eigenvectors for the first system in
 *                                      the sequence, in case the algorithm is run with "random" parameter. Each vector is a separate column in the matrix V.
 * \param Lambda        Elemental DistMatrix of doubles. Outputs the guessed eigenvalues on exit, if ub_iter /= block on input.
 * \param upper Pointer of type R (double). Contains a single real value, the upper bound estimate of the eigenvalue spectrum.
 * \return void
 */
template<typename F> void
lanczos(UpperOrLower              uplo,
        const DistMatrix<F>&      A,
        DistMatrix<F>&            v,
        const int                 min_iter,
        const int                 max_iter,
        const int                 block,
        Real* const               upper,
        Real* const               lower,
        Real* const               lambda)
{
  Real alpha_r, beta_r;
  F    alpha_c, beta_c;

  DistMatrix<F>  z(A.Height(), 1, A.Grid());
  DistMatrix<F> v0(A.Height(), 1, A.Grid());

  Matrix<Real> dd(min_iter, 1);
  Matrix<F>    ee(min_iter, 1);
  Matrix<Real> ww(min_iter, 1);
  Matrix<F>    ZZ(min_iter, min_iter);

  DistMatrix<Real> d_(max_iter, 1, A.Grid());
  DistMatrix<F>    e_(max_iter, 1, A.Grid());
  DistMatrix<Real> w_(max_iter, 1, A.Grid());

  alpha_r = Nrm2(v);

  alpha_c = F(1/alpha_r);
  Scale(alpha_c, v);

  Zeros(v0, A.Height(), 1);

  beta_r = 0.0;

  for (int k = 0; k < max_iter; )
    {
      Hemv(uplo, F(1.0), A, v, F(0.0), z);

      alpha_c = Dot(z, v);
      Axpy(-alpha_c, v, z);
      Axpy(F(-beta_r), v0, z);

      beta_r = Nrm2(z);

      v0 = v;
      Zeros(v, A.Height(), 1);
      Axpy(F(1/beta_r), z, v);

      if (k < min_iter)
        {
          dd.Set(k, 0, RealPart(alpha_c));
          ee.Set(k, 0, beta_r);
        }

      if (block != min_iter)
        {
          d_.Set(k, 0, RealPart(alpha_c));
          if (k < max_iter) e_.Set(k, 0, beta_r);
        }

      k += 1;
      if (k == min_iter)
        {
          HermitianTridiagEig(dd, ee, ww, ZZ, ASCENDING);

          Real* w = ww.Buffer();
          F*    Z = ZZ.Buffer();
          alpha_r = fabs(w[0]) > fabs(w[k-1]) ? fabs(w[0]) : fabs(w[k-1]);

          *upper = alpha_r + fabs(beta_r)*fabs(Z[(k-1)*k+k-1]);
        }
    }

  if (lower != NULL && lambda != NULL)
    {
      HermitianTridiagEig(d_, e_, w_, ASCENDING);

      *lower  = w_.Get(block, 0);
      *lambda = w_.Get(0, 0);
    }

  //if (mpi::Rank(mpi::COMM_WORLD) == 0)
  //  cout << *upper << endl;

  mpi::Barrier(A.Grid().Comm());

  return;
}

#endif  // ELECHFSI_LANCZOS
