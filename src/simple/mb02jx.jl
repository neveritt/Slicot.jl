
"""
Compute a low rank QR factorization with column pivoting of a
K*M-by-L*N block Toeplitz matrix T with blocks of size (K,L);
specifically,
                      T P =  Q R,

where R is lower trapezoidal, P is a block permutation matrix
and Q^T Q = I. The number of columns in R is equivalent to the
numerical rank of T with respect to the given tolerance TOL1.
Note that the pivoting scheme is local, i.e., only columns
belonging to the same block in T are permuted.
"""
function mb02jx(job::Char, col::Array{Float64,2}, row::Array{Float64,2},
    tol1=0.0, tol2=0.0)
  ## QR factorization of block Toeplitz matrix  ##
  ## INPUTS/Outputs:
  #--->    job = 'Q' for Q and R, 'R' for only R
  #--->    col = Array of dimension (k*m, l) First column of blocks T.
  #--->    row = Array of dimension (k, l*n) First row of blocks T.
  l::Int = size(col)[2]
  k::Int = size(row)[1]
  m = convert(Int,size(col)[1]/k)
  n = convert(Int,size(row)[2]/l)
  p::Int = 0
  s::Int = n

  TC = col
  TR = row[:,l+1:end]
  ldtc = size(TC)[1]
  ldtr = size(TR)[1]
  Q = Array{Float64}(m*k, min(s*l, min(m*k, n*l)-p*l))
  ldq = size(Q)[1]
  R = Array{Float64}(l*n, min( s*l, min( m*k,n*l )-p*l ))
  ldr = size(R)[1]


  if job == 'Q'
    ldwork = max(3, (m*k + (n-1)*l)*(l + 2*k) + 9*l + max(m*k,(n-1)*l))
  elseif job == 'R'
    ldwork = max(3, (n-1)*l*(l + 2*k + 1) + 9*l, m*k*(l+1) + l)
  end
  DWORK = Array{Float64}(ldwork)

  rnk = l*n
  JPVT = Array{BlasInt}(min(m*k,n*l))

  #Call the subroutine
  Raw.mb02jx!(job, k, l, m, n, TC, ldtc, TR, ldtr, rnk, Q, ldq, R, ldr, JPVT, 0.0, 0.0, DWORK, ldwork)

  Rt = UpperTriangular(R')
  arglist = methods(Raw.mb02jx!).defs.sig
  if job == 'Q'
    return Q, Rt, JPVT
  elseif job == 'R'
    return nothing, Rt, JPVT
  end
end
