
"""
Computes a lower triangular matrix R and a matrix Q with
Q^T Q = I such that
    T  =  Q R ,

where T is a k*m-by-l*n block Toeplitz matrix with blocks of size k-by-l.
"""
function mb02jd(job::Char, col::Array{Float64,2}, row::Array{Float64,2})
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
  DWORK = Array{Float64}(1)
  ldwork = -1

  #Call the subroutine for optimal size of workspace
  Raw.mb02jd!(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, ldwork)
  ldwork = 2*convert(Int,round(DWORK[1]))
  DWORK = Array{Float64}(ldwork)

  #Call the subroutine
  Raw.mb02jd!(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, ldwork)

  Rt = UpperTriangular(R')
  arglist = methods(Raw.mb02jd!).defs.sig
  if job == 'Q'
    return Q, Rt
  elseif job == 'R'
    return Rt
  end
end
