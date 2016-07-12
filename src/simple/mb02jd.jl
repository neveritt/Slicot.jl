
"""
Computes a lower triangular matrix R and a matrix Q with
Q^T Q = I such that
    T  =  Q R^T ,

where T is a k*m-by-l*n block Toeplitz matrix with blocks of size
(k,l). The first column of T will be denoted by TC and the first
row by TR. It is assumed that the first min(m*k, n*l) columns of T
have full rank.

By subsequent calls of this routine the factors Q and R can be
computed block column by block column.
"""
function mb02jd(job::Char, k::Integer, l::Integer, m::Integer,
    n::Integer, p::Integer, s::Integer, TC::Array{Float64,2},
    ldtc::Integer, TR::Array{Float64,2}, ldtr::Integer,
    Q::Array{Float64,2}, ldq::Integer,
    R::Array{Float64,2}, ldr::Integer,
    DWORK::Array{Float64,1}, ldwork::Integer=-1)
    ## QR factorization of block Toeplitz matrix  ##
    ## INPUTS/Outputs:
    #--->    job = 'Q' for Q and R, 'R' for only R
    #--->      k = The number of rows in one block of T.  k >= 0
    #--->      l = The number of columns in one block of T.  l >= 0.
    #--->      m = The number of blocks in one block column of T.  m >= 0.
    #--->      n = The number of blocks in one block row of T.  n >= 0.
    #--->      p = The number of previously computed block columns of R.
    #              p*l < min(m*k,n*l) + l and p >= 0.
    #--->      s = The number of block columns of R to compute.
    #              (p+s)*l < min(m*k,n*l ) + l and s >= 0.
    #--->     TC = Array(ldtc, l) On entry, if p = 0, the leading m*k-by-l part
    #              of this Array must contain the first block column of T.
    #--->   ldtc = The leading dimension of the array TC
    #--->     TR = Array(ldTr, (n-1)*l) On entry, if p = 0, the leading k-by-(n-1)*l
    #              part of this array must contain the first block row of T
    #              without the leading k-by-l block.
    #--->   ldtr = The leading dimension of the array TR
    #--->      Q = Array(ldq, min(s*l, min(m*k, n*l)-p*l))
    #              On entry, if job = 'Q'  and  p > 0, the leading m*k-by-l
    #              part of this array must contain the last block column of Q
    #              from a previous call of this routine.
    #              On exit, if JOB = 'Q'  and  info = 0, the leading
    #              m*k-by-min( s*l, min( m*k,n*l )-p*l ) part of this array
    #              contains the p-th to (p+s)-th block columns of the factor Q.
    #--->    ldq = The leading dimension of the array Q.
    #              LDQ >= max(1,m*k), if JOB = 'Q';
    #              LDQ >= 1,          if JOB = 'R'.
    #--->      R = Array(lDR,min( s*l, min( m*k,n*l )-p*l ))
    #              On entry, if p > 0, the leading (n-p+1)*l-by-l
    #              part of this array must contain the nozero part of the
    #              last block column of R from a previous call of this
    #              routine. One exit, if INFO = 0, the leading
    #              min( n, n-p+1 )*l-by-min( s*l, min( m*k,n*l )-p*l )
    #              part of this array contains the nonzero parts of the p-th
    #              to (p+s)-th block columns of the lower triangular
    #              factor R.
    #              Note that elements in the strictly upper triangular part
    #              will not be referenced.
    #--->    ldr = The leading dimension of the array R.
    #              LDR >= max( 1, min( n, n-p+1 )*l )
    #--->  DWORK = Array(LDWORK)
    #              On exit, if INFO = 0, DWORK(1) returns the optimal value
    #              of LDWORK.
    #              On exit, if INFO = -17,  DWORK(1) returns the minimum
    #              value of LDWORK.
    #              If JOB = 'Q', the first 1 + ( (n-1)*l + m*k )*( 2*k + l )
    #              elements of DWORK should be preserved during successive
    #              calls of the routine.
    #              If JOB = 'R', the first 1 + (n-1)*l*( 2*k + l ) elements
    #              of DWORK should be preserved during successive calls of
    #              the routine.
    #---> ldwork = Length of array DWORK.
    #              JOB = 'Q':
    #              LDWORK >= 1 + ( m*k + ( n - 1 )*l )*( l + 2*k ) + 6*l
    #                         + max( m*k,( n - max( 1,p )*l ) );
    #              JOB = 'R':
    #              If p = 0,
    #                LDWORK >= max( 1 + ( n - 1 )*l*( l + 2*k ) + 6*l
    #                                 + (n-1)*l, m*k*( l + 1 ) + l );
    #              If p > 0,
    #                LDWORK >= 1 + (n-1)*l*( l + 2*k ) + 6*l + (n-p)*l.
    #              For optimum performance LDWORK should be larger.

    #              If LDWORK = -1, then a workspace query is assumed;
    #              the routine only calculates the optimal size of the
    #              DWORK array, returns this value as the first entry of
    #              the DWORK array, and no error message related to LDWORK
    #              is issued by XERBLA.
@assert k >= 0 string("k must be larger than 0")
@assert l >= 0 string("l must be larger than 0")
@assert m >= 0 string("m must be larger than 0")
@assert n >= 0 string("n must be larger than 0")
@assert p >= 0 string("p must be larger than 0")
@assert s >= 0 string("s must be larger than 0")
@assert p < min(m*k, n*l) string("p must be smaller than min(m*k,n*l)")
@assert (p+s)*l < min(m*k, n*l) + l string("(p+s)*l must be smaller than min(m*k,n*l) + l")

    #Determine required params for mode

    if job == 'Q'
      ldwork_min = 1 + (m*k + (n-1)*l)*(l + 2*k) + 6*l + max(m*k, (n-max(1,p)*l))
      if ldwork < 0
          #If ldwork isn't specified, set it to the min value
          ldwork = ldwork_min
      elseif ldwork < ldwork_min
        warn("ldwork is not >= the recommended size")
        throw(DomainError())
      end
    elseif job == 'R'
      if p == zero(typeof(p))
        ldwork_min = max( 1 + (n-1)*l*(l + 2*k) + 6*l + (n-1)*l, m*k*(l + 1) + l )
        if ldwork < 0
            #If ldwork isn't specified, set it to the min value
            ldwork = ldwork_min
        elseif ldwork < ldwork_min
          warn("ldwork is not >= the recommended size")
          throw(DomainError())
        end
      elseif p > 0
        ldwork_min = 1 + (m*k + (n-1)*l)*(l + 2*k) + 6*l + max( m*k,(n-max(1, p)*l))
        if ldwork < 0
            #If ldwork isn't specified, set it to the min value
            ldwork = ldwork_min
        elseif ldwork < ldwork_min
          warn("ldwork is not >= the recommended size")
          throw(DomainError())
        end
      end
    else
        #Invalid job mode.
        warn("job must be either 'R' or 'C'")
        throw(DomainError())
    end

    #Validate all input
    if size(TC) != (ldtc,l)
      warn("TC must have dimension (ldtc, l)")
      throw(DomainError())
    elseif size(TR) != (ldtr, (n-1)*l)
      warn("TR must have dimension (n-1)*l")
      throw(DomainError())
    elseif size(Q) != (ldq, min(s*l, min(m*k, n*l)-p*l))
      warn("Q must have dimension (ldq, min(s*l, min(m*k, n*l)-p*l))")
      throw(DomainError())
    elseif size(R) != (ldr,min( s*l, min(m*k,n*l)-p*l ))
      warn("R must have dimension (ldr,min(s*l, min(m*k,n*l)-p*l))")
      throw(DomainError())
    elseif length(DWORK) != ldwork
      warn("DWORK must have length dwork")
      throw(DomainError())
    end


    #Call the subroutine
    INFO = mb02jd!(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, ldwork)

    arglist = methods(mb02jd!).defs.sig

    if job == 'Q'
      return Q, R
    elseif job == 'R'
      return R
    end
end
