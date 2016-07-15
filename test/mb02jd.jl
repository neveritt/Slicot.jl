using ToeplitzMatrices
using ControlCore
using ControlToolbox
using IdentificationToolbox
using BenchmarkTools
using Slicot

N = 20000
na = 1000
nb = na
nk = 1
B = [0,1,1]
A = [1,0.3,0.1]
Θ0 = [B[2:end]; zeros(nb-length(B)+1); B[2:end]; zeros(nb-length(B)+1); A[2:end]; zeros(na-length(A)+1)]

u1 = randn(N)
u2 = randn(N)
u3 = randn(N)
u4 = randn(N)

H1 = tf([0,0,1],A,1)
ev1 = randn(N)
y1 = filt(B,A,u1) + filt(B,A,u2) + filt(B,A,u3) + filt(B,A,u4) + 0.3*lsim(H1,ev1)

H2 = tf([0,0,1],A,1)
ev2 = randn(N)
y2 = filt(B,A,u1) + filt(B,A,u2) + filt(B,A,u3) + filt(B,A,u4) + 0.3*lsim(H2,ev2)

nu = 4
ny = 2
u = reshape([u1;u2;u3;u4],N,nu)
y = reshape([y1;y2],N,2)

T = Float64
Phi = zeros(T,N-1,(nu+1)*nb)
for iu=1:nu
  Phi[:,iu:nu+1:end-nu+iu] = Toeplitz([zeros(nk-1); u[1:end-nk,iu]], [u[1,iu]; zeros(nb-1)])
end
for iy=1:ny
  Phi[:,nu+1:nu+1:end] = Toeplitz(-y[1:end-1,iy], [-y[1,iy]; zeros(nb-1)])
end
Phi = Phi/5/γ

γ  = norm([u[:];y])
Y = y[2:end,1]/5/γ


k = 1
l = nu+1
m = N-1
n = na
p = 0
s = n

println("k: ",k)

job = 'Q'
TC = hcat(u[1:end-1,:],y[1:end-1,1])
ldtc = size(TC)[1]
TR = zeros(Float64,k,(n-1)*l)
ldtr = size(TR)[1]
Q = Array{Float64}(m*k, min(s*l, min(m*k, n*l)-p*l))
ldq = size(Q)[1]
R = Array{Float64}(m, min( s*l, min( m*k,n*l )-p*l ))
ldr = size(R)[1]
DWORK = Array{Float64}(1)
ldwork = -1


ldwork = m*m
Slicot.Raw.mb02jd!(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, -1)
#@benchmark Slicot.Raw.mb02jd!(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, -1)
ldwork = 2*convert(Int,round(DWORK[1]))
DWORK = Array{Float64}(ldwork)
#@benchmark Slicot.Raw.mb02jd!(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, ldwork)


Q, Rt = Slicot.Simple.mb02jd(job, TC, TR)
Slicot.Raw.mb02jd!(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, ldwork)


Rt = UpperTriangular(R[1:l*n,1:l*n]')
Q
b = Q.'*Y
var1 = Rt\b

# simple
k = 1
l = nu+ny
job = 'Q'
col = hcat(u[1:end-1,:],-y[1:end-1,1])/5/γ
row = hcat(col[1:k,1:l], zeros(Float64,k,(na-1)*l))
Q,Rt = Slicot.Simple.mb02jd(job, col, row)

b = Q.'*Y
var1 = Rt\b


# Raw
k = 1
l = nu+ny
m = N-1
n = na
p = 0
s = n
job = 'Q'
TC = Phi[:,1:l]
ldtc = size(TC)[1]
TR = Phi[1,l+1:end]
ldtr = size(TR)[1]
Q = Array{Float64}(m*k, min(s*l, min(m*k, n*l)-p*l))
ldq = size(Q)[1]
R = Array{Float64}(l*n, min( s*l, min( m*k,n*l )-p*l ))
ldr = size(R)[1]
DWORK = Array{Float64}(m*m)
ldwork = length(DWORK)

#Slicot.Simple.mb02jd(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, ldwork)
Slicot.Raw.mb02jd!(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, ldwork)

Rt = UpperTriangular(R[1:(nu+1)*n,1:(nu+1)*n]')

Q
b = Q.'*Y
var1 = Rt\b

#Q*R.'[1:2*n,1:2*n]

#@benchmark qr(Phi)
#@benchmark mb02jd(job, k, l, m, n, p, s, TC, ldtc, TR, ldtr, Q, ldq, R, ldr, DWORK, ldwork)

# Raw JX
k = 1
l = nu+ny
m = N-1
n = na
p = 0
s = n
job = 'Q'
TC = Phi[:,1:l]
ldtc = size(TC)[1]
TR = Phi[1,l+1:end]
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
JPVT = Array{Slicot.BlasInt}(min(m*k,n*l))
Slicot.Raw.mb02jx!(job, k, l, m, n, TC, ldtc, TR, ldtr, rnk, Q, ldq, R, ldr, JPVT, 0.0, 0.0, DWORK, ldwork)

Rt3 = UpperTriangular(R[1:l*n,1:l*n]')
b = (Q.'*Y)
var3[JPVT] = (Rt3\b)

#Simple JX
k = 1
l = nu+ny
job = 'Q'
col = hcat(u[1:end-1,:],-y[1:end-1,1])/5/γ
row = hcat(col[1:k,1:l], zeros(Float64,k,(na-1)*l))
Q, Rt4, jpvt = Slicot.Simple.mb02jx(job, col, row)

b = Q.'*Y
var3 = zeros(size(col)[1])
var3[jpvt] = (Rt4\b)
var3

norm(var1[1:6]-Θ0[1:6])
#norm(var2-Θ0)
norm(var3[1:6]-Θ0[1:6])


@benchmark Slicot.Simple.mb02jd(job, col, row)
@benchmark Slicot.Simple.mb02jx(job, col, row)

A = randn(2,2)
B = randn(2,1)
C = randn(1,2)
D = randn(1,1)
n = 2
m = 1
p = 1
r = tb04ad('R', n, m, p, A, B, C, D)


m = na
iy = 1
γ  = norm([u[:];y])
Y = y[2:end,iy]/5/γ
col = hcat(u[1:end-1,:], -y[1:end-1,iy])/5/γ
row = hcat(col[1:k,1:l],zeros(Float64,k,(m-1)*l))
Q,R,jpvt = Slicot.Simple.mb02jx('Q', col, row)
b = Q.'*Y
theta[jpvt,iy] = R\b
