import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy

plt.rcParams.update({'font.size': 22})
plt.interactive(True)

# read data file
tec=np.genfromtxt("tec.dat", dtype=None,comments="%")

#text='VARIABLES = X Y P U V u2 v2 w2 uv mu_sgs prod'

x=tec[:,0]
y=tec[:,1]
p=tec[:,2]
u=tec[:,3]
v=tec[:,4]
uu=tec[:,5]
vv=tec[:,6]
ww=tec[:,7]
uv=tec[:,8]
k=0.5*(uu+vv+ww)

if max(y) == 1.:
   ni=170
   nj=194
   nu=1./10000.
else:
   nu=1./10595.
   if max(x) > 8.:
     nj=162
     ni=162
   else:
     ni=402
     nj=162

viscos=nu

u2d=np.reshape(u,(nj,ni))
v2d=np.reshape(v,(nj,ni))
p2d=np.reshape(p,(nj,ni))
x2d=np.reshape(x,(nj,ni))
y2d=np.reshape(y,(nj,ni))
uu2d=np.reshape(uu,(nj,ni)) #=mean{v'_1v'_1}
uv2d=np.reshape(uv,(nj,ni)) #=mean{v'_1v'_2}
vv2d=np.reshape(vv,(nj,ni)) #=mean{v'_2v'_2}
k2d=np.reshape(k,(nj,ni))   #=mean{0.5(v'_iv'_i)}

u2d=np.transpose(u2d)
v2d=np.transpose(v2d)
p2d=np.transpose(p2d)
x2d=np.transpose(x2d)
y2d=np.transpose(y2d)
uu2d=np.transpose(uu2d)
vv2d=np.transpose(vv2d)
uv2d=np.transpose(uv2d)
k2d=np.transpose(k2d)


# set periodic b.c on west boundary
#u2d[0,:]=u2d[-1,:]
#v2d[0,:]=v2d[-1,:]
#p2d[0,:]=p2d[-1,:]
#uu2d[0,:]=uu2d[-1,:]


# read k and eps from a 2D RANS simulations. They should be used for computing the damping function f
k_eps_RANS = np.loadtxt("k_eps_RANS.dat")
k_RANS=k_eps_RANS[:,0]
diss_RANS=k_eps_RANS[:,1]
vist_RANS=k_eps_RANS[:,2]

ntstep=k_RANS[0]

k_RANS_2d=np.reshape(k_RANS,(ni,nj))/ntstep       # modeled turbulent kinetic energy
diss_RANS_2d=np.reshape(diss_RANS,(ni,nj))/ntstep # modeled dissipation
vist_RANS_2d=np.reshape(vist_RANS,(ni,nj))/ntstep # turbulent viscosity, AKN model

# set small values on k & eps at upper and lower boundaries to prevent NaN on division
diss_RANS_2d[:,0]= 1e-10
k_RANS_2d[:,0]= 1e-10
vist_RANS_2d[:,0]= nu
diss_RANS_2d[:,-1]= 1e-10
k_RANS_2d[:,-1]= 1e-10
vist_RANS_2d[:,-1]= nu

# set Neumann of p at upper and lower boundaries
p2d[:,1]=p2d[:,2]
p2d[:,-1]=p2d[:,-1-1]

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("xc_yc.dat")
xf=xc_yc[:,0]
yf=xc_yc[:,1]
xf2d=np.reshape(xf,(nj,ni))
yf2d=np.reshape(yf,(nj,ni))
xf2d=np.transpose(xf2d)
yf2d=np.transpose(yf2d)

# delete last row
xf2d = np.delete(xf2d, -1, 0)
yf2d = np.delete(yf2d, -1, 0)
# delete last columns
xf2d = np.delete(xf2d, -1, 1)
yf2d = np.delete(yf2d, -1, 1)

# compute the gradient dudx, dudy at point P
dudx= np.zeros((ni,nj))
dudy= np.zeros((ni,nj))
dvdx= np.zeros((ni,nj))
dvdy= np.zeros((ni,nj))

dudx,dudy=dphidx_dy(xf2d,yf2d,u2d)
dvdx,dvdy=dphidx_dy(xf2d,yf2d,v2d)

################################ vector plot
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
k=6# plot every forth vector
ss=3.2 #vector length
plt.quiver(x2d[::k,::k],y2d[::k,::k],u2d[::k,::k],v2d[::k,::k],width=0.01)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("vector plot")
plt.savefig('vect_python.eps')

################################ contour plot
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("contour pressure plot")
plt.savefig('piso_python.eps')

################################ contour plot
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.contourf(x2d,y2d,k_RANS_2d, 50)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("contour k RANS plot")
plt.savefig('k_rans.eps')

#************
# plot uv
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i=10
plt.plot(uv2d[i,:],y2d[i,:],'b-')
plt.xlabel('$\overline{u^\prime v^\prime}$')
plt.ylabel('y/H')
plt.savefig('uv_python.eps')

#%%%%%%%%%%%%%%%%%%%%% grid
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
for i in range (0,ni):
   plt.plot(x2d[i,:],y2d[i,:])

for j in range (0,nj):
   plt.plot(x2d[:,j],y2d[:,j])

#plt.axis([0,5,0,5])
plt.title('grid')
plt.savefig('grid_python.eps')


plt.show()

#  Q1.4

duudx,duudy = dphidx_dy(xf2d,yf2d,uu2d)
duudx_dx,duudx_dy = dphidx_dy(xf2d,yf2d,duudx)
duudy_dx,duudy_dy = dphidx_dy(xf2d,yf2d,duudy)

dvvdx,dvvdy = dphidx_dy(xf2d,yf2d,vv2d)

duvdx,duvdy = dphidx_dy(xf2d,yf2d,uv2d)
duvdx_dx,duvdx_dy = dphidx_dy(xf2d,yf2d,duvdx)
duvdy_dx,duvdy_dy = dphidx_dy(xf2d,yf2d,duvdy)

C_nu = 0.09
C1 = 1.5
C2 = 0.6
C1_w = 0.5
C2_w = 0.3
sig_k = 1
rho = 1

x_pos = 10

i = 1
j = 1
mu = 1/10595
visc_diff_11 = mu*(np.add(duudx_dx, duudy_dy))
P_11 = -2*np.add(np.multiply(uu2d, dudx), np.multiply(uv2d, dudy))
P_strain = 0

i = 1
j = 2
mu = 1/10595
visc_diff_12 = mu*(np.add(duvdx_dx, duvdy_dy))
P_12 = -np.add(np.add(np.multiply(uu2d, dvdx), np.multiply(uv2d, dvdy)), np.add(np.multiply(uv2d, dudx), np.multiply(vv2d, dudy)))
P_strain_12_2 = -C2*rho*P_12
eps = diss_RANS_2d

P_strain_12 = P_strain_12_2*(1-3/2*C2_w*f) - 3/2*C1_w*eps/k*uv2d*f - C1*rho*eps/k2d*uv2d
