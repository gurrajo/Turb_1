import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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



#  Q1.4

duuudx, duuudy = dphidx_dy(xf2d, yf2d, uu2d*u2d)
duuvdx, duuvdy = dphidx_dy(xf2d, yf2d, uu2d*v2d)

duvudx, duvudy = dphidx_dy(xf2d, yf2d, uv2d*u2d)
duvvdx, duvvdy = dphidx_dy(xf2d, yf2d, uv2d*v2d)

rey_stress_11 = np.add(duuudx, duuvdy)
rey_stress_12 = np.add(duvudx, duvvdy)

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
eps = diss_RANS_2d

i = 1
j = 1
mu = 1/10595
visc_diff_11 = mu*(np.add(duudx_dx, duudy_dy))
P_11 = -2*np.add(np.multiply(uu2d, dudx), np.multiply(uv2d, dudy))
P_strain = 0

duudx_eddy = np.zeros((ni,nj))
duudy_eddy = np.zeros((ni,nj))
for i in range(ni):
    for j in range(nj):
        duudx_eddy[i, j] = duudx[i, j] * (C_nu * k_RANS_2d[i, j] ** 2 / (eps[i, j] * sig_k))
        duudy_eddy[i, j] = duudy[i, j] * (C_nu * k_RANS_2d[i, j] ** 2 / (eps[i, j] * sig_k))

duudx_eddy_dx, duudx_eddy_dy = dphidx_dy(xf2d,yf2d,duudx_eddy)
duudy_eddy_dx, duudy_eddy_dy = dphidx_dy(xf2d,yf2d,duudy_eddy)


D_11 = np.add(duudx_eddy_dx, duudy_eddy_dy)
eps_11 = -2/3*eps
sum_terms_11 = visc_diff_11+P_11+D_11+eps_11

fig1,ax1 = plt.subplots()
plt.plot(visc_diff_11[x_pos, :], y2d[x_pos, :])
plt.plot(P_11[x_pos, :], y2d[x_pos, :])
plt.plot(D_11[x_pos, :], y2d[x_pos, :])
plt.plot(eps_11[x_pos, :], y2d[x_pos, :])
plt.ylabel("$y$")
plt.title("all Reynolds stress terms for i=j=1", fontsize=15)
plt.legend(['Visc_diff','Production','Turb_diff','Dissipation'],prop={'size': 10})
plt.grid()
plt.savefig('stresses_11.eps')

fig1,ax1 = plt.subplots()
plt.plot(visc_diff_11[x_pos, :], y2d[x_pos, :])
plt.plot(P_11[x_pos, :], y2d[x_pos, :])
plt.plot(D_11[x_pos, :], y2d[x_pos, :])
plt.plot(eps_11[x_pos, :], y2d[x_pos, :])
plt.ylim([0,0.05])
plt.xlim([-0.05,0.05])
plt.ylabel("$y$")
plt.title("all Reynolds stress terms for i=j=1", fontsize=15)
plt.legend(['Visc_diff','Production','Turb_diff','Dissipation'],prop={'size': 8})
plt.grid()
plt.savefig('stresses_11_zoomed.eps')


i = 1
j = 2
mu = 1/10595
visc_diff_12 = mu*(np.add(duvdx_dx, duvdy_dy))
P_12 = -(uu2d*dvdx + uv2d*dvdy) - (uv2d*dudx + vv2d*dudy)
P_strain_12_2 = -C2*rho*P_12

def n_i(x,y):
    dist = np.zeros((ni, 2))
    for i in range(ni):
        dist[i, 0] = np.sqrt((x-x2d[i, 0])**2 + (y-y2d[i, 0])**2)
        dist[i, 1] = np.sqrt((x - x2d[i, -1]) ** 2 + (y - y2d[i, -1]) ** 2)
    return np.min(dist)

f = np.zeros((nj,1))
P_strain_12 = np.zeros((nj, 1))
for j in range(nj):
    if j == 0 or j == nj-1:
        f[j] = 1
    else:
        dist = n_i(x2d[x_pos, j], y2d[x_pos, j])
        f[j] = np.min([k_RANS_2d[x_pos, j]**(3/2)/(2.55*dist*eps[x_pos, j]), 1])

    P_strain_12[j] = P_strain_12_2[x_pos,j]*(1-(3/2)*C2_w*f[j]) - (3/2)*(C1_w*eps[x_pos, j]/k_RANS_2d[x_pos, j])*uv2d[x_pos, j]*f[j] - (C1*rho*eps[x_pos,j]/k_RANS_2d[x_pos,j])*uv2d[x_pos,j]

duvdx_eddy = np.zeros((ni, nj))
duvdy_eddy = np.zeros((ni, nj))
for i in range(ni):
    for j in range(nj):
        duvdx_eddy[i, j] = duvdx[i, j] * (C_nu * k_RANS_2d[i, j] ** 2 / (eps[i, j] * sig_k))
        duvdy_eddy[i, j] = duvdy[i, j] * (C_nu * k_RANS_2d[i, j] ** 2 / (eps[i, j] * sig_k))

duvdx_eddy_dx, duvdx_eddy_dy = dphidx_dy(xf2d,yf2d,duvdx_eddy)
duvdy_eddy_dx, duvdy_eddy_dy = dphidx_dy(xf2d,yf2d,duvdy_eddy)

D_12 = np.add(duvdx_eddy_dx, duvdy_eddy_dy)

eps_12 = 0
sum_terms_12 = visc_diff_12[x_pos, :] + P_12[x_pos, :] + D_12[x_pos, :] + np.transpose(P_strain_12)
fig1,ax1 = plt.subplots()
plt.plot(visc_diff_12[x_pos, :], y2d[x_pos, :])
plt.plot(P_12[x_pos, :], y2d[x_pos, :])
plt.plot(D_12[x_pos, :], y2d[x_pos, :])
plt.plot(P_strain_12, y2d[x_pos, :])
plt.ylabel("$y$")
plt.title("all Reynolds stress terms for i=1, j=2", fontsize=15)
plt.legend(['Visc_diff','Production','Turb_diff','Press_strain'],prop={'size': 6})
plt.grid()
plt.savefig('stresses_12.eps')

fig1,ax1 = plt.subplots()
plt.plot(visc_diff_12[x_pos, :], y2d[x_pos, :])
plt.plot(P_12[x_pos, :], y2d[x_pos, :])
plt.plot(D_12[x_pos, :], y2d[x_pos, :])
plt.plot(P_strain_12, y2d[x_pos, :])
plt.ylim([0,0.05])
plt.xlim([-0.05,0.05])
plt.ylabel("$y$")
plt.title("all Reynolds stress terms for i=1, j=2", fontsize=15)
plt.legend(['Visc_diff','Production','Turb_diff','Press_strain'],prop={'size': 6})
plt.grid()
plt.savefig('stresses_12_zoomed.eps')





# 1.8:
bouss_11 = np.zeros((ni,nj))
bouss_12 = np.zeros((ni,nj))
S_11 = dudx
S_12 = 1/2*np.add(dudy, dvdx)
for i in range(ni):
    for j in range(nj):
        bouss_11[i, j] = -2*C_nu*k_RANS_2d[i, j]**2/eps[i, j]*S_11[i, j] + k_RANS_2d[i, j]*2/3
        bouss_12[i, j] = -2 * C_nu * k_RANS_2d[i, j] ** 2 / eps[i, j] * S_12[i, j]

bouss_duuudx, bouss_duuudy = dphidx_dy(xf2d, yf2d, uu2d*u2d)
bouss_duuvdx, bouss_duuvdy = dphidx_dy(xf2d, yf2d, uu2d*v2d)

bouss_duvudx, bouss_duvudy = dphidx_dy(xf2d, yf2d, uv2d*u2d)
bouss_duvvdx, bouss_duvvdy = dphidx_dy(xf2d, yf2d, uv2d*v2d)

bouss_stress_11 = np.add(bouss_duuudx, bouss_duuvdy)
bouss_stress_12 = np.add(bouss_duvudx, bouss_duvvdy)

fig1,ax1 = plt.subplots()
plt.plot(bouss_stress_11[x_pos, :], y2d[x_pos, :])
plt.plot(bouss_stress_12[x_pos, :], y2d[x_pos, :])
plt.plot(rey_stress_11[x_pos, :], y2d[x_pos, :])
plt.plot(rey_stress_12[x_pos, :], y2d[x_pos, :])
#plt.plot(sum_terms_11[x_pos, :], y2d[x_pos, :])
#plt.plot(np.transpose(sum_terms_12), y2d[x_pos, :])
plt.ylabel("$y$")
plt.title("Stresses", fontsize=15)
plt.legend(['Bouss_11','Bouss_12', 'rey_stress_11', 'rey_stress_12', 'sum_terms_11', 'sum_terms_12'], prop={'size': 6})
plt.grid()
plt.savefig('bouss_stresses.eps')

# fig1,ax1 = plt.subplots()
# #plt.plot(bouss_stress_11[x_pos, :], y2d[x_pos, :])
# #plt.plot(bouss_stress_12[x_pos, :], y2d[x_pos, :])
# plt.plot(rey_stress_11[x_pos, :], y2d[x_pos, :])
# plt.plot(rey_stress_12[x_pos, :], y2d[x_pos, :])
# plt.plot(sum_terms_11[x_pos, :], y2d[x_pos, :])
# plt.plot(np.transpose(sum_terms_12), y2d[x_pos, :])
# plt.ylim([0,0.2])
# plt.xlim([-0.5,0.5])
# plt.ylabel("$y$")
# plt.title("Stresses", fontsize=15)
# plt.legend(['Bouss_11', 'Bouss_12', 'rey_stress_11', 'rey_stress_12', 'sum_terms_11', 'sum_terms_12'], prop={'size': 6})
# plt.grid()
# plt.savefig('bouss_stresses_zoom.eps')

# 1.9:
P_k = -np.add(np.add(np.multiply(uu2d, dudx), np.multiply(uv2d, dudy)), np.add(np.multiply(vv2d, dvdy), np.multiply(uv2d, dvdx)))
fig1,ax1 = plt.subplots()
ax1.set_facecolor((0,0,0))
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.contourf(x2d,y2d,P_k, levels=np.linspace(-0.1, -0.0001, 20))
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Production plot", fontsize=15)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
plt.savefig('Pk.eps')

# 1.10:
S_22 = dvdx
lam_1 = np.zeros((ni,nj))
for i in range(ni):
    for j in range(nj):
        S = np.zeros((2, 2))
        S[0, 0] = S_11[i, j]
        S[0, 1] = S_12[i, j]
        S[1, 0] = S_12[i, j]
        S[1, 1] = S_22[i, j]
        W, v = np.linalg.eig(S)
        lam_1[i, j] = W[0]
upper_lim = np.divide(k_RANS_2d, (3*np.abs(lam_1)))
nu_t_limit = np.clip(vist_RANS_2d, None, upper_lim)
limit_effect = np.subtract(vist_RANS_2d, nu_t_limit)

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.contourf(x2d,y2d,limit_effect, 50)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("contour limit effect plot", fontsize=15)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=9)
plt.savefig('limit_effect.eps')

plt.show()
