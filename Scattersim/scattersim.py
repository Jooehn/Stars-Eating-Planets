"""
Created on Tue Mar 26 13:22:07 2019

@author: jooehn

Script that generates new velocities for two planets that undergo scattering.

The main purpose of this code is to restrict the parameter space for which
a planet-host star collision becomes more probable. We have included two major
simplifications, as the contributions from the host star are neglected and 
one of the orbits are kept completely circular, which means that the results
should be analysed with this in mind.
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.patches import Arc
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.size']= 16
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['mathtext.fontset'] = 'cm'

class Scatter:
    def __init__(self,p1data,p2data,M_star,theta=0):
    
        self.M_s    = M_star
        self.G      = 4*np.pi**2
        self.theta  = np.deg2rad(theta)
        self.rjtoau = 1/2150
        
        kmtoau = 1/149597900
        kmstoauyr = (3600*24*365.25)*kmtoau 
        self.auyrtokms = 1/kmstoauyr
        
        self.pdata = np.vstack([p1data,p2data])
        
        self.dcoll = self.pdata[0,3] + self.pdata[1,3]
        self.q     = self.pdata[:,2]/M_star
        
        self.get_phi_isec()
        self.get_r2()
        self.calc_v()
        self.calc_rhill()
        
    def calc_rhill(self):
        a,e,m,_ = self.pdata.T
        
        self.rhill = a*(1-e)*(m/(3*self.M_s))**(1/3)
        
    def get_phi_isec(self):
        """Obtains the angle of intersection of the orbits given the planet data"""
        
        #As we have assumed that one of the orbits are circular, we only need to
        #find the angle for the eccentric orbit
        
        a1, e1, m1,_ = self.pdata[0]
        a2, e2, m2,_ = self.pdata[1]
        
        phic = np.arccos(((a2*(1-e2**2))/a1-1)/e2)
    
        self.phic = np.zeros((2,2))
    
        self.phic[0] +=  phic+self.theta,phic
        self.phic[1] += -phic+self.theta,-phic
    
    def get_r2(self):
        """Obtains the distance from the centre of mass at the points of intersection
        between two orbits. One of them must have zero eccentricity."""
        
        #Mainly used as a check for the formulae
        
        a2, e2, m2, _ = self.pdata[1]
        
        #Formula from Windmark (2009)
        self.rc = (a2*(1-e2**2))/(1+e2*np.cos(self.phic[0,1]))
        
    def calc_v(self):
        """Computes the velocity vectors of the two planets in our system"""
        
        a,e,m,_ = self.pdata.T
        
        #We calculate the radial and tangential velocities at the two crossing
        #points we have found
        
        vrc1 = e*np.sin(self.phic[0])*np.sqrt(((self.G*self.M_s)/(a*(1-e**2))))
        vtc1 = np.sqrt(((self.G*self.M_s)/(a*(1-e**2))))*(1+e*np.cos(self.phic[0]))
        
        vrc2 = e*np.sin(self.phic[1])*np.sqrt(((self.G*self.M_s)/(a*(1-e**2))))
        vtc2 = np.sqrt(((self.G*self.M_s)/(a*(1-e**2))))*(1+e*np.cos(self.phic[1]))
        
        #Fix such that functions support additional crossing points
        
        self.v1 = np.zeros((2,2))
        self.v2 = np.zeros((2,2))
        
        self.v1[0]  += vrc1[0],vtc1[0]
        self.v1[1]  += vrc2[0],vtc2[0]
        self.v2[0]  += vrc1[1],vtc1[1]
        self.v2[1]  += vrc2[1],vtc2[1]
        
        self.vrel   = self.v1 - self.v2
        self.vcm    = (m[0]*self.v1+m[1]*self.v2)/(m.sum())
        
        self.vb1    = self.v1 - self.vcm
        self.vb2    = self.v2 - self.vcm
        
        #We also save the initial energy
        
        v1normsq = self.v1[0,0]**2+self.v1[0,1]**2
        v2normsq = self.v2[0,0]**2+self.v2[0,1]**2
        
        self.L1 = m1*self.v1[0,1]*self.rc
        self.L2 = m2*self.v2[0,1]*self.rc
        
        self.E1 = 0.5*m[0]*v1normsq - self.G*m[0]*self.M_s/self.rc
        self.E2 = 0.5*m[1]*v2normsq - self.G*m[1]*self.M_s/self.rc
        
        self.L = self.L1+self.L2
        self.E = self.E1+self.E2
        
    def get_defang(self,b):        
        """Finds the deflection angle given one or a set of impact parameter
        values"""
        
        m1, m2 = self.pdata[:,2]
        
        vnormsq = (self.vrel[0,0]**2+self.vrel[0,1]**2)
        
        psi = np.arctan((b*vnormsq)/(self.G*(m1+m2)))
        
        self.defang = (np.pi - 2*psi)
        
        #Furthermore, knowing b anv vrel, we can obtain the distance at closest
        #approach between the planet
        
        #We first find the eccentricity of the close encounter
        
        e = np.sqrt(1+(b**2*vnormsq**2)/(self.G**2*(m1+m2)))
        
        #Then the minimum distance becomes
        
        self.dmin = (b**2*vnormsq**2)/(self.G*(m1+m2)*(1+e))
        
    def scatter(self,N=1,b=None):
        """Performs the scattering between the two planets in our system and
        computes the new velocities."""
        
        if b is None:
            bmax = self.rc*0.05
            b = np.random.uniform(-bmax,bmax,N)
        
        self.get_defang(b)
        
        self.b = b
        
        #We first calculate the new velocities w.r.t. the centre of mass
        
        mrot1 = np.array([[np.cos(self.defang),-np.sin(self.defang)],\
                         [np.sin(self.defang),np.cos(self.defang)]]).T
        
        mrot2 = np.array([[np.cos(self.defang),-np.sin(self.defang)],\
                         [np.sin(self.defang),np.cos(self.defang)]]).T
        
        self.vb1n = np.zeros((np.size(b),2,2))
        self.vb2n = np.zeros((np.size(b),2,2))
          
        self.vb1n[:,0] = np.matmul(mrot1,self.vb1[0])
        self.vb1n[:,1] = np.matmul(mrot1,self.vb1[1])
        
        self.vb2n[:,0] = np.matmul(mrot2,self.vb2[0])
        self.vb2n[:,1] = np.matmul(mrot2,self.vb2[1])
        
        #We can then easily obtain the new velocities for each planet
        
        self.v1n = self.vb1n + self.vcm
        self.v2n = self.vb2n + self.vcm
        
        #Now, we have all the information we need to compute the new orbital
        #parameters
        
        L1 = m1*self.v1n[:,0,1]*self.rc
        L2 = m2*self.v2n[:,0,1]*self.rc
        
        v1norm = np.linalg.norm(self.v1n[:,0],axis=1)
        v2norm = np.linalg.norm(self.v2n[:,0],axis=1)
        
        E1 = 0.5*m1*v1norm**2-self.G*m1*self.M_s/self.rc
        E2 = 0.5*m2*v2norm**2-self.G*m2*self.M_s/self.rc
        
        self.at1 = -(self.G*m1*self.M_s)/(2*E1)
        self.at2 = -(self.G*m2*self.M_s)/(2*E2)
        
        self.et1 = np.sqrt(1+(2*E1*L1**2)/(self.G**2*m1**3*self.M_s**2))
        self.et2 = np.sqrt(1+(2*E2*L2**2)/(self.G**2*m2**3*self.M_s**2))
        
        #We check if energy and angular momentum is conserved
        
        self.L1n = L1
        self.L2n = L2
        self.E1n = E1
        self.E2n = E2
        
        self.Ln = L1+L2
        self.En = E1+E2
        
        self.dL = self.L-self.Ln
        self.dE = (self.E-self.En)/self.E
        
        if not np.allclose(self.dL,0) & np.allclose(self.dE,0):
            
            print('dE or dL is not conserved')
        
    def plot_orbit(self):
        """Plots the circular and eccentric orbits and marks the point of crossing."""
        
        a1, e1, m1, _ = self.pdata[0]
        a2, e2, m2, _ = self.pdata[1]
        
        ang = np.linspace(0,2*np.pi,1000)
        
        x1 = a1*np.cos(ang)
        y1 = a1*np.sin(ang)
        
        #We work out the semi-latus rectum
        p = a2*(1-e2**2)
        
        #This yields the following r for our orbit
        rvals = p/(1+e2*np.cos(ang))
        
        #We then find the corresponding x and y coordinates of the eccentric orbit
        x2 = rvals*np.cos(ang)
        y2 = rvals*np.sin(ang)
        
        #Finally we compute the coordinates of the orbit crossing
        xc1 = self.rc*np.cos(self.phic[0,1])
        yc1 = self.rc*np.sin(self.phic[0,1])
        
        xc2 = self.rc*np.cos(self.phic[1,1])
        yc2 = self.rc*np.sin(self.phic[1,1])
        
        fig, ax = plt.subplots(figsize=(8,6))
        
        ax.plot(x1,y1,'b-',label='$\mathrm{Orbit\ 1}$')
        ax.plot(x2,y2,'r-',label='$\mathrm{Orbit\ 2}$')
        ax.plot(0,0,marker='+',color='tab:gray',ms=10)
        ax.plot(xc1,yc1,'k+',markersize=7,label='$r_1 = r_2$')
        ax.plot(xc2,yc2,'k+',markersize=7) #Due to symmetry, we get two crossings
        
        ax.set_aspect('equal')
        
        xmax = int(np.ceil(np.amax(np.absolute([x1,x2]))))
        ymax = int(np.ceil(np.amax(np.absolute([x1,x2]))))
        
        ax.set_xlim(-xmax,xmax)
        ax.set_ylim(-ymax,ymax)
        ax.set_yticks(np.arange(-ymax,ymax+1,1))
        ax.set_xlabel('$x\ \mathrm{[AU]}$')
        ax.set_ylabel('$y\ \mathrm{[AU]}$')
        
        ax.legend(prop={'size':13})
        
        self.add_ptable(ax)
        self.add_date(fig)
        
    def plot_vectri(self,planet=1):
        """Plots the vector triangle for a given planet after performing a 
        scattering with impact parameter b."""
        
        fig, ax = plt.subplots(figsize=(10,8))
        
        #We extract the information we need
        
        idx = 0
        
        if planet == 1:
            
            v   = self.v1[idx]
            vb  = self.vb1[idx]
            vbn = self.vb1n[0,idx]
            vcm = self.vcm[idx]
            vn  = self.v1n[0,idx]
            
            pcol = 'b'
            
        elif planet == 2:

            v   = self.v2[idx]
            vb  = self.vb2[idx]
            vbn = self.vb2n[0,idx]
            vcm = self.vcm[idx]
            vn  = self.v2n[0,idx]
        
            pcol = 'r'
        
        #We then compute relevant values such as vector magnitudes
        
        vbnorm = np.linalg.norm(vb)
        
        #We also make a circle showing all possible values that the new c.o.m
        #velocity can take
        
        ang = np.linspace(0,2*np.pi,1000)
        
        xc = vbnorm*np.cos(ang)+vcm[1]
        yc = vbnorm*np.sin(ang)+vcm[0]
        
        #Plots vb1
        vbp, = ax.plot([-vb[1],0],[-vb[0],0],ls='--',color='g',lw=1.5,\
                       label='$v_{b,'+'{}'.format(planet)+'}$')
        self.add_arrow(vbp,direction='right')
        #Plots v
        vp, = ax.plot([-vb[1],vcm[1]],[-vb[0],vcm[0]],ls='-',color='g',lw=1.5,\
                      label='$v_'+'{}'.format(planet)+'}$')
        
        self.add_arrow(vp)
        #Plots vcm
        vcmp, = ax.plot([0,vcm[1]],[0,vcm[0]],'-k',lw=1.5,\
                        label='$v_\mathrm{cm}$')
        self.add_arrow(vcmp)
        #Plots the position of vb in the circle
        ax.plot([vcm[1],vcm[1]+vb[1]],[vcm[0],vcm[0]+vb[0]],linestyle='--',color='tab:gray',lw=1.5)
        #Plots circle of possible vbn values
        ax.plot(xc,yc,'k-',lw=1.5)
        #Plots vbn
        vbnp, = ax.plot([vcm[1],vn[1]],[vcm[0],vn[0]],color='m',ls='--',lw=1.5,\
                        label=r'$\tilde{v}_'+'{b,'+'{}'.format(planet)+'}$')
        self.add_arrow(vbnp,direction='right')
        #Plots vn
        vnp, = ax.plot([0,vn[1]],[0,vn[0]],color='m',ls='-',lw=1.5,\
                       label=r'$\tilde{v}_'+'{}'.format(planet)+'}$')
        self.add_arrow(vnp,direction='right')
            
        #We also plot a vector pointing towards the position of the host star
        
        xhs, yhs = np.cos(self.phic[idx,0]+np.pi),np.sin(self.phic[idx,0]+np.pi)
        
        hsp, = ax.plot([0,xhs],[0,yhs],'-',color='tab:gray',\
                       label=r'$\hat{r}_{\star}$')
        self.add_arrow(hsp,position=xhs)
        
        #Plots a few markers in the figure
        ax.plot(0,0,marker='+',color='tab:gray',ms=12,mew=1.3,label='$\mathrm{Centre\ of\ mass}$')
        ax.plot(-vb[1],-vb[0],marker='o',color=pcol,ms=8,label='$m_{}$'.format(planet))
        ax.plot(vcm[1],vcm[0],'ko',ms=2)
        
        #Finally, we adjust the axes, add labels and a title
        ax.set_xlabel('$v_t\ \mathrm{[AU\ yr}^{-1}]$')
        ax.set_ylabel('$v_r\ \mathrm{[AU\ yr}^{-1}]$')
#        ax.set_title('$\mathrm{Vector\ triangle\ for\ a\ scattering\ with\ }b =' + '{0:.0f}'\
#                     .format(self.b/self.rjtoau)+'\ \mathrm{R}_J$')
        
#        vp, = ax.plot([0,vcm[1]-vb[1]],[0,vcm[0]-vb[0]],'b-.',lw=1.5)
        
        ymax = np.ceil(np.amax(np.absolute([vcm[0]-vb[0],vcm[0]+vb[0]])))
        if ymax < 3:
            ymax = 3
        
            ax.set_ylim(-ymax,ymax)
            
        ylim = ax.get_ylim()
        
        ax.set_ylim(ylim[0],ylim[1]-0.01)

        #We add information regarding the impact parameter
        xmax = ax.get_xlim()[1]
        ax.text(xmax+.5,ymax-0.3,'$b =' + '{0:.0f}'.format(self.b/self.rjtoau)+\
                '\ \mathrm{R}_J$',bbox=dict(facecolor='None', alpha=0.5))
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':12})
        ax.set_aspect('equal')  
        
        self.add_ptable(ax)
        self.add_date(fig)
        
    def plot_vels(self,planet):
        """Plots the velocity vectors and their new components for both planets"""
        fig, ax = plt.subplots(figsize=(8,6))
        
        idx = 0
            
        v1   = self.v1[idx]*self.auyrtokms
        vb1  = self.vb1[idx]*self.auyrtokms
        vb1n = self.vb1n[0,idx]*self.auyrtokms
        vcm = self.vcm[idx]*self.auyrtokms
        v1n  = self.v1n[0,idx]*self.auyrtokms
            
        v2   = self.v2[idx]*self.auyrtokms
        vb2  = self.vb2[idx]*self.auyrtokms
        vb2n = self.vb2n[0,idx]*self.auyrtokms
        v2n  = self.v2n[0,idx]*self.auyrtokms
        
        vbold1, = ax.plot([0,vb1[1]],[0,vb1[0]],'r--',label=r'$v_{b,1}$')
        vbnew1, = ax.plot([0,vb1n[1]],[0,vb1n[0]],'g--',label=r'$\tilde{v}_{b,1}$')
        
        vold1, = ax.plot([0,v1[1]],[0,v1[0]],'r-',label=r'$v_{1}$')
        vnew1, = ax.plot([0,v1n[1]],[0,v1n[0]],'g-',label=r'$\tilde{v}_{1}$')
        
        vcmo1, = ax.plot([vb1[1],vb1[1]+vcm[1]],[vb1[0],vb1[0]+vcm[0]],'--',color='tab:gray')
        vcmn1, = ax.plot([vb1n[1],vb1n[1]+vcm[1]],[vb1n[0],vb1n[0]+vcm[0]],'--',color='tab:gray')
        
        vbold2, = ax.plot([0,vb2[1]],[0,vb2[0]],'b--',label=r'$v_{b,2}$')
        vbnew2, = ax.plot([0,vb2n[1]],[0,vb2n[0]],'c--',label=r'$\tilde{v}_{b,2}$')
        
        vold2, = ax.plot([0,v2[1]],[0,v2[0]],'b-',label=r'$v_{2}$')
        vnew2, = ax.plot([0,v2n[1]],[0,v2n[0]],'c-',label=r'$\tilde{v}_{2}$')
        
        vcmo2, = ax.plot([vb2[1],vb2[1]+vcm[1]],[vb2[0],vb2[0]+vcm[0]],'--',color='tab:gray')
        vcmn2, = ax.plot([vb2n[1],vb2n[1]+vcm[1]],[vb2n[0],vb2n[0]+vcm[0]],'--',color='tab:gray')
        
        
        self.add_arrow(vbnew1)
        self.add_arrow(vbold1)
        
        self.add_arrow(vnew1)
        self.add_arrow(vold1)
        
        self.add_arrow(vbnew2)
        self.add_arrow(vbold2)
        
        ax.set_aspect('equal')
        
        xmax = np.ceil(np.amax(np.absolute([v1[1],vb1[1],v1n[1],vb1n[1]])))
        ymax = np.ceil(np.amax(np.absolute([v1[0],vb1[0],v1n[0],vb1n[0]])))
        
        maxv =int(np.max([xmax,ymax]))
        
        ax.set_xlabel('$v_t\ \mathrm{[km\ s}^{-1}]$')
        ax.set_ylabel('$v_r\ \mathrm{[km\ s}^{-1}]$')
        ax.set_xlim(-maxv,maxv)
        ax.set_ylim(-maxv,maxv)
        
        ax.grid(True)
        
        ax.legend()
        
    def plot_defang_dmin(self):
        """Plots the deflection angle as a function of the impact parameter and
        the minimum distance as a function of the deflection angle"""
        fig, ax = plt.subplots(1,2,sharey=False,figsize=(14,6))
        
        #We use a different colour for the impact parameters for which we get
        #a collision between the planets
        
        col = np.asarray(['k']*np.size(self.b))
        
        col[self.dcoll>=self.dmin] = 'r'
        
        ax[0].scatter(self.b,np.rad2deg(self.defang),c=col,s=2)
        
        ax[0].set_xlabel('$b\ [\mathrm{AU}]$')
        ax[0].set_ylabel(r'$\delta\ [\degree]$')
      
        rmark = ax[0].scatter([],[],c='r',s=3,marker='o',label='$d_{min}\leq r_1+r_2$')
        kmark = ax[0].scatter([],[],c='k',s=3,marker='o',label='$d_{min}>r_1+r_2$')
        
        ax[0].legend(handles=[rmark,kmark],prop={'size':13})
        
        #We also plot the minimum distance as a function of the deflection angle
        
        ax[1].scatter(np.rad2deg(self.defang),self.dmin/self.rjtoau,c=col,s=2)
        
        ax[1].set_ylabel('$d_{min}\ [R_J]$')
        ax[1].set_xlabel(r'$\delta\ [\degree]$')
        
        ax[1].set_xlim(0,360)
        ax[1].set_ylim(0.5*self.dcoll/self.rjtoau,(self.dmin/self.rjtoau).max())
        ax[1].set_yscale('log')
        
        self.add_ptable(ax[0])
        self.add_date(fig)
        
#        fig.subplots_adjust(left=0.2,bottom=0.2)
        
    def plot_new_orb(self,bvals):
        """Plots the new orbital elements after scattering"""
        
        #We save the combined radius of the planets to set up a check for 
        #physical collisions between the planets
        
        self.scatter(b = bvals)
        
        #We mark the scattering events that lead to collision with red, while
        #the rest are black
        
        col = np.asarray(['k']*np.size(bvals))
        
        col[self.dcoll>=self.dmin] = 'r'
        
        #We then set up the plotting parameters
        
        xmax1 = np.amax(self.at1)+0.1*np.amax(self.at1)
        ymax1 = np.amax(self.et1)+0.1*np.amax(self.et1)
        
        xmax2 = np.amax(self.at2)+0.1*np.amax(self.at2)
        ymax2 = np.amax(self.et2)+0.1*np.amax(self.et2)
        
        dx1 = 1
        dx2 = 1
        dy1 = 0.1
        dy2 = 0.1
        
        if xmax1 > 10:
            dx1 = 5
            xmax1 = 10
        if xmax2 >10:
            dx2 = 5
            xmax2 = 10
        if ymax1 > 1:
            dy1 = 0.5
        if ymax2 > 1:
            dy2 = 0.5
            
        #Next, we plot the relevant data
        
        fig, ax = plt.subplots(1,2,sharey=False,figsize=(12,6))
        
        ax[0].scatter(self.at1,self.et1,c=col,s=1,marker='o')
        ax[1].scatter(self.at2,self.et2,c=col,s=1,marker='o')
            
        ax[0].set_xlabel(r'$\tilde{a}_1\ \mathrm{[AU]}$')
        ax[0].set_ylabel(r'$\tilde{e}_1$')
        if any(abs(xmax1-self.at1)>1):
            ax[0].set_xlim(0,xmax1)
            ax[0].set_xticks(np.arange(0,int(xmax1)+dx1,dx1))
        ax[0].set_ylim(0,ymax1)
        if ymax1 > 0.1:
            ax[0].set_yticks(np.arange(0,np.around(ymax1,1)+dy1,dy1))
        
        ax[1].set_xlabel(r'$\tilde{a}_2\ \mathrm{[AU]}$')
        ax[1].set_ylabel(r'$\tilde{e}_2$')
        if any(abs(xmax2-self.at2)>1):
            ax[1].set_xlim(0,xmax2)
            ax[1].set_xticks(np.arange(0,int(xmax2)+dx2,dx2))
        ax[1].set_ylim(0,ymax2)
        if ymax2 > 0.1:
            ax[1].set_yticks(np.arange(0,np.around(ymax2,1)+dy2,dy2))
        
        #Handles for our legend
        rmark = ax[1].scatter([],[],c='r',s=3,marker='o',label='$d_{min}\leq r_1+r_2$')
        kmark = ax[1].scatter([],[],c='k',s=3,marker='o',label='$d_{min}>r_1+r_2$')
        
        ax[1].legend(handles=[rmark,kmark],prop={'size':14})
        
        self.add_ptable(ax[0])
        self.add_date(fig)
        
        #We also make a plot with the eccentricities as a function of the 
        #impact parameter b
        
        fig2, ax2 = plt.subplots(figsize=(10,6))
        
        ax2.plot(self.b,self.et1,color='b',label='$\mathrm{Orbit\ 1}$')
        ax2.plot(self.b,self.et2,color='r',label='$\mathrm{Orbit\ 2}$')
        
        bmin = self.b[self.dcoll>=self.dmin].min()
        bmax = self.b[self.dcoll>=self.dmin].max()
        
        ax2.set_xlabel('$b\ \mathrm{[AU]}$')
        ax2.set_ylabel(r'$\tilde{e}$')
        
        #Adds grey region representing impact between the planets
        ax2.axvspan(bmin,bmax,alpha=0.5,color='tab:grey',label='$d_{min}\leq r_1+r_2$')
        
        ax2.legend(prop={'size': 13})
        
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        
        self.add_ptable(ax2)
        self.add_date(fig2)
        
    def add_arrow(self,line, position=None, direction='right', size=14, color=None):
        #Thanks SO
        """Add an arrow to a line.
    
        line:       Line2D object
        position:   x-position of the arrow. If None, mean of xdata is taken
        direction:  'left' or 'right'
        size:       size of the arrow in fontsize points
        color:      if None, line color is taken."""
        
        if color is None:
            color = line.get_color()
        
        xvals = line.get_xdata()
        yvals = line.get_ydata()
    
        x0,x1 = xvals[0],xvals[-1]
        y0,y1 = yvals[0],yvals[-1]
        
        xdata = np.linspace(x0,x1,100)
        ydata = np.linspace(y0,y1,100)
    
        if position is None:
            position = xdata.mean()
        # find closest index
        if position == x1:
            start_ind = -2
        else:
            start_ind = np.argmin(np.absolute(xdata - position))
        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1
    
        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color),
            size=size
        )
        
    def add_ptable(self,ax,loc='top'):
        """Adds a table """
        a, e, _, r = self.pdata.T
        celldata  = [[a[0],e[0],'{:.0e}'.format(self.q[0]),r[0]/self.rjtoau],\
                     [a[1],e[1],'{:.0e}'.format(self.q[1]),r[1]/self.rjtoau]]
        tabcol    = ['$a\ \mathrm{[AU]}$','$e$','$q$','$R\ [R_J]}$']
        tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']
        
        table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
                  loc=loc,cellLoc='center')
        
        table.set_fontsize(10)
        
        table.scale(1, 1.2)
        
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        
    def add_date(self,fig):
        
        date = datetime.datetime.now()
        
        datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
        
        fig.text(0.88,0.95,datestr,bbox=dict(facecolor='None'),fontsize=14)
        
#We set up the data for the run

#The data for the two planets

plt.close('all')

rjtoau = 1/2150

a1, a2 = 1, 1.1
e1, e2 = 0, 0.8
m1, m2 = 1e-3, 1e-3
r1, r2 = 1*rjtoau, 1*rjtoau

p1data = np.array([a1,e1,m1,r1])
p2data = np.array([a2,e2,m2,r2])

#We set up the class object with orbits corresponding to the given parameters
#We can also choose to misalign the semi-major axes by an angle theta given
#in radians

theta = 0
Mstar = 1

SC = Scatter(p1data,p2data,Mstar,theta=theta)

#Now, we can call the functions

#The orbits can be plotted using the following function 

SC.plot_orbit()

#We then perform a single scattering with an impact parameter b

b = 0.1
#b = -10*rjtoau
#b = 0.001
SC.scatter(b = b)
#The corresponding vector triangle is given by

#SC.plot_vels(1)
SC.plot_vectri(1)
SC.plot_vectri(2)
#print(np.rad2deg(SC.defang))

#We can also plot the resulting orbital elements after a scatterings with a set
#of bvals given in an interval
bmax = 0.075*SC.rc
bvals = np.linspace(-bmax,bmax,1000)
SC.plot_new_orb(bvals)
SC.plot_defang_dmin()