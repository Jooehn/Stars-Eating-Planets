"""
Created on Tue Mar 26 13:22:07 2019

@author: John Wimarsson

Script that generates new orbital parameters for two planets that undergo scattering.

The main purpose of this code is to restrict the parameter space for which
a planet-host star collision becomes more probable. We have included two major
simplifications, as the contributions from the host star are neglected and 
that the orbits are coplanar, which means that the results
should be analysed with this in mind.
"""

import numpy as np
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import seaborn as sns
from plotfuncs import *
from smooth_1d import smooth

class Scatter:
    def __init__(self,p1data,p2data,M_star,theta=0):
        
        #We first initialise our constants
        self.M_s    = M_star
        self.G      = 4*np.pi**2
        self.theta  = np.deg2rad(theta)
        self.metoms = 1/332946
        self.mjtoms = 300*self.metoms
        self.rjtoau = 1/2150
        self.retoau = self.rjtoau/11
        self.rstoau = 1/215
        self.rjtors = self.rjtoau/self.rstoau
        
        kmtoau = 1/149597900
        kmstoauyr = (3600*24*365.25)*kmtoau 
        self.auyrtokms = 1/kmstoauyr
        
        #We set the critical radius to the physical radius of the host star
        self.R_s = self.calc_stellar_radius()
        self.rcrit  = self.R_s*self.rstoau
        
        #Next we save the planet data in a container
        self.pdata = np.vstack([p1data,p2data])
        
        #We also compute the radius for the planets
        self.calc_planet_radius()
        
        #This allows us to define the critical distance for planet-planet collision
        self.q     = self.pdata[:,2]/M_star
        
        crossing1 = self.cross_check()
        crossing2 = self.get_isec()
        if not np.all([crossing1,crossing2]):
            raise ValueError('The orbits do not cross')
        
        self.calc_v()
        self.calc_rhill()
        self.calc_dcrit()
        
    def calc_planet_radius(self):
        """Calculates the radius of a planet with a given mass using the 
        mass-radius relations from Tremaine & Dong (2012) and Zeng, Sasselov &
        Jacobsen (2016)."""
        m1,m2 = self.pdata.T[2]
        
        Rvals = []
        for m in [m1,m2]:
            #We use the TD12 mass-relation for our planets if they are above 2.62
            #Earth masses
            if m>=2.62*self.metoms:
                R = 10**(0.087+0.141*np.log10(m/self.mjtoms)-0.171*np.log10(m/self.mjtoms)**2)*self.rjtoau 
            else:
                CMF = 0.33 #Core mass fraction of the Earth
                
                R = (1.07-0.21*CMF)*(m/self.metoms)**(1/3.7)*self.retoau
            Rvals.append(R)

        self.pdata = np.insert(self.pdata,3,Rvals,axis=1)
        
    def calc_dcrit(self):
        
        self.dcrit = self.pdata[:,3].sum()
    
    def calc_stellar_radius(self,Z=0.02):
        """We use the analytical prescription from Tout et al. (1996) (T96) to 
        estimate the ZAMS radius of our star"""
        
        m = self.M_s
        #We load in the coefficients for the constants
        C = np.loadtxt('zams_coeff.txt',skiprows=1,delimiter=',',usecols=[1,2,3,4,5])[7:]
        
        #We assume Solar metallicity and compute eq. (4) in T96
        Z_s = 0.02
    
        theta = C[:,0]+C[:,1]*np.log10(Z/Z_s)+C[:,2]*np.log10(Z/Z_s)**2+\
                C[:,3]*np.log10(Z/Z_s)**3+C[:,4]*np.log10(Z/Z_s)**4
                
        #We now use equation (2) from T96 to estimate the radius
        
        Rzams = (theta[0]*m**2.5 + theta[1]*m**6.5+theta[2]*m**11+theta[3]*m**19+theta[4]*m**19.5)/\
                (theta[5]+theta[6]*m**2+theta[7]*m**8.5+m**18.5+theta[8]*m**19.5)
        
        return Rzams
        
    def calc_rhill(self):
        """Computes the Hill radius at the point where the orbits cross"""
        a,e,m,_ = self.pdata.T
        
        self.Rhill = self.rc[:,None]*(m/(3*self.M_s))**(1/3)
        
        self.Rhill_mut = (m.sum()/(3*self.M_s))**(1/3)*a.sum()*0.5
        
    def get_orbit_r(self,ang1,ang2=None,new=False):
        """Computes the r-values for each point on an orbit"""
        if ang2 is None:
            ang2 = ang1-self.theta
        
        a1, e1, m1, _ = self.pdata[0]
        a2, e2, m2, _ = self.pdata[1]
            
        #We work out the semi-latus rectum
        p1 = a1*(1-e1**2)
        p2 = a2*(1-e2**2)
        
        #This yields the following r for our orbit
        r1 = p1/(1+e1*np.cos(ang1))
        r2 = p2/(1+e2*np.cos(ang2))
        
        return r1,r2
        
    def cross_check(self):
        """Checks if the two specified orbits will ever cross"""
        
        a1, e1, m1,_ = self.pdata[0]
        a2, e2, m2,_ = self.pdata[1]
        
        if a1>a2:
            if (a1*(1-e1)) > (a2*(1+e2)):
                return False
            elif np.allclose((a1*(1-e1)),(a2*(1+e2))):
                return False
        elif a1<a2:
            if (a1*(1+e1)) < (a2*(1-e2)):
                return False
            elif np.allclose((a1*(1+e1)),(a2*(1-e2))):
                return False
            
        return True
    
    def get_isec(self):
        """Obtains the angle of intersection of the orbits given the planet data"""
        
        a1, e1, m1,_ = self.pdata[0]
        a2, e2, m2,_ = self.pdata[1]
        
        #We go over all angles in an interval to check the distances to the 
        #current position of the planets. If they are equal we have a collision
        #We use steps of 0.1 degrees
        
        ang = np.deg2rad(np.arange(.1,360+.1,.1))
        
        phi1 = ang
        phi2 = ang-self.theta
        
        r1 = a1*(1-e1**2)/(1+e1*np.cos(phi1))
        r2 = a2*(1-e2**2)/(1+e2*np.cos(phi2))
        
        #We then calculate the difference between these values
        rdiff = r1-r2
        
        #If these are equal to zero at any point, the corresponding angle
        #corresponds to a crossing. Otherwise, we look for the point where
        #the differences changes sign. The true angle value will then be
        #in between the values for which the sign change occurs.
        
        self.phic = np.zeros(2)
        cidx = np.array([])
        
        crossing = False
        if any(np.isclose(rdiff,0)):
            cidx = np.where(np.isclose(rdiff,0))[0]
            if np.size(cidx)==2:
                self.phic[0] += ang[cidx[0]]#,ang[cidx[0]]
                self.phic[1] += ang[cidx[1]]#,ang[cidx[1]]
                crossing = True
        elif not crossing:
            #We find the sign for each element and use np.roll to find the two points
            #where it changes
            rdsign = np.sign(rdiff)
            sz = rdsign == 0
            while sz.any():
                rdsign[sz] = np.roll(rdsign, 1)[sz]
                sz = rdsign == 0
            schange = ((np.roll(rdsign, 1) - rdsign) != 0).astype(int)
            #We set the first element to zero due to the circular shift of
            #numpy.roll
            schange[0] = 0
            #Finally we check where the array is equal to one and extract the
            #corresponding indices
            scidx = np.where(schange)[0]
            
            if np.size(scidx)==2:
                cidx = scidx
                crossing = True
            elif np.size(scidx)==0:
                crossing = False
            else:
                cidx = np.append(cidx,scidx)
                crossing = True
        
            #We now peform a linear extrapolation 
            if crossing:
                
                cidx = cidx.astype(int)
                k1 = (r1[cidx]-r1[cidx-1])/(ang[cidx]-ang[cidx-1])
                m1 = r1[cidx-1]-k1*ang[cidx-1]
                k2 = (r2[cidx]-r2[cidx-1])/(ang[cidx]-ang[cidx-1])
                m2 = r2[cidx-1]-k2*ang[cidx-1]
                
                #We then have everything we need to find our crossing angles
                self.phic = (m2-m1)/(k1-k2)
        
                #We also compute and save the distance between the orbits crossings and
                #the host star
                self.rc = (a1*(1-e1**2))/(1+e1*np.cos(self.phic))
                
            if np.size(self.phic)<2:
                crossing = False
                
            return crossing
        
    def calc_v(self):
        """Computes the velocity vectors of the two planets in our system"""
        
        a,e,m,_ = self.pdata.T
        
        a1, a2 = a
        m1, m2 = m
        
        #We calculate the radial and tangential velocities at the two crossing
        #points we have found
        
        vrc1 = e*np.sin(self.phic[0])*np.sqrt(((self.G*self.M_s)/(a*(1-e**2))))
        vtc1 = np.sqrt(((self.G*self.M_s)/(a*(1-e**2))))*(1+e*np.cos(self.phic[0]))
        
        vrc2 = e*np.sin(self.phic[1])*np.sqrt(((self.G*self.M_s)/(a*(1-e**2))))
        vtc2 = np.sqrt(((self.G*self.M_s)/(a*(1-e**2))))*(1+e*np.cos(self.phic[1]))
        
        #We now save these values in two new containers for the velocities
        
        self.v1 = np.zeros((2,2))
        self.v2 = np.zeros((2,2))
        
        self.v1[0]  += vrc1[0],vtc1[0]
        self.v1[1]  += vrc2[0],vtc2[0]
        self.v2[0]  += vrc1[1],vtc1[1]
        self.v2[1]  += vrc2[1],vtc2[1]
        
        #We can subsequently obtain the relative velocity as well as the 
        #centre-of-mass velocity
        
        self.vrel   = self.v1 - self.v2
        self.vcm    = (m[0]*self.v1+m[1]*self.v2)/(m.sum())
        
        #The velocities relative the centre-of-mass is 
        
        self.vb1    = self.v1 - self.vcm
        self.vb2    = self.v2 - self.vcm
        
        #We also save the initial energies and angular momenta
        
        v1normsq = self.v1[:,0]**2+self.v1[:,1]**2
        v2normsq = self.v2[:,0]**2+self.v2[:,1]**2
        
        self.L1 = m1*self.v1[:,1]*self.rc
        self.L2 = m2*self.v2[:,1]*self.rc
        
#        self.E1 = -0.5*self.G*m1*self.M_s/a1
#        self.E2 = -0.5*self.G*m2*self.M_s/a2
        self.E1 = 0.5*m1*v1normsq-self.G*m1*self.M_s/self.rc
        self.E2 = 0.5*m2*v2normsq-self.G*m2*self.M_s/self.rc
        
        self.L = self.L1+self.L2
        self.E = self.E1+self.E2
        
    def get_defang(self,b):        
        """Finds the deflection angle given one or a set of impact parameter
        values"""
        
        m1, m2 = self.pdata[:,2]
        
        vnormsq = (self.vrel[:,0]**2+self.vrel[:,1]**2)
        
        psi = np.arctan((b[:,None]*vnormsq)/(self.G*(m1+m2)))
        
        self.defang = (np.pi - 2*psi)
        
        #Furthermore, knowing b and vrel, we can obtain the distance at closest
        #approach between the planet
        
        #We first find the eccentricity of the close encounter
        
        e_sc = np.sqrt(1+(b[:,None]**2*vnormsq**2)/(self.G**2*(m1+m2)))
        
        #Then the minimum distances between the planets becomes
        
        self.dmin    = np.zeros((np.size(b),2))
        self.dmin[:] = (b[:,None]**2*vnormsq)/(self.G*(m1+m2)*(1+e_sc))
        
    def scatter(self,N=1,b=None):
        """Performs the scattering between the two planets in our system and
        computes the new velocities."""
        
        m1, m2 = self.pdata[:,2]
        
        if b is None:
            bmax = self.find_bmax()
            b = np.random.uniform(-bmax,bmax,N)
        
        self.get_defang(b)
        
        self.b = b
        
        #We first calculate the new velocities w.r.t. the centre of mass
        
        mrot11 = np.array([[np.cos(self.defang[:,0]),-np.sin(self.defang[:,0])],\
                         [np.sin(self.defang[:,0]),np.cos(self.defang[:,0])]]).T
        
        mrot12 = np.array([[np.cos(self.defang[:,1]),-np.sin(self.defang[:,1])],\
                         [np.sin(self.defang[:,1]),np.cos(self.defang[:,1])]]).T
        
        mrot21 = np.array([[np.cos(self.defang[:,0]),-np.sin(self.defang[:,0])],\
                         [np.sin(self.defang[:,0]),np.cos(self.defang[:,0])]]).T    
    
        mrot22 = np.array([[np.cos(self.defang[:,1]),-np.sin(self.defang[:,1])],\
                         [np.sin(self.defang[:,1]),np.cos(self.defang[:,1])]]).T
        
        self.vb1n = np.zeros((np.size(b),2,2))
        self.vb2n = np.zeros((np.size(b),2,2))
          
        self.vb1n[:,0] = np.matmul(mrot11,self.vb1[0])
        self.vb1n[:,1] = np.matmul(mrot12,self.vb1[1])
        
        self.vb2n[:,0] = np.matmul(mrot21,self.vb2[0])
        self.vb2n[:,1] = np.matmul(mrot22,self.vb2[1])
        
        #We can then easily obtain the new velocities for each planet
        
        self.v1n = self.vb1n + self.vcm
        self.v2n = self.vb2n + self.vcm
        
        #Now, we have all the information we need to compute the new orbital
        #parameters
        
        L1 = m1*self.v1n[:,:,1]*self.rc
        L2 = m2*self.v2n[:,:,1]*self.rc
        
        v1norm = np.zeros([np.size(b),2])
        v2norm = np.zeros([np.size(b),2])
        v1norm[:,0] = np.linalg.norm(self.v1n[:,0],axis=1)
        v1norm[:,1] = np.linalg.norm(self.v1n[:,1],axis=1)
        v2norm[:,0] = np.linalg.norm(self.v2n[:,0],axis=1)
        v2norm[:,1] = np.linalg.norm(self.v2n[:,1],axis=1)
        
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
        self.dE = self.E-self.En
        
        if not np.allclose(self.dL,0) & np.allclose(self.dE,0):
            
            raise ValueError('dE or dL is not conserved')
#            print('dE or dL is not conserved')
            
        #We also check if the new orbital parameters will bring the stars close
        #enough to the star for a collision.
        
        self.rmin = np.zeros((np.size(b),2,2))
        self.rmin[:,0] = self.at1*(1-self.et1)
        self.rmin[:,1] = self.at2*(1-self.et2)
        
        self.scoll      = np.zeros((np.size(b),2,2),dtype=bool)
        self.scoll[:,0,0] = (self.rmin[:,0,0] <= self.rcrit) & (self.et1[:,0] < 1) &\
                            (self.dmin[:,0] > self.dcrit)
        self.scoll[:,0,1] = (self.rmin[:,0,1] <= self.rcrit) & (self.et1[:,1] < 1) &\
                            (self.dmin[:,1] > self.dcrit)
        self.scoll[:,1,0] = (self.rmin[:,1,0] <= self.rcrit) & (self.et2[:,0] < 1) &\
                            (self.dmin[:,0] > self.dcrit)
        self.scoll[:,1,1] = (self.rmin[:,1,1] <= self.rcrit) & (self.et2[:,1] < 1) &\
                            (self.dmin[:,1] > self.dcrit)
        
        #We also check which objects have been ejected
        self.eject      = np.zeros((np.size(b),2,2),dtype=bool)
        self.eject[:,0,0] = (self.dmin[:,0] > self.dcrit) & (self.et1[:,0] >= 1)
        self.eject[:,0,1] = (self.dmin[:,1] > self.dcrit) & (self.et1[:,1] >= 1)
        self.eject[:,1,0] = (self.dmin[:,0] > self.dcrit) & (self.et2[:,0] >= 1)
        self.eject[:,1,1] = (self.dmin[:,1] > self.dcrit) & (self.et2[:,1] >= 1)
        
        #And finally the b-values which lead to mergers
        self.merger       = np.zeros((np.size(b),2),dtype=bool)
        self.merger[:,0]  = self.dmin[:,0] <= self.dcrit
        self.merger[:,1]  = self.dmin[:,1] <= self.dcrit
        
        #We can also compute the critical eccentricity for a given semi-major axis
        #that will yield a star-planet collision
        self.ecrit      = np.zeros((np.size(b),2,2))
        self.ecrit[:,0] = 1 - self.rcrit/self.at1
        self.ecrit[:,1] = 1 - self.rcrit/self.at2
        
    def find_bmax(self,fac = 0.1,step=0.01):
        """Finds an optimal bmax for the system at hand by requiring that the 
        the outer edges of the range of b-values should be where the change in
        eccentricity due to a scattering at the corresponding b-values should 
        be smaller than some tolerance."""
        
        a,e,m,R = self.pdata.T
        
        #We choose the tolerance
        tol = 1e-4
        
        #Our initial guess is 
        bmax_guess = fac*self.Rhill_mut

        bvals = np.linspace(-bmax_guess,bmax_guess,1e3)
        
        self.scatter(b = bvals)
        
        #We compute the gradients at the edges of the new eccentricity values
        det111 = abs(np.gradient(self.et1[:,0])[0])>tol
        det112 = abs(np.gradient(self.et1[:,0])[-1])>tol
        det121 = abs(np.gradient(self.et1[:,1])[0])>tol
        det122 = abs(np.gradient(self.et1[:,1])[-1])>tol
        det211 = abs(np.gradient(self.et2[:,0])[0])>tol
        det212 = abs(np.gradient(self.et2[:,0])[-1])>tol
        det221 = abs(np.gradient(self.et2[:,1])[0])>tol
        det222 = abs(np.gradient(self.et1[:,1])[-1])>tol
        
        #We check if the values at the edges of our b-value range has changed
        #less than some tolerance
        
        good_range = np.any([det111,det112,det121,det122,det211,det212,det221,det222])
                    
        #If this is true we are happy with our guess, otherwise we keep iterating
        #and adjusting the guess until we have a good guess
        while good_range:
            fac += step
            
            bmax_guess = fac*self.Rhill_mut
            bvals = np.linspace(-bmax_guess,bmax_guess,1e3)
        
            self.scatter(b = bvals)
        
            det111 = abs(np.gradient(self.et1[:,0])[0])>tol
            det112 = abs(np.gradient(self.et1[:,0])[-1])>tol
            det121 = abs(np.gradient(self.et1[:,1])[0])>tol
            det122 = abs(np.gradient(self.et1[:,1])[-1])>tol
            det211 = abs(np.gradient(self.et2[:,0])[0])>tol
            det212 = abs(np.gradient(self.et2[:,0])[-1])>tol
            det221 = abs(np.gradient(self.et2[:,1])[0])>tol
            det222 = abs(np.gradient(self.et1[:,1])[-1])>tol
            
            good_range = np.any([det111,det112,det121,det122,det211,det212,det221,det222])
            
            if fac >= 1:
                break
        
        self.bmax = bmax_guess
        
        return bmax_guess
        
    def collfinder(self,N,disp=True,col1='b',col2='r',pmass=False):
        """Carries out Monte Carlo simulations of subsequent scattering events
        between two planets in a pre-defined planetary system. We draw one value
        from a uniform distribution of impact parameters and then scatter the 
        two objects until the system has been resolved, i.e. 
            1. The orbits do not cross anymore
            2. One of the planets is ejected
            3. One of the planets collides with the host star
            4. The two planets undergo a merger
            5. We have had more than 1e3 scatterings
        
            N: The number of simulations we want to perform"""
        
        #We extract the initial params
        a,e,m,R = self.pdata.T
        
        #We can choose to use point masses
        if pmass:
            R = np.array([0,0])
            self.pdata[:,3] = R
            self.calc_dcrit()
        
        #We want to find a range of b which would be suitable for our needs
        #This should be given by the point where change in eccentricity
        #with respect to the initial values is less than some tolerance
            
        bmax = self.find_bmax()
        
        #We now have our initial set of impact parameter values and can begin
        #to detect interesting orbital configurations
        
        #Monte Carlo simulation where we draw a bunch of impact parameter
        #values for an initial system with specified orbital properties
        
        N_att = 1e4
        
        b_mc = np.random.uniform(-bmax,bmax,N)
        cid_mc = np.random.randint(0,2,N)
        
        self.scatter(b = b_mc)
        #We save the dmin values
        dmin_mc = self.dmin
        vinf_mc = np.sqrt(self.vrel[:,0]**2+self.vrel[:,1]**2)
        
        at1vals = self.at1
        et1vals = self.et1
        at2vals = self.at2
        et2vals = self.et2
        
        #We also save the array containing info on which impact parameters
        #that have led to planet-consumption
        
        scoll_mc = self.scoll
        merger_mc = self.merger
        eject_mc  = self.eject
        
        #We set up a figure for which we plot the progress of the MC simulation
        if self._mc_axes is None:
            fig, [ax0,ax1] = plt.subplots(1,2,figsize=(12,6),sharey=True)
            
            for i in [ax0,ax1]:
                i.set_xlabel('$b\ [R_J]$')
                i.set_ylim(-0.1,1.1)
                i.set_xlim(-5,5)
            
            ax1xticks = ax1.xaxis.get_major_ticks()
            [ax1xticks[i].label1.set_visible(False) for i in range(2)]
            ax0.set_ylabel(r'$\tilde{e}$')        
            fig.subplots_adjust(wspace = 0)
            
            #We set up a second figure to visualise the systems where we get scatterings
            fig2, ax2 = plt.subplots(figsize=(10,6))
            
            ax2.set_xlabel(r'$\tilde{a}\ [AU]$')
            ax2.set_ylabel(r'$\tilde{e}$')        
            ax2.set_xlim(1e-3,50)
            ax2.set_ylim(-0.1,1.1)
            ax2.set_xscale('log')
            
            ax2.axvline(self.rcrit,linestyle='--',color='tab:grey',alpha=0.5)
            
            fig3, ax3 = plt.subplots()
            
            ax3.set_title('$\mathrm{Number\ of\ scatterings\ to\ get\ CME}$')
            ax3.set_xlabel('$\mathrm{r_\mathrm{min}}$')
            ax3.set_ylabel('$\mathrm{N_\mathrm{CE}}$')
            ax3.set_xscale('log')
            ax3.set_yscale('symlog')
            ax3.set_xlim(1e-3,20)
            ax3.set_ylim(0,1.2e4)
            
            ax3.axvline(self.R_s*self.rstoau,linestyle='dashed',color='k')
            
            fig4, ax4 = plt.subplots()
            
            ax4.set_title('$\mathrm{Final\ eccentricities\ before\ CME}$')
            ax4.set_xlabel('$e_\mathrm{CME}$')
            ax4.set_ylabel('$e_\mathrm{surv}$')
            ax4.set_xlim(-0.01,1.01)
            ax4.set_ylim(-0.01,1.01)
            
            fig5, ax5 = plt.subplots()
            
            ax5.set_title('$\mathrm{Minimum separation before CME}$')
            ax5.set_xlabel('$d_\mathrm{min}\ [\mathrm{AU}]$')
            ax5.set_ylabel('$v_\infty\ [\mathrm{km\ s}^{-1}]$')
            ax5.set_xscale('symlog',linthreshx = 1e-4)
            ax5.set_xlim(-1e-5,1e3)
            ax5.set_ylim(0,40)
            
            self._mc_figs = [fig,fig2,fig3,fig4,fig5]
        else:
            ax0, ax1, ax2, ax3, ax4, ax5 = self._mc_axes
            fig, fig2, fig3, fig4, fig5 = self._mc_figs
        #We loop through all scatter scenarios and use the new orbital params
        #for our new system, we keep doing this until we get either a collision
        #or have more than N_att attempts to get one
        
        #We set up containers
        
        barr  = []
        
        rmin = np.zeros((N,2))
        self._cme_info = np.zeros((N,7))
        
        for i in range(N):
            
            if disp:
                print('Currently on system: ',i)
            
            #We set up necessary variables
            itr    = 0
            outcome= 6 #The original outcome is unresolved
            coll   = False
            eject1 = False
            eject2 = False
            
            #outcome is a parameter that identifies the outcome of a scattering
            #with a number
            #0. Consumption of planet 1
            #1. Consumption of planet 2
            #2. Ejection of planet 1
            #3. Ejection of planet 2
            #4. Planet-planet merger
            #5. Survival by not crossing
            #6. Survival by being unresolved, i.e. itr >= N_att
            
            #We save the parameters for the current simulation in new containers
            cid = cid_mc[i]
            scolli = scoll_mc[i,:,cid]
            mergeri = merger_mc[i]
            ejecti = eject_mc[i]
            bn = b_mc[i]
            dmin = dmin_mc[i,cid]
            vinf = vinf_mc[cid]
            
            at1i = at1vals[i,cid]
            et1i = et1vals[i,cid]
            at2i = at2vals[i,cid]
            et2i = et2vals[i,cid]
            
            rmin[i,0] = at1i*(1-et1i)
            rmin[i,1] = at2i*(1-et2i)
            
            #we save the old parameters
            bvals  = [bn]
            
            mc_at1 = [at1i]
            mc_et1 = [et1i]
            mc_at2 = [at2i]
            mc_et2 = [et2i]
            
            mc_dmin = [dmin]
            mc_vinf = [vinf]
            
            #We then perform our various scatterings and detect mergers
            while np.all(scolli==False):
                
                itr += 1
                    
                self.pdata[0,0] = at1i
                self.pdata[0,1] = et1i
                self.pdata[1,0] = at2i
                self.pdata[1,1] = et2i
                
                #We check if the new orbits cross. If they don't we break the run
                
                crossing1 = self.cross_check()
                
                crossing2 = self.get_isec()
                if not np.all(np.array([crossing1,crossing2])):
                    outcome = 5
                    
                if outcome!=5:
                    #If we get a merger or an ejection we terminate as well
                    if np.any(mergeri):
                        coll = True
                        outcome = 4
                    elif np.any(ejecti[0]):
                        eject1 = True
                        outcome = 2
                    elif np.any(ejecti[1]):
                        eject2 = True
                        outcome = 3
                        
                if outcome in list(range(2,6)):
                    
                    self._cme_info[i] = np.array([itr,outcome,bn,at1i,et1i,at2i,et2i])
                    break
                    
                self.calc_v()
                self.calc_rhill()
                
                #We then carry out a new scattering
                
                #We compute a new bmax for every scattering
                bmax = self.find_bmax()
                
                bn = np.random.uniform(-bmax,bmax)
                
                self.scatter(b = np.array([bn]))
                scolli = self.scoll[0,:,cid]
                ejecti = self.eject[0,:,cid]
                mergeri = self.merger[0,cid]
                
                #We save the old parameters
                mc_at1.append(at1i)
                mc_et1.append(et1i)
                mc_at2.append(at2i)
                mc_et2.append(et2i)
                
                #We also save the impact parameter
                bvals.append(bn)
                
                #...and the velocity at infinity
                vinf = np.sqrt(self.vrel[cid,0]**2+self.vrel[cid,1]**2)
                mc_vinf.append(vinf)
                
                #..and the dmin values
                mc_dmin.append(self.dmin[0,cid])
                
                #We update the orbital elements
                at1i = self.at1[0,cid]
                et1i = self.et1[0,cid]
                at2i = self.at2[0,cid]
                et2i = self.et2[0,cid]
                
                #We also update rmin if its smaller than before
                if (at1i*(1-et1i) < rmin[i,0]) & (at1i*(1-et1i) > self.R_s):
                    rmin[i,0] = at1i*(1-et1i)
                    
                if (at2i*(1-et2i) < rmin[i,1]) & (at2i*(1-et2i) > self.R_s) :
                    rmin[i,1] = at2i*(1-et2i)
                
                #If we have had too many iterations, the system is unresolved
                if itr>=N_att and not np.any(scolli):
                    self._cme_info[i] = np.array([itr+1,outcome,bn,at1i,et1i,at2i,et2i])
                    break
            
            if np.any(scolli):
                
                #We also have to log the values for which we get a consumption
                bvals.append(bn)
                
                mc_at1.append(at1i)
                mc_et1.append(et1i)
                mc_at2.append(at2i)
                mc_et2.append(et2i)
                
                if np.any(scolli[0]):
                    outcome = 0
                elif np.any(scolli[1]):
                    outcome = 1
                
                #We also save all the scattering info in our main array
                self._cme_info[i] = np.array([itr+1,outcome,bn,at1i,et1i,at2i,et2i])
            barr.append(bvals)
            
            #############################################################
            ########################## Plotting #########################
            #############################################################
            
            #We want to plot the information from the final encounter that makes
            #the planet end up on a CME orbit
            if self._cme_info[i,0] < 2:
                fceid = -1
            else:
                fceid = -2
            
            #We use different markers for each outcome of interest
            if outcome == 0:
                ax0.plot(bvals[-1]/self.rjtoau,mc_et1[-1],'*',markersize=7,markerfacecolor='tab:green',\
                          markeredgecolor='k',markeredgewidth=0.5,zorder=2)
                
                ax2.plot(mc_at1[fceid],mc_et1[fceid],'o',markersize=5,markerfacecolor=col1,\
                           markeredgecolor='k',markeredgewidth=0.1,zorder=2)
                ax2.plot(mc_at2[fceid],mc_et2[fceid],'o',markersize=5,markerfacecolor=col2,\
                           markeredgecolor='k',markeredgewidth=0.1,zorder=2)
                
                ax2.plot([mc_at1[fceid],mc_at2[fceid]],[mc_et1[fceid],mc_et2[fceid]],'-',color='tab:gray'\
                         ,alpha=0.3,zorder=1)
                    
                ax3.plot(rmin[i,0],self._cme_info[i,0],'*',markersize=7,markerfacecolor=col1,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
                    
                ax4.plot(mc_et1[fceid],mc_et2[fceid],'*',markersize=7,markerfacecolor=col1,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
                    
                ax5.plot(mc_dmin[fceid],mc_vinf[fceid]*self.auyrtokms,'*',markersize=7,markerfacecolor=col1,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
            elif outcome == 1:
                ax1.plot(bvals[-1]/self.rjtoau,mc_et2[-1],'*',markersize=7,markerfacecolor='tab:green',\
                          markeredgecolor='k',markeredgewidth=0.5,zorder=2)
                
                ax2.plot(mc_at1[fceid],mc_et1[fceid],'o',markersize=5,markerfacecolor=col1,\
                           markeredgecolor='k',markeredgewidth=0.1,zorder=2)
                ax2.plot(mc_at2[fceid],mc_et2[fceid],'o',markersize=5,markerfacecolor=col2,\
                           markeredgecolor='k',markeredgewidth=0.1,zorder=2)
                
                ax2.plot([mc_at1[fceid],mc_at2[fceid]],[mc_et1[fceid],mc_et2[fceid]],'-',color='tab:gray'\
                         ,alpha=0.3,zorder=1)
                
                ax3.plot(rmin[i,1],self._cme_info[i,0],'*',markersize=7,markerfacecolor=col2,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
                    
                ax4.plot(mc_et2[fceid],mc_et1[fceid],'*',markersize=7,markerfacecolor=col2,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
                    
                ax5.plot(mc_dmin[fceid],mc_vinf[fceid]*self.auyrtokms,'*',markersize=7,markerfacecolor=col2,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
            elif outcome == 2:
                ax0.plot(bvals[-1],mc_et1[-1],'^',markersize=7,markerfacecolor='tab:orange',\
                          markeredgecolor='k',markeredgewidth=0.5,zorder=1)
                    
                ax3.plot(rmin[i,0],self._cme_info[i,0],'^',markersize=7,markerfacecolor=col1,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
                    
                ax4.plot(mc_et1[fceid],mc_et2[fceid],'^',markersize=7,markerfacecolor=col1,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
                    
                ax5.plot(mc_dmin[fceid],mc_vinf[fceid]*self.auyrtokms,'^',markersize=7,markerfacecolor=col1,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
            elif outcome == 3:                
                ax1.plot(bvals[-1],mc_et2[-1],'^',markersize=7,markerfacecolor='tab:orange',\
                          markeredgecolor='k',markeredgewidth=0.5,zorder=1)
                
                ax3.plot(rmin[i,1],self._cme_info[i,0],'^',markersize=7,markerfacecolor=col2,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
                
                ax4.plot(mc_et2[fceid],mc_et1[fceid],'^',markersize=7,markerfacecolor=col2,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
                
                ax5.plot(mc_dmin[fceid],mc_vinf[fceid]*self.auyrtokms,'^',markersize=7,markerfacecolor=col2,\
                           markeredgecolor='none',markeredgewidth=0.1,zorder=2)
            elif outcome == 4:
                ax0.plot(bvals[-1],mc_et1[-1],'P',markersize=7,markerfacecolor='tab:gray',\
                          markeredgecolor='k',markeredgewidth=0.5,zorder=1)
                ax1.plot(bvals[-1],mc_et2[-1],'P',markersize=7,markerfacecolor='tab:gray',\
                          markeredgecolor='k',markeredgewidth=0.5,zorder=1)
                ax3.plot(rmin[i,0],self._cme_info[i,0],'P',markersize=7,markerfacecolor=col1,\
                           markeredgecolor=col2,markeredgewidth=0.7,zorder=1)
                
                ax4.plot(mc_et1[fceid],mc_et2[fceid],'P',markersize=7,markerfacecolor=col1,\
                           markeredgecolor=col2,markeredgewidth=0.7,zorder=2)
                
                ax5.plot(mc_dmin[fceid],mc_vinf[fceid]*self.auyrtokms,'P',markersize=7,markerfacecolor=col1,\
                           markeredgecolor=col2,markeredgewidth=0.7,zorder=2)
        
        self._mc_axes = [ax0,ax1,ax2,ax3,ax4,ax5]
        
    def test_bvals(self):
        """Solves the Kepler problem for two orbits, given a varying set of
        initial phases and computes the minimum distance between the two planets
        at varying positions."""
        
        a1, e1, m1,_ = self.pdata[0]
        a2, e2, m2,_ = self.pdata[1]
        
        #We set the maximum integration time to the max period
        P1 = np.sqrt(a1**3/(self.M_s+m1))
        P2 = np.sqrt(a2**3/(self.M_s+m2))
        T  = np.max([P1,P2])
        
        #We set up equally spaced points in time for the entire max period
        tvals = np.linspace(0,T,250)
        
        #We also need an empty container for the minimum separations
        
        dlist = []
        
        #We set up initial values for the mean anomaly
        M01 = np.zeros((500))
        M02 = np.linspace(0,2*np.pi,500)
        
        if e1 >= 0.8:
            E1 = np.full(500,np.pi)
        else:
            E1 = M01
        if e2 >= 0.8:
            E2 = np.full(500,np.pi)
        else:
            E2 = M02
        
        #We compute the mean motion
        n1 = 2*np.pi/P1
        n2 = 2*np.pi/P2
        
        #We also need a figure to plot the positions of the planets on
        fig, ax = plt.subplots(figsize=(10,6))

        for t in tvals:
            
            #For each iteration, we solve the Kepler equation using a simple
            #Newton method approach
            
            M1 = M01+t*n1
            M2 = M02+t*n2
            
            #We solve the Kepler equation for both orbits
            for i in range(100):
                
                E1 = E1-(E1-e1*np.sin(E1)-M1)/(1-e1*np.cos(E1))
                E2 = E2-(E2-e2*np.sin(E2)-M2)/(1-e2*np.cos(E2))
            
            #We then compute the cartesian coordinates for the position of each
            #of our two planets
            x1 = a1*(np.cos(E1)-e1)
            y1 = (a1*np.sqrt(1-e1**2))*np.sin(E1)
            x2 = a2*(np.cos(E2)-e2)
            y2 = (a2*np.sqrt(1-e2**2))*np.sin(E2)
            
            ax.scatter(x1[0],y1[0],c='b',s=3,label='$\mathrm{Orbit\ 1}$')
            ax.scatter(x2[0],y2[0],c='r',s=3,label='$\mathrm{Orbit\ 2}$')
            
            pos1 = np.array([x1,y1])
            pos2 = np.array([x2,y2])
            
            dlist.append(np.ravel(np.linalg.norm(pos1-pos2,axis=0)))
            
        dvals = np.asarray(dlist)
        
        dmin = dvals.min(axis=0)
        
        bhandle = ax.scatter([],[],c='b',s=3,label='$\mathrm{Orbit\ 1}$')
        rhandle = ax.scatter([],[],c='r',s=3,label='$\mathrm{Orbit\ 2}$')
        
        ax.set_xlabel('$x\ \mathrm{[AU]}$')
        ax.set_ylabel('$y\ \mathrm{[AU]}$')
        ax.legend(handles=[bhandle,rhandle],prop={'size':12})
        ax.set_aspect('equal')
        self.add_ptable(ax)
        add_date(fig)
        
        fig2, ax2 = plt.subplots(figsize=(10,6))
        ax2.plot(np.rad2deg(M02),dmin/self.rjtoau)
        ax2.set_xlabel(r'$\phi_2\ [\degree]$')
        ax2.set_ylabel('$b_{min}\ [R_J]$')
        ax2.set_yscale('log')
        
        add_AUax(ax2,self.rjtoau)
        
        self.add_ptable(ax2)
        add_date(fig2,xcoord=0.9075) 
        
        #Next we find the local minima in our dmin array
        minidx = (np.diff(np.sign(np.diff(dmin))) > 0).nonzero()[0] + 1# local min
        
        minmask = dmin<0.1*dmin.max()
        
        minidx = np.intersect1d(minidx,np.where(minmask))
        
        #We then find new intervals to investigate, looking at an angle range of
        #phi_min-0.1<phi2<phi+0.1 radians
        
        M02min =  M02[minidx]
        
        M0fr = np.deg2rad(3)
        
        M02f1 = np.linspace(M02min[0]-M0fr,M02min[0]+M0fr,500)
        M02f2 = np.linspace(M02min[1]-M0fr,M02min[1]+M0fr,500)
        
        M02fr = np.concatenate([M02f1,M02f2])
        M01fr = np.zeros(len(M02fr))
        
        if e1 >= 0.8:
            E1f = np.full(len(M02fr),np.pi)
        else:
            E1f = M01fr
        if e2 >= 0.8:
            E2f = np.full(len(M02fr),np.pi)
        else:
            E2f = M02fr
        
        dlistf = []
        
        tvalsf = np.linspace(0,T,500)
        
        for t in tvalsf:
            
            #For each iteration, we solve the Kepler equation using a simple
            #Newton method approach
            
            M1f = M01fr+t*n1
            M2f = M02fr+t*n2
            
            #We solve the Kepler equation for both orbits
            for i in range(100):
                
                E1f = E1f-(E1f-e1*np.sin(E1f)-M1f)/(1-e1*np.cos(E1f))
                E2f = E2f-(E2f-e2*np.sin(E2f)-M2f)/(1-e2*np.cos(E2f))
            
            #We then compute the cartesian coordinates for the position of each
            #of our two planets
            x1f = a1*(np.cos(E1f)-e1)
            y1f = (a1*np.sqrt(1-e1**2))*np.sin(E1f)
            x2f = a2*(np.cos(E2f)-e2)
            y2f = (a2*np.sqrt(1-e2**2))*np.sin(E2f)
            
            pos1f = np.array([x1f,y1f])
            pos2f = np.array([x2f,y2f])
            
            dlistf.append(np.ravel(np.linalg.norm(pos1f-pos2f,axis=0)))
            
        dvalsf = np.asarray(dlistf)
        
        dminf = dvalsf.min(axis=0)
            
        fig3, ax3 = plt.subplots(figsize=(10,6))
        ax3.scatter(np.rad2deg(M02fr),dminf/self.rjtoau,s=3)
        ax3.set_xlabel(r'$\phi_2\ [\degree]$')
        ax3.set_ylabel('$b_{min}\ [R_J]$')
        ax3.set_yscale('log')
        
        #We find the local minima once more and calculate the slopes of the
        #lines left and right of the minima using a simple polyfit
        
        halfidx = int(0.5*len(dminf))
        
        minidxf1 = (np.diff(np.sign(np.diff(dminf[:halfidx]))) > 0).nonzero()[0] + 1# local min
        minidxf2 = (np.diff(np.sign(np.diff(dminf[halfidx:]))) > 0).nonzero()[0] + 1# local min
        
        minidxf1 = minidxf1[0]
        minidxf2 = minidxf2[0]+halfidx
        
        dminf1l = dminf[:minidxf1]
        M02f1l  = M02fr[:minidxf1]
        dminf1r = dminf[minidxf1:halfidx]
        M02f1r  = M02fr[minidxf1:halfidx]
        
        dminf2l = dminf[halfidx:minidxf2]
        M02f2l  = M02fr[halfidx:minidxf2]
        dminf2r = dminf[minidxf2:]
        M02f2r  = M02fr[minidxf2:]
        
        slope = np.zeros(4)
        
        slope[0] = np.polyfit(M02f1l,dminf1l,1)[0]
        slope[1] = np.polyfit(M02f1r,dminf1r,1)[0]
        slope[2] = np.polyfit(M02f2l,dminf2l,1)[0]
        slope[3] = np.polyfit(M02f2r,dminf2r,1)[0]
        
        print(slope)
        
        return dminf
    
    def plot_orbit(self):
        """Plots the two planetary orbits and marks the point of crossing."""
        
        #We extract the relevant data
        a1, e1, m1, _ = self.pdata[0]
        a2, e2, m2, _ = self.pdata[1]
        
        ang = np.linspace(0,2*np.pi,1000)
        
        r1, r2 = self.get_orbit_r(ang)
        
        x1 = r1*np.cos(ang)
        y1 = r1*np.sin(ang)
        x2 = r2*np.cos(ang)
        y2 = r2*np.sin(ang)
        
        #Finally we compute the coordinates of the orbit crossing
        xc1 = self.rc[0]*np.cos(self.phic[0])
        yc1 = self.rc[0]*np.sin(self.phic[0])
        
        xc2 = self.rc[1]*np.cos(self.phic[1])
        yc2 = self.rc[1]*np.sin(self.phic[1])
        
        #This can now be plotted in a diagram
        fig, ax = plt.subplots(figsize=(8,6),dpi=200)
        
        ax.plot(x1,y1,'b-',label='$\mathrm{Orbit\ 1}$')
        ax.plot(x2,y2,'r-',label='$\mathrm{Orbit\ 2}$')
        ax.plot(0,0,marker='+',color='tab:gray',ms=10)
        ax.plot(xc1,yc1,'k+',markersize=7,label='$r_1 = r_2$')
        ax.text(xc1-0.2,yc1+0.1,'$\mathrm{A}$')
        ax.plot(xc2,yc2,'k+',markersize=7)
        ax.text(xc2-0.2,yc2-0.3,'$\mathrm{B}$')
        
        ax.set_aspect('equal')
        
        xmax = int(np.ceil(np.amax(np.absolute([x1,x2]))))
        ymax = int(np.ceil(np.amax(np.absolute([x1,x2]))))
        
        ax.set_xlim(-xmax,xmax)
        ax.set_ylim(-xmax,xmax)
        ax.set_yticks(ax.get_xticks())
        ax.set_xlabel('$x\ \mathrm{[AU]}$')
        ax.set_ylabel('$y\ \mathrm{[AU]}$')
        
        ax.legend(prop={'size':13})
        
        self.add_ptable(ax)
        add_date(fig)
        
    def plot_orbit_new(self):
        """Plots the two planetary orbits and marks the point of crossing."""
        
        #We extract the relevant data
        m1 = self.pdata[0,2]
        m2 = self.pdata[1,2]
        
        ang = np.linspace(0,2*np.pi,1000)
        
        r1, r2 = self.get_orbit_r(ang)
        
        x1 = r1*np.cos(ang)
        y1 = r1*np.sin(ang)
        x2 = r2*np.cos(ang)
        y2 = r2*np.sin(ang)
        
        #Finally we compute the coordinates of the orbit crossing
        xc1 = self.rc[0]*np.cos(self.phic[0])
        yc1 = self.rc[0]*np.sin(self.phic[0])
        
        xc2 = self.rc[1]*np.cos(self.phic[1])
        yc2 = self.rc[1]*np.sin(self.phic[1])
        
        #This can now be plotted in a diagram
        fig, ax = plt.subplots(figsize=(8,6),dpi=200)
        
        ax.plot(x1,y1,'b-',label='$\mathrm{Orbit\ 1}$')
        ax.plot(x2,y2,'r-',label='$\mathrm{Orbit\ 2}$')
        ax.plot(0,0,marker='+',color='tab:gray',ms=10)
        ax.plot(xc1,yc1,'k+',markersize=7,label='$r_1 = r_2$')
        ax.text(xc1-0.2,yc1+0.1,'$\mathrm{A}$')
        ax.plot(xc2,yc2,'k+',markersize=7)
        ax.text(xc2-0.2,yc2-0.3,'$\mathrm{B}$')
        
        ax.set_aspect('equal')
        
        xmax = int(np.ceil(np.amax(np.absolute([x1,x2]))))
        ymax = int(np.ceil(np.amax(np.absolute([x1,x2]))))
        
        ax.set_xlim(-xmax,xmax)
        ax.set_ylim(-xmax,xmax)
        ax.set_yticks(ax.get_xticks())
        ax.set_xlabel('$x\ \mathrm{[AU]}$')
        ax.set_ylabel('$y\ \mathrm{[AU]}$')
        
        ax.legend(prop={'size':13})
        
        self.add_ptable(ax)
        add_date(fig)
        
    def plot_vectri(self,planet=1,idx=0):
        """Plots the vector triangle for a given planet after performing a 
        scattering with impact parameter b."""
        
        fig, ax = plt.subplots(figsize=(10,8))
        
        #We extract the information we need
        
        if planet == 1:
            
            vb  = self.vb1[idx]*self.auyrtokms
            vcm = self.vcm[idx]*self.auyrtokms
            vn  = self.v1n[0,idx]*self.auyrtokms
            
            pcol = 'b'
            
        elif planet == 2:

            vb  = self.vb2[idx]*self.auyrtokms
            vcm = self.vcm[idx]*self.auyrtokms
            vn  = self.v2n[0,idx]*self.auyrtokms
        
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
        add_arrow(vbp,direction='right')
        #Plots v
        vp, = ax.plot([-vb[1],vcm[1]],[-vb[0],vcm[0]],ls='-',color='g',lw=1.5,\
                      label='$v_'+'{}'.format(planet)+'}$')
        
        add_arrow(vp)
        #Plots vcm
        vcmp, = ax.plot([0,vcm[1]],[0,vcm[0]],'-k',lw=1.5,\
                        label='$v_\mathrm{cm}$')
        add_arrow(vcmp)
        #Plots the position of vb in the circle
        ax.plot([vcm[1],vcm[1]+vb[1]],[vcm[0],vcm[0]+vb[0]],linestyle='--',color='tab:gray',lw=1.5)
        #Plots circle of possible vbn values
        ax.plot(xc,yc,'k-',lw=1.5)
        #Plots vbn
        vbnp, = ax.plot([vcm[1],vn[1]],[vcm[0],vn[0]],color='m',ls='--',lw=1.5,\
                        label=r'$\tilde{v}_'+'{b,'+'{}'.format(planet)+'}$')
        add_arrow(vbnp,direction='right')
        #Plots vn
        vnp, = ax.plot([0,vn[1]],[0,vn[0]],color='m',ls='-',lw=1.5,\
                       label=r'$\tilde{v}_'+'{}'.format(planet)+'}$')
        add_arrow(vnp,direction='right')
            
        #We also plot a vector pointing towards the position of the host star
        
        xhs, yhs = 10*np.cos(self.phic[idx]+np.pi),10*np.sin(self.phic[idx]+np.pi)
        
        hsp, = ax.plot([0,xhs],[0,yhs],'-',color='tab:gray',\
                       label=r'$\hat{r}_{\star}$')
        add_arrow(hsp,position=xhs)
        
        #Plots a few markers in the figure
        ax.plot(0,0,marker='+',color='tab:gray',ms=12,mew=1.3,label='$\mathrm{Centre\ of\ mass}$')
        ax.plot(-vb[1],-vb[0],marker='o',color=pcol,ms=8,label='$m_{}$'.format(planet))
        ax.plot(vcm[1],vcm[0],'ko',ms=2)
        
        #Finally, we adjust the axes, add labels and a title
        ax.set_xlabel('$v_t\ \mathrm{[km\ s}^{-1}]$')
        ax.set_ylabel('$v_r\ \mathrm{[km\ s}^{-1}]$')
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
#        ax.text(xmax+.75,ymax-0.3,'$b =' + '{0:.0f}'.format(self.b/self.rjtoau)+\
#                '\ \mathrm{R}_J$')#,bbox=dict(facecolor='None', alpha=0.5))
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':12})
        ax.set_aspect('equal')  
        
        self.add_ptable(ax)
        add_date(fig)
        
    def plot_vels(self,idx):
        """Plots the velocity vectors and their new components for both planets"""
        fig, ax = plt.subplots(figsize=(8,6))
            
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
        
        add_arrow(vbnew1)
        add_arrow(vbold1)
        
        add_arrow(vnew1)
        add_arrow(vold1)
        
        add_arrow(vbnew2)
        add_arrow(vbold2)
        
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
        
    def plot_defang_dmin(self,idx=0):
        """Plots the deflection angle as a function of the impact parameter and
        the minimum distance as a function of the deflection angle"""
        fig, ax = plt.subplots(1,2,sharey=False,figsize=(14,6))
        
        #We use a different colour for the impact parameters for which we get
        #a collision between the planets
        
        col = np.asarray(['k']*np.size(self.b))
        
        anysc1 = np.any(self.scoll[:,:,0],axis=1)
        anysc2 = np.any(self.scoll[:,:,1],axis=1)
        
        anysc  = np.any([anysc1,anysc2],axis=0)
        
        col[anysc] = 'g'
        col[self.dcrit>=self.dmin[:,idx]] = 'r'
        
        ax[0].scatter(self.b/self.rjtoau,np.rad2deg(self.defang[:,idx]),c=col,s=2)
        
        ax[0].set_xlabel('$b\ [R_J]$')
        ax[0].set_ylabel(r'$\delta\ [\degree]$')
      
        rmark = ax[0].scatter([],[],c='r',s=3,marker='o',label='$d_{min}\leq d_{crit}$')
        kmark = ax[0].scatter([],[],c='k',s=3,marker='o',label='$d_{min}>d_{crit}$')
        gmark = ax[0].scatter([],[],c='g',s=3,marker='o',label='$\mathrm{Star}-\mathrm{planet\ encounter}$')
        
        ax[0].legend(handles=[rmark,gmark,kmark],prop={'size':12})
        
        #We also plot the minimum distance as a function of the deflection angle
        
        ax[1].scatter(np.rad2deg(self.defang[:,idx]),self.dmin[:,idx]/self.rjtoau,c=col,s=2)
        
        ax[1].set_ylabel('$d_{min}\ [R_J]$')
        ax[1].set_xlabel(r'$\delta\ [\degree]$')
        
        ax[1].set_xlim(0,360)
#        ax[1].set_ylim(0.5*self.dcrit/self.rjtoau,(self.dmin/self.rjtoau).max())
        ax[1].set_yscale('log')
        
        self.add_ptable(ax[0])
        add_date(fig)
        
#        fig.subplots_adjust(left=0.2,bottom=0.2)
        
    def plot_new_orb(self,bvals=None,idx=0):
        """Plots the new orbital elements after scattering"""
        
        #We save the combined radius of the planets to set up a check for 
        #physical collisions between the planets
    
        self.scatter(b = bvals)
        
        #We mark the scattering events that lead to collision with red, while
        #the rest are black
        
        col1 = np.asarray(['k']*np.size(bvals))
        col2 = np.asarray(['k']*np.size(bvals))
        
        col1[self.scoll[:,0,idx]] = 'g'
        col1[self.merger[:,0]] = 'r'
        col2[self.scoll[:,1,idx]] = 'g'
        col2[self.merger[:,1]] = 'r'
        #We then set up the plotting parameters
        
        xmax1 = np.amax(self.at1[:,idx])+0.1*np.amax(self.at1[:,idx])
        ymax1 = np.amax(self.et1[:,idx][self.E2n[:,idx]<0])\
                +0.3*np.amax(self.et1[:,idx][self.E1n[:,idx]<0])
        
        xmax2 = np.amax(self.at2[:,idx])+0.1*np.amax(self.at2[:,idx])
        ymax2 = np.amax(self.et2[:,idx][self.E2n[:,idx]<0])\
                +0.3*np.amax(self.et2[:,idx][self.E2n[:,idx]<0])
        
        dx1 = 1
        dx2 = 1
        dy1 = 0.1
        dy2 = 0.1
        
        if xmax1 > 10:
            dx1 = 5
            xmax1 = 10
        if xmax2 > 10:
            dx2 = 5
            xmax2 = 10
        if ymax1 > 1:
            dy1 = 0.5
        if ymax2 > 1:
            dy2 = 0.5
            
        #Next, we plot the relevant data
        
        fig, ax = plt.subplots(1,2,sharey=False,figsize=(12,6),dpi=200)
        
        ax[0].scatter(self.at1[:,idx],self.et1[:,idx],s=1,c=col1,marker='o')
        ax[1].scatter(self.at2[:,idx],self.et2[:,idx],s=1,c=col2,marker='o')
            
        ax[0].set_xlabel(r'$\tilde{a}_1\ \mathrm{[AU]}$')
        ax[0].set_ylabel(r'$\tilde{e}_1$')
        if any(abs(xmax1-self.at1[:,idx])>1):
            ax[0].set_xlim(0,xmax1)
            ax[0].set_xticks(np.arange(0,int(xmax1)+dx1,dx1))
        ax[0].set_ylim(0,ymax1)
        if ymax1 > 0.1:
            ax[0].set_yticks(np.arange(0,np.around(ymax1,1),dy1))
        
        ax[1].set_xlabel(r'$\tilde{a}_2\ \mathrm{[AU]}$')
        ax[1].set_ylabel(r'$\tilde{e}_2$')
        if any(abs(xmax2-self.at2[:,idx])>1):
            ax[1].set_xlim(0,xmax2)
            ax[1].set_xticks(np.arange(0,int(xmax2)+dx2,dx2))
        ax[1].set_ylim(0,ymax2)
        if ymax2 > 0.1:
            ax[1].set_yticks(np.arange(0,np.around(ymax2,1),dy2))
        
        #Handles for our legend
        omark = ax[1].scatter([],[],c='r',s=3,marker='o',label='$d_{min}\leq d_{crit}$')
        gmark = ax[1].scatter([],[],c='g',s=3,marker='o',label=r'$\tilde{e}_{i} > e_{crit}$')
        kmark = ax[1].scatter([],[],c='k',s=3,marker='o',label='$d_{min}>d_{crit}$')
        
        if np.any(self.scoll):
            hlist = [omark,gmark,kmark]
        else:
            hlist = [omark,kmark]
            
        ax[1].legend(handles=hlist,prop={'size':14})
        
        self.add_ptable(ax[0])
        add_date(fig)
        
        #We also make a plot with the eccentricities as a function of the 
        #impact parameter b
        
        fig2, ax2 = plt.subplots(figsize=(8,6),dpi=200)

        orb1, = ax2.plot(self.b/self.bmax,self.et1[:,idx],color='b',label='$\mathrm{Orbit\ 1}$')
        orb2, = ax2.plot(self.b/self.bmax,self.et2[:,idx],color='r',label='$\mathrm{Orbit\ 2}$')             
        
        #We add a line representing the critical eccentricity for collision with
        #the Sun given the new semi-major axis produced by given impact parameter
        
        #To do so, we need the range of b for which we will have an interaction
        #between the planets, which are given by bmin and bmax
        
        if np.any(self.dcrit>=self.dmin):
            bmin = self.b[self.dcrit>=self.dmin[:,idx]].min()
            bmax = self.b[self.dcrit>=self.dmin[:,idx]].max()
        else:
            bmin = 0
            bmax = 0
        
        elim1, = ax2.plot(self.b/self.bmax,self.ecrit[:,0,idx],'b--',alpha=0.5,\
                 label=r'$e_{1,crit}$')
        elim2, = ax2.plot(self.b/self.bmax,self.ecrit[:,1,idx],'r--',alpha=0.5,\
                 label=r'$e_{2,crit}$')
        
        et1mask = (self.ecrit[:,0,idx]<=self.et1[:,idx]) & (self.et1[:,idx]<1)
        et2mask = (self.ecrit[:,1,idx]<=self.et2[:,idx]) & (self.et2[:,idx]<1) 
        
        #We create the shaded region for orbit 1
        ax2.fill_between(self.b[self.b<=bmin]/self.bmax,self.ecrit[:,0,idx][self.b<=bmin]\
                         ,self.et1[:,idx][self.b<=bmin],alpha=0.6,\
                         where=et1mask[self.b<=bmin],color='g')
        ax2.fill_between(self.b[self.b>=bmax]/self.bmax,self.ecrit[:,0,idx][self.b>=bmax],\
                         self.et1[:,idx][self.b>=bmax],alpha=0.6,\
                         where=et1mask[self.b>=bmax],color='g')
        
        #We do the same for the second orbit
        ax2.fill_between(self.b[self.b<=bmin]/self.bmax,self.ecrit[:,1,idx][self.b<=bmin]\
                         ,self.et2[:,idx][self.b<=bmin],alpha=0.6,\
                         where=et2mask[self.b<=bmin],color='g')
        ax2.fill_between(self.b[self.b>=bmax]/self.bmax,self.ecrit[:,1,idx][self.b>=bmax],\
                         self.et2[:,idx][self.b>=bmax],alpha=0.6,\
                         where=et2mask[self.b>=bmax],color='g')
        
        #We now add a legend to the plot
        sczone = plt.Rectangle((0, 0), 1, 1, fc="g",alpha=0.6,label=r'$e_{i,crit}<\tilde{e}_i<1$')
#        elim2 = plt.Rectangle((0, 0), 1, 1, fc="r",alpha=0.5,label=r'$e_{2,crit}<\tilde{e}_1<1$')
        
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.95, box.height])    
        
        ax2.set_xlabel('$b\ [b_\mathrm{max}]$')
        ax2.set_ylabel(r'$\tilde{e}$')

        ax2.set_xlim(-1,1)        
#        ax2.set_xlim(-0.01/self.bmax,0.01/self.bmax)
        ax2.set_ylim(-0.1,1.1)
        
        #Adds grey region representing impact between the planets
        gzone = ax2.axvspan(bmin/self.bmax,bmax/self.bmax,alpha=0.75,color='tab:grey',zorder=-1,\
                            label='$d_{min}\leq d_{crit}$')
        
        if np.any(self.scoll):
            hlist = [orb1,orb2,elim1,elim2,gzone,sczone]
        else:
            hlist = [orb1,orb2,elim1,elim2,gzone]
        
        ax2.legend(handles=hlist,prop={'size': 13},loc='best')
        
        self.add_ptable(ax2)
        add_date(fig2)
        
        #We plot the minimum radius to the star as a function of impact parameter
        
        fig3, ax3 = plt.subplots(figsize=(8,6),dpi=200)
        
        rmin1, = ax3.plot(self.b/self.bmax,self.rmin[:,0,idx]/self.rstoau,\
                          label='$\mathrm{Orbit\ 1}$',color='b')
        rmin2, = ax3.plot(self.b/self.bmax,self.rmin[:,1,idx]/self.rstoau,\
                          label='$\mathrm{Orbit\ 2}$',color='r')
        
        #We only use this line if we have r_crit != R_star
#        rcrit = ax3.axhline(self.rcrit/self.rstoau,linestyle='--',label='$r_{crit}$',\
#                             color='tab:gray')
        
        rcrit = ax3.axhline(self.R_s,linestyle='--',label='$R_\star$',\
                             color='black',alpha=0.8)
        
        rcrits = np.asarray([self.rcrit]*len(self.b))/self.rstoau
        
        ax3.fill_between(self.b[self.b<=bmin]/self.bmax,rcrits[self.b<=bmin]\
                         ,self.rmin[:,0,idx][self.b<=bmin]/self.rstoau,alpha=0.6,\
                         where=et1mask[self.b<=bmin],color='g')
        ax3.fill_between(self.b[self.b>=bmax]/self.bmax,rcrits[self.b>=bmax]\
                         ,self.rmin[:,0,idx][self.b>=bmax]/self.rstoau,alpha=0.6,\
                         where=et1mask[self.b>=bmax],color='g')
        #We do the same for the second orbit
        ax3.fill_between(self.b[self.b<=bmin]/self.bmax,rcrits[self.b<=bmin]\
                         ,self.rmin[:,1,idx][self.b<=bmin]/self.rstoau,alpha=0.6,\
                         where=et2mask[self.b<=bmin],color='g')
        ax3.fill_between(self.b[self.b>=bmax]/self.bmax,rcrits[self.b>=bmax]\
                         ,self.rmin[:,1,idx][self.b>=bmax]/self.rstoau,alpha=0.6,\
                         where=et2mask[self.b>=bmax],color='g')
        
        ax3.axvspan(bmin/self.bmax,bmax/self.bmax,alpha=0.75,color='tab:grey',zorder=-1,\
                        label='$d_{min}\leq d_{crit}$')
        
        ax3.set_xlabel('$b\ [b_\mathrm{max}]$')
        ax3.set_ylabel(r'$r_{min}\ [R_\odot]$')
        
        ax3.set_xlim(-1,1)
#        ax3.set_ylim(0,2/self.rstoau)
        ax3.set_yscale('log')
        
        ax3.legend(prop={'size':12})
        
        add_AUax(ax3,self.rstoau,True)
        
        self.add_ptable(ax3)
        add_date(fig3,xcoord=0.9075) 
        
    def add_ptable(self,ax,loc='top'):
        """Adds a table containing relevant properties of the system we are
        investigating"""
        
        a, e, _, r = self.pdata.T
        celldata  = [[a[0],e[0],'{:.0e}'.format(self.q[0]),'{:.2f}'.format(r[0]/self.retoau)],\
                     [a[1],e[1],'{:.0e}'.format(self.q[1]),'{:.2f}'.format(r[1]/self.retoau)]]
        tabcol    = ['$a$ [AU]','$e$',r'$q_\star$',r'$R_p\ [\mathrm{R}_\oplus]$']
        tabrow    = ['Orbit 1','Orbit 2']
        
        table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
                  loc=loc,cellLoc='center')
        
        table.set_fontsize(11.8)
        
        table.scale(1, 1.45)
        
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)