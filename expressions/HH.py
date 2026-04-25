from qgis.core import *
from qgis.gui import *
import numpy as np
from scipy.optimize import root_scalar
@qgsfunction(group='Custom', referenced_columns=[])
def culCap(SHP, Q, D, B, S, N, L, NB):
    '''
    Computes the Estimated Culvert Capacity using HDS-5 Methodology
    <li>SHP, 'box' or 'circle'
    <li>Q, flow rate, cfs</li>
    <li>D, height of barrel, ft</li>
    <li>B= width of barrel, ft</li>
    <li>S= slope of barrel, ft</li>
    <li>N= mannning roughness</li>
    <li>L- length of barrel, ft</li>
    <li>NB= no of barrels</li>


    Simplistic Culvert capacity Analysis using HDS-5 methodology, specifically the equations from appendix A.
    Author: Aaron Sprague, Water Resources Solutions LLC
    Date: September 2024
    Contact: asprague@wrs-rc.com    
    variable list-
    HWi= Headwater depth above inlet control section invert, ft =
    D= Interior height of culvert barrel, ft =
    dc= crital depth, ft =  (q^2g)^(1/3)
    q=unit discharge box culvert full flow, cfs ft = D
    Vc= Velocity at critical depth, ft/s 
    Hc= Specific head at critical depth,ft = dc + Vc^2/2g
    Qc= Ap(gYh)^0.5
    Ap= Area of flow prism, ft^2
    g= 32.17
    Q= Discharge, ft^3/s =
    A= Full cross sectional area of culvert barrel, ft^2 =
    S= Culvert barrel slope, ft/ft =
    K,M,c,Y = Constants from Tables A.1 A.2 A.3 = 0.061, 0.75, 0.0423, 0.82 
    Ku= Unit conversion = 1
    Ks= Slope correction = -0.5
    N = mannings n
    b = Width of culvert, ft
    culvert_flow = (1.49 / mannings_n) * D * (culvert_height / (2*D ** (2/3)) * (slope_ft_per_ft ** 0.5)
    '''
    Q=Q/NB
    if S>0.02:
        S=0.02
    #acceleration due to gravity ft/s^2
    g= 32.17
    #Unit and Slope Correction HDS-5 A.2.1 Pg 191
    Ku, Ks=1,-0.5 
    #Constants from HDS-5 Third Edition Table A-1 
    if SHP =='BOX':
        '''Inlet Control'''
        K,M,c,Y= 0.061,0.75,0.0423,0.82
        A= D*B
        #S= 0.02 #slope of barrel, ft/ft    
        q=Q/B                                               #unit flow rate aka 'little q'
        dc=(q**2/g)**(1/3)                                  #critical depth
        Vc= (g*dc)**(1/2)                                   #critical velocity using Froude
        Hc= dc+((Vc**2)/(2*g))                              #Specific head at critical depth
        Q_AD=Q/(A*D**0.5)                                   #ratio for applicability of equations
        Qmax=3.0*A*(D**0.5)
        EQA1=((Hc/D)+K*((Ku*Q)/((A*D)**0.5))**M)+Ks*S #HDS-5 EQ A.1
        AA, BB, CC, DD, EE, FF=0.144138, 0.461363, -0.092151, 0.020003, -0.0013645, 0.000035843 
        
        F=1.8113*(Q/(B*(D**1.5)))
        if Q_AD < 0.5:
            Hwi_D=(inlet_control_headwater(Q,"rectangular",B))/D
        elif Q_AD >= 0.5 and Q_AD<=3.0:
            Hwi_D= abs(AA+(BB*(Q/(B*D**1.5)))+(CC*(Q/(B*D**1.5))**2)+(DD*(Q/(B*D**1.5))**3)+(EE*(Q/(B*D**1.5))**4)+(FF*(Q/(B*D**1.5))**5)-0.5*S)
        else:
            Hwi_D=((0.6*A*g*(2*g*d)**0.5)+0.5*D)/D
            
        HH=Hwi_D*D
            
        '''        
        #F=1.8113*(Q/(B*D**1.5))
        #EQ_Poly= ((AA+BB*F+CC*F**2+DD*F**3+EE*F**4+FF*F**5)*D-0.5*D*S)/D
        EQ_Poly=abs(AA+(BB*(Q/(B*D**1.5)))+(CC*(Q/(B*D**1.5))**2)+(DD*(Q/(B*D**1.5))**3)+(EE*(Q/(B*D**1.5))**4)+(FF*(Q/(B*D**1.5))**5)-0.5*S)
        EQA3= c*(((Ku*Q)/(A*D**0.5))**2)+(Y+Ks*S)          #HDS-5 EQ A.3
        if Q_AD <= 0.5:
            HWi_D = EQA1
        elif Q_AD<=3.0:
            HWi_D = EQ_Poly
        elif Q_AD<=3.5:
            HWi_D= (EQ_Poly+EQA3)/2
        else:
            HWi_D= EQA3
        HW=HWi_D*D #flow depth above upstream culvert invert    
        HH=HW
        outlet control
        #full flow
        #TW= tailwater depth above the outlet invert, ft optain from manning for downstream channel
        #LS= drop through culvert
        ke=0.5      #enterance loss coefficient
        Ku=29       #unit constant USC:29 SI:19.63  
        V=Q/A       #barrel velocity
        Hv=(V**2)/(2*g) #velocity head
        He=ke*Hv    #enterance loss
        Hf= ((Ku*(N**2)*L)/(A/(2*D+2*B))**1.33)*Hv #friction loss in barrel
        #Ho=Hv
        H=2*Hv+He+Hf #total barrel losses
        LS=L*S
        TW=0
        if ((dc+D)/2)>TW:
            HW0=((dc+D)/2)+H-LS
        else:
            HW0= TW +H-LS  
        if HW0>HW:
            HH=HW0
        else:
            HH=HW
        '''
    else: #SHP =='CIRCULAR':
        '''Inlet Control'''      
        K,M,c,Y=0.0045,2.0,0.0317,0.69
        A=3.14159*((D/2)**2)
        #Q= flow rate cfs, placeholder get from GIS
        #D= height of barrel, placeholder get from GIS
        #B= width of barrel, placeholder get from GIS
        #L= length of barrel, placeholder get from GIS
        #A= D*B      #cross sectional area of full barell
        #N= manning friction factor
        #S= 0.02 #slope of barrel, ft/ft 

        #q=Q/B                   #unit flow rate aka 'little q'
        yc_cir, _, _= critical_depth_circular(D,Q,g)      #critical depth
        dc=yc_cir
        Vc= (g*dc)**(1/2)       #critical velocity using Froude
        Hc= dc+((Vc**2)/(2*g))  #Specific head at critical depth
        Q_AD=Q/(A*D**0.5)       #ratio for applicability of equations
        Qmax=3.0*A*(D**0.5)
        EQA1=((Hc/D)+K*((Ku*Q)/((A*D)**0.5))**M)+Ks*S #HDS-5 EQ A.1
        AA, BB, CC, DD, EE, FF=0.167287, -0.558766, -0.159813, 0.0420069, -0.0036925, 0.000125169   
        #F=1.8113*(Q/(B*D**1.5))
        #EQ_Poly= ((AA+BB*F+CC*F**2+DD*F**3+EE*F**4+FF*F**5)*D-0.5*D*S)/D
        EQ_Poly= abs(AA+(BB*(Q/(B*D**1.5)))+(CC*(Q/(B*D**1.5))**2)+(DD*(Q/(B*D**1.5))**3)+(EE*(Q/(B*D**1.5))**4)+(FF*(Q/(B*D**1.5))**5)-0.5*S)
        EQA3= c*(((Ku*Q)/(A*D**0.5))**2)+(Y+Ks*S)          #HDS-5 EQ A.3
        '''
        if Q_AD < 0.5:
            Hwi_D=(inlet_control_headwater(Q,"circular",B,D))/D
        elif Q_AD >= 0.5 and Q_AD<=3.0:
            Hwi_D= abs(AA+(BB*(Q/(B*D**1.5)))+(CC*(Q/(B*D**1.5))**2)+(DD*(Q/(B*D**1.5))**3)+(EE*(Q/(B*D**1.5))**4)+(FF*(Q/(B*D**1.5))**5)-0.5*S)
        else:
            Hwi_D=EQ3
        '''
        if Q_AD<=3.0:
            Hwi_D= abs(AA+(BB*(Q/(B*D**1.5)))+(CC*(Q/(B*D**1.5))**2)+(DD*(Q/(B*D**1.5))**3)+(EE*(Q/(B*D**1.5))**4)+(FF*(Q/(B*D**1.5))**5)-0.5*S)
        else:
            Hwi_D=D-D*((Qf-Qd)/(0.1*Qf))
        HH=Hwi_D*D
        '''
        #full flow
        #TW= tailwater depth above the outlet invert, ft optain from manning for downstream channel
        #LS= drop through culvert
        ke=0.2      #enterance loss coefficient
        Ku=29       #unit constant USC:29 SI:19.63  
        V=Q/A       #barrel velocity
        Hv=(V**2)/(2*g) #velocity head
        He=ke*Hv    #enterance loss
        Hf= ((Ku*(N**2)*L)/(A/(2*D+2*B))**1.33)*Hv #friction loss in barrel
        #Ho=Hv
        H=2*Hv+He+Hf #total barrel losses
        LS=L*ST
        TW=0
        if ((dc+D)/2)>TW:
            HW0=((dc+D)/2)+H-LS
        else:
            HW0= TW +H-LS
        if HW0>HW:
            HH=HW0
        else:
            HH=HW
        #HH=HW    
        '''
    return HH
def _circ_geom(D, y):
    """
    Circular geometry relations for a partially full pipe.

    Parameters
    ----------
    D : float
        Pipe diameter (units: length).
    y : float
        Flow depth measured from the pipe invert (0 <= y <= D).

    Returns
    -------
    A : float
        Cross-sectional flow area at depth y.
    T : float
        Top width at depth y (chord length at the water surface).
    theta : float
        Half central angle (radians).
    """
    r = 0.5 * D
    # Clamp y to avoid numerical issues at exactly 0 or D
    y_clamped = np.clip(y, 0.0, D)
    # theta from cos(theta) = 1 - y/r; ensure argument in [-1, 1]
    arg = 1.0 - y_clamped / r
    arg = np.clip(arg, -1.0, 1.0)
    theta = np.arccos(arg)

    # Area of circular segment with central angle 2*theta
    # A = r^2 * (theta - 0.5 * sin(2*theta))
    A = (r ** 2) * (theta - 0.5 * np.sin(2.0 * theta))

    # Top width (chord) T = 2 * r * sin(theta)
    T = 2.0 * r * np.sin(theta)

    return A, T, theta


def _froude_minus_one(D, Q, g, y):
    """
    Compute Fr(y) - 1 for the circular section.

    Fr = (Q/A) / sqrt(g * (A/T)) = Q / sqrt(g * A^3 / T)
    So, Fr - 1 = Q / sqrt(g * A^3 / T) - 1

    To improve numerical stability near edges, we directly evaluate:
        F(y) = Q^2 * T / (g * A^3) - 1
    which has the same root as Fr - 1.

    Returns
    -------
    float
        Value of F(y) = Q^2*T/(g*A^3) - 1
    """
    A, T, _ = _circ_geom(D, y)

    # Guard against zero area or zero top width
    if A <= 0.0 or T <= 0.0:
        # As y -> 0, A,T->0 and Fr -> +inf (supercritical), so F -> +inf
        # As y -> D, T->0 and Fr -> 0 (subcritical), so F -> -1
        # We approximate those limits here:
        return np.inf if y <= 0.0 else -1.0

    return (Q * Q) * T / (g * (A ** 3)) - 1.0


def critical_depth_circular(
    D,
    Q,
    g=32.17,
    tol=1e-10,
    max_iter=200,
    fallback_half_diam=True
):
    """
    Compute the critical depth in a circular pipe by enforcing Fr(y) = 1
    using a robust bracketed bisection method.

    If the solver fails to converge, the function returns 0.5*D if
    fallback_half_diam=True (as requested); otherwise it returns np.nan.

    Parameters
    ----------
    D : float
        Pipe diameter (length units).
    Q : float
        Discharge (volume per time, must be consistent with length units).
    g : float, optional
        Gravitational acceleration (default: 9.80665 m/s^2).
    tol : float, optional
        Absolute tolerance on depth (and on function value for safety).
    max_iter : int, optional
        Maximum number of bisection iterations.
    fallback_half_diam : bool, optional
        If True, return D/2 when convergence fails.

    Returns
    -------
    y_crit : float
        Critical depth (same units as D).
    converged : bool
        True if the method converged; False otherwise.
    iters : int
        Number of iterations performed.
    """

    # Trivial/degenerate cases:
    if D <= 0:
        return (np.nan, False, 0)
    if Q <= 0:
        # No meaningful critical condition; trigger fallback.
        return ((0.5 * D) if fallback_half_diam else np.nan, False, 0)

    # Set a safe open-interval bracket (avoid exact endpoints where T=0 or A=0)
    eps = 1e-12 * D if D > 0 else 1e-12
    y_lo = 1e-8 * D + eps
    y_hi = D - (1e-8 * D + eps)

    # Evaluate function at bracket ends
    f_lo = _froude_minus_one(D, Q, g, y_lo)
    f_hi = _froude_minus_one(D, Q, g, y_hi)

    # We need a sign change for bisection; try to gently expand the bracket if necessary.
    # In well-behaved cases:
    #   near y=0  -> Fr >> 1 (supercritical) -> F > 0
    #   near y=D  -> Fr << 1 (subcritical)  -> F < 0
    # so f_lo > 0 and f_hi < 0 is typical.
    if not (np.isfinite(f_lo) and np.isfinite(f_hi) and (f_lo * f_hi < 0)):
        # Try a few interior samples to find a sign change
        found = False
        for frac in np.linspace(0.05, 0.95, 19):
            y_mid = y_lo + frac * (y_hi - y_lo)
            f_mid = _froude_minus_one(D, Q, g, y_mid)
            if np.isfinite(f_mid):
                if f_lo * f_mid < 0:
                    y_hi, f_hi = y_mid, f_mid
                    found = True
                    break
                if f_mid * f_hi < 0:
                    y_lo, f_lo = y_mid, f_mid
                    found = True
                    break
        if not found:
            # Could not establish a valid bracket -> fallback
            return ((1.0 * D) if fallback_half_diam else np.nan, False, 0)

    # Bisection iterations
    iters = 0
    for iters in range(1, max_iter + 1):
        y_mid = 0.5 * (y_lo + y_hi)
        f_mid = _froude_minus_one(D, Q, g, y_mid)

        if not np.isfinite(f_mid):
            # If numerical issues occur, slightly perturb the midpoint
            y_mid = np.nextafter(y_mid, y_hi)
            f_mid = _froude_minus_one(D, Q, g, y_mid)

        # Convergence checks
        if abs(f_mid) < 1e-12:  # function-level tight check
            return (y_mid, True, iters)
        if abs(y_hi - y_lo) < tol:
            return (y_mid, True, iters)

        # Maintain the bracket
        if f_lo * f_mid < 0:
            y_hi, f_hi = y_mid, f_mid
        else:
            y_lo, f_lo = y_mid, f_mid

    # If we reach here, max_iter hit -> fallback
    return ((0.5 * D) if fallback_half_diam else np.nan, False, iters)
    

def area_circular(diameter, depth):
    """Compute area of flow in a circular culvert for given depth."""
    r = diameter / 2
    if depth <= 0:
        return 0.0
    if depth >= diameter:
        return math.pi * r**2
    theta = 2 * math.acos((r - depth) / r)
    return (r**2 / 2) * (theta - math.sin(theta))

def top_width_circular(diameter, depth):
    """Compute top width for given depth in a circular culvert."""
    r = diameter / 2
    if depth <= 0:
        return 0.0
    if depth >= diameter:
        return diameter
    theta = 2 * math.acos((r - depth) / r)
    return 2 * r * math.sin(theta / 2)

def inlet_control_headwater(Q, shape, width=None, diameter=None, g=32.17, alpha=1.0, Ke=0.5, epsilon=1e-6):
    """
    Compute headwater depth for inlet control using modified minimum energy equation.
    Supports rectangular and circular culverts.
    """
    if Q <= 0:
        return 0.0
    
    if shape == "rectangular":
        if width is None:
            raise ValueError("Width required for rectangular shape.")
        y_crit = (Q**2 / (g * width**2))**(1/3)
        A = width * y_crit
    elif shape == "circular":
        if diameter is None:
            raise ValueError("Diameter required for circular shape.")
        y_crit = critical_depth_circular(Q, diameter, g)
        A = area_circular(diameter, y_crit)
    else:
        raise ValueError("Shape must be 'rectangular' or 'circular'.")
    
    v = Q / A
    velocity_head = (alpha + Ke) * (v**2) / (2 * g) * (Q / (Q + epsilon))
    HW = y_crit + velocity_head
    return HW
