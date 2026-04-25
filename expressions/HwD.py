from qgis.core import *
from qgis.gui import *

@qgsfunction(group='Hydraulics', referenced_columns=[])
def culCapPoly(shape, Q, B, D, S):
    """
    Calculates Headwater Depth using Polynomial Best fit equation for inlet control from USGS Culvert Design System.
    <h2>Inputs:</h2>
    <ul>
      <li>shape: Box or Circular</li>
      <li>Q: flow rate, cfs</li>
      <li>B: barrel width, ft</li>
      <li>D: barrel height, ft</li>
      <li>S: barrel slope, ft/ft</li>
    </ul>
    """
    if S>0.02:
        S=0.02
    if shape =='Box':
        AA, BB, CC, DD, EE, FF=0.144138, 0.461363, -0.092151, 0.020003, -0.0013645, 0.000035843 
        HH= AA+(BB*(Q/(B*D**1.5)))+(CC*(Q/(B*D**1.5))**2)+(DD*(Q/(B*D**1.5))**3)+(EE*(Q/(B*D**1.5))**4)+(FF*(Q/(B*D**1.5))**5)-0.5*S 
    else:
        AA, BB, CC, DD, EE, FF=0.167287, -0.558766, -0.159813, 0.0420069, -0.0036925, 0.000125169
        HH= AA+(BB*(Q/(B*D**1.5)))+(CC*(Q/(B*D**1.5))**2)+(DD*(Q/(B*D**1.5))**3)+(EE*(Q/(B*D**1.5))**4)+(FF*(Q/(B*D**1.5))**5)-0.5*S     
    return abs(HH*D)
