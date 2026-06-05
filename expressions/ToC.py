from qgis.core import *
from qgis.gui import *

@qgsfunction(group='Custom', referenced_columns=[])
def ToC(l, CN, Y, units):
    """
    Calculates the sum of the two parameters value1 and value2.
    l=flow length,ft or m
    CN=Curve Number
    Y=average slope of watershed, %
    units='ft' for feet, 'm' for meters 
    """
    from swe2d.units import USC_FT_PER_SI_M
    s=(1000/CN)-10
    if units=='m':
        l=l*USC_FT_PER_SI_M
    Tc=(l**0.8*(s+1)**0.7)/(1140* Y**0.5)
    if Tc<0.083 :
        Tc=0.083
    return Tc
