from qgis.core import *
from qgis.gui import *

@qgsfunction(group='Custom', referenced_columns=[])
def SCS_Qp(Tc_hr, CN, area, area_units, P_in, dt_hr, duration_hr, ia_ratio):
    """
   Compute hydrograph and peak discharge using:
      - NRCS/SCS CN runoff (cumulative excess),
      - SCS Type II 24-hr distribution (dimensionless depth),
      - SCS 484 dimensionless unit hydrograph routing.

    #Tc_hr: float,
    #CN: float,
    #area: float,
    #area_units: str = "acres",   # "acres" or "mi2"
    #P_in: float = 3.0,           # total point precipitation depth (inches)
    #dt_hr: float = 0.1,          # computational time step (hrs), default 6-min
    #duration_hr: float = 24.0,   # storm duration (hrs)
    #ia_ratio: float = 0.20       # Ia/S ratio (use 0.05 if your standard specifies)
    """

    import numpy as np


    #Tc_hr: float,
    #CN: float,
    #area: float,
    #area_units: str = "acres",   # "acres" or "mi2"
    #P_in: float = 3.0,           # total point precipitation depth (inches)
    #dt_hr: float = 0.1,          # computational time step (hrs), default 6-min
    #duration_hr: float = 24.0,   # storm duration (hrs)
    #ia_ratio: float = 0.20       # Ia/S ratio (use 0.05 if your standard specifies)





    """
    Tc_hr=value4
    CN=value2
    area=value5
    area_units="acres"
    P_in=value3
    dt_hr=0.0166667
    duration_hr=24.0
    ia_ratio=0.20

    Compute hydrograph and peak discharge using:
      - NRCS/SCS CN runoff (cumulative excess),
      - SCS Type II 24-hr distribution (dimensionless depth),
      - SCS 484 dimensionless unit hydrograph routing.

    Returns:
      t (hrs): time vector for runoff hydrograph
      Q (cfs): simulated discharge hydrograph at basin outlet
      Qp (cfs): peak discharge
      tp_hr (hrs): time of peak
    """

    # -----------------------------
    # 0) Units & constants
    # -----------------------------
    if area_units.lower() in ["ac", "acre", "acres"]:
        A_mi2 = area / 640.0
    elif area_units.lower() in ["mi2", "sqmi", "square miles", "mi^2"]:
        A_mi2 = area
    else:
        raise ValueError("area_units must be 'acres' or 'mi2'")

    # Peaking factor (PRF) for standard SCS DUH
    PRF = 484.0  # cfs·hr / (mi^2·in) when used as Qp = PRF*A*Q/Tp

    # Basin lag (t_L) and time-to-peak (Tp) of unit hydrograph
    # SCS relation: lag ≈ 0.6*Tc, and Tp = dt/2 + lag
    lag_hr = 0.6 * Tc_hr
    Tp_hr = dt_hr / 2.0 + lag_hr

    # -----------------------------
    # 1) Type II 24-hr cumulative distribution (% of total)
    #    Source: tabular Type II cumulative mass curve at quarter-hour resolution
    #    (interpolated to dt). Replace with your local table as needed.
    # -----------------------------
    # time (hr) → cumulative % (0..100). Excerpted points; linear interpolation fills-in.
    typeII_pts = np.array([
      
    [0.000,0.00],
    [0.240,0.20],
    [0.504,0.50],
    [0.744,0.80],
    [1.008,1.10],
    [1.248,1.40],
    [1.512,1.70],
    [1.752,2.00],
    [1.992,2.30],
    [2.256,2.60],
    [2.496,2.90],
    [2.760,3.20],
    [3.000,3.50],
    [3.240,3.80],
    [3.504,4.10],
    [3.744,4.40],
    [4.008,4.80],
    [4.248,5.20],
    [4.512,5.60],
    [4.752,6.00],
    [4.992,6.40],
    [5.256,6.80],
    [5.496,7.20],
    [5.760,7.60],
    [6.000,8.00],
    [6.240,8.50],
    [6.504,9.00],
    [6.744,9.50],
    [7.008,10.00],
    [7.248,10.50],
    [7.512,11.00],
    [7.752,11.50],
    [7.992,12.00],
    [8.256,12.60],
    [8.496,13.30],
    [8.760,14.00],
    [9.000,14.70],
    [9.240,15.50],
    [9.504,16.30],
    [9.744,17.20],
    [10.00,18.10],
    [10.24,19.10],
    [10.51,20.30],
    [10.75,21.80],
    [10.99,23.60],
    [11.25,25.70],
    [11.49,28.30],
    [11.76,38.70],
    [12.00,66.30],
    [12.24,70.70],
    [12.50,73.50],
    [12.74,75.80],
    [13.00,77.60],
    [13.24,79.10],
    [13.51,80.40],
    [13.75,81.50],
    [13.99,82.50],
    [14.25,83.40],
    [14.49,84.20],
    [14.76,84.90],
    [15.00,85.60],
    [15.24,86.30],
    [15.50,86.90],
    [15.74,87.50],
    [16.00,88.10],
    [16.24,88.70],
    [16.51,89.30],
    [16.75,89.80],
    [16.99,90.30],
    [17.25,90.80],
    [17.49,91.30],
    [17.76,91.80],
    [18.00,92.20],
    [18.24,92.60],
    [18.50,93.00],
    [18.74,93.40],
    [19.00,93.80],
    [19.24,94.20],
    [19.51,94.60],
    [19.75,95.00],
    [19.99,95.30],
    [20.25,95.60],
    [20.49,95.90],
    [20.76,96.20],
    [21.00,96.50],
    [21.24,96.80],
    [21.50,97.10],
    [21.74,97.40],
    [22.00,97.70],
    [22.24,98.00],
    [22.51,98.30],
    [22.75,98.60],
    [22.99,98.90],
    [23.25,99.20],
    [23.49,99.50],
    [23.76,99.80],
    [24.00,100.00]
    ])
    # Note: Percentages above are consistent with standard Type II shape (50% at ~12 hr).
    # Use local WinTR-55/Atlas-14 tables if your jurisdiction requires exact ordinates.
    # Ref: City of Lewiston Type II table; HydroCAD rainfall tables; HEC-HMS storm docs.  # noqa
    # (citations provided in the narrative)

    # Build uniform time grid for the 24-hr storm
    n = int(round(duration_hr / dt_hr)) + 1
    t_storm = np.linspace(0.0, duration_hr, n)

    # Interpolate cumulative percent curve onto dt grid
    cum_pct = np.interp(t_storm, typeII_pts[:, 0], typeII_pts[:, 1])
    cum_pct = np.clip(cum_pct, 0.0, 100.0)

    # Convert to cumulative rainfall depth (inches)
    P_cum = (cum_pct / 100.0) * P_in

    # Incremental rainfall per time step (inches)
    P_inc = np.diff(P_cum, prepend=0.0)

    # -----------------------------
    # 2) Curve Number runoff (cumulative excess)
    # -----------------------------
    # S (inches) and Ia = ia_ratio*S. Cumulative excess via CN equation.
    S = (1000.0 / CN) - 10.0
    Ia = ia_ratio * S

    def cn_cum_excess(P):
        # Cumulative excess Pe for cumulative rainfall depth P (inches)
        if P <= Ia:
            return 0.0
        return ((P - Ia) ** 2) / (P + 0.8 * S)

    Pe_cum = np.array([cn_cum_excess(p) for p in P_cum])

    # Incremental rainfall excess per time step (inches)
    Pe_inc = np.diff(Pe_cum, prepend=0.0)
    Pe_inc[Pe_inc < 0] = 0.0  # guard against small negative due to interpolation noise

    # -----------------------------
    # 3) SCS 484 Dimensionless Unit Hydrograph
    # -----------------------------
    # Ordinates q/qp vs t/Tp (NRCS DUH standard table)
    duhtab = np.array([
        [0.0, 0.000], [0.1, 0.030], [0.2, 0.100], [0.3, 0.190],
        [0.4, 0.310], [0.5, 0.470], [0.6, 0.660], [0.7, 0.820],
        [0.8, 0.930], [0.9, 0.990], [1.0, 1.000], [1.1, 0.990],
        [1.2, 0.930], [1.3, 0.860], [1.4, 0.780], [1.5, 0.680],
        [1.6, 0.560], [1.7, 0.460], [1.8, 0.390], [1.9, 0.330],
        [2.0, 0.280], [2.2, 0.207], [2.4, 0.147], [2.6, 0.107],
        [2.8, 0.077], [3.0, 0.055], [3.2, 0.040], [3.4, 0.029],
        [3.6, 0.021], [3.8, 0.015], [4.0, 0.011], [4.5, 0.005],
        [5.0, 0.000]
    ])
    # Time base ≈ 5*Tp
    tb_hr = 5.0 * Tp_hr
    # UH time grid (extend beyond storm for routing)
    t_uh = np.arange(0.0, tb_hr + dt_hr, dt_hr)
    # Interpolate q/qp on t/Tp
    q_over_qp = np.interp(t_uh / Tp_hr, duhtab[:, 0], duhtab[:, 1], left=0.0, right=0.0)

    # Peak of unit hydrograph for 1 inch over 1 mi^2: Qp_unit = PRF / Tp
    Qp_unit = PRF / Tp_hr  # cfs per (in·mi^2)

    # Unit hydrograph ordinates (cfs per inch per square mile)
    q_unit = q_over_qp * Qp_unit  # shape function for a 1-in excess on 1 mi^2

    # -----------------------------
    # 4) Discrete convolution: excess -> outflow hydrograph
    # -----------------------------
    # Convolution of Pe_inc (in) with UH (cfs per in per mi^2), scaled by area (mi^2)
    Q = A_mi2 * np.convolve(Pe_inc, q_unit)[:len(Pe_inc) + len(q_unit) - 1]

    # Build output time vector (storm + UH tail)
    t = np.arange(0.0, dt_hr * len(Q), dt_hr)

    # Peak discharge & time
    Qp = float(np.max(Q))
    tp_hr = float(t[np.argmax(Q)])
    return Qp
