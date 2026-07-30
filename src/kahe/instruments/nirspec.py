"""  NIRSPEC instrument-specific constants and functions
"""
N = 2048
LEFT_MARGIN = 10
RIGHT_MARGIN = 60
FLAT_MIN = 0.5
FLAT_MAX = 1.5
GAIN = 3.01
READ_NOISE = 28 #CDS, one co-add
MIN_ORDER_SEPARATION = 10
OVERSCAN_WIDTH = 4
HIGH_ERROR = 1e10
NIRSPEC_RES = 37500


#For empirically fitting the trace
SAVGOL_N = [51, 101, 101, 41, 21, 21, 21, 21]
SAVGOL_O = [3, 3, 3, 3, 3, 3, 2, 3]

# Upper and lower bounds for filters 
NIRSPEC1_70 = (1433), (1524)
NIRSPECHEI_1 = (1078), (1278)