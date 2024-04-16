from examples.van_der_pol import VanDerPol
from examples.attitude_control import AttitudeControl
from examples.burgers import BurgersPDE
from examples.uav import FixedWing


ocp_dict = {'van_der_pol': VanDerPol,
            'attitude_control': AttitudeControl,
            'burgers': BurgersPDE,
            'uav': FixedWing}
