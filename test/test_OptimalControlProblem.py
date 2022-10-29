import pytest

from . import ocp_dict

vdp = ocp_dict["van_der_pol"][0]()
print(vdp.linearizations[0])
