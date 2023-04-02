ocp_dict = {}

from example_problems.van_der_pol import van_der_pol
ocp_dict["van_der_pol"] = {
    "ocp": van_der_pol.VanDerPol, "config": van_der_pol.config
}
