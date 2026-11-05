"""
Channels, material and constant of the simulator.
Numbers will be changed according to results of simulations
"""

from dataclasses import dataclass, field, asdict

import torch

#materials (first 8 channels)
GROUND, GRASS, STONE, CINDER, TREE, METAL, ICE, LAVA = range(8)
MATERIALS = ("ground", "grass", "stone", "cinder", "tree", "metal", "ice", "lava")
N_MATERIALS = 8
SOIL = (GROUND, GRASS, STONE, CINDER)
STRUCTURES = (TREE, METAL, ICE, LAVA)
FLAMMABLE = (GRASS, TREE)

#Other 12 channels (Water, oil, fire, electric charge, rest after electric charge, temperature
#humidity, combustibility, vapor, smoke, wind in x and y directions)
W, O = 8, 9
F, E, R = 10, 11, 12
T, H, C, V, S = 13, 14, 15, 16, 17
WX, WY = 18, 19
N_CHANNELS = 20

CHANNEL_NAMES = MATERIALS + ("W", "O", "F", "E", "R", "T", "H", "C", "V", "S", "wind_x", "wind_y")
#Channels changed with continuous increments
CONTINUOUS = (W, O, F, E, R, T, H, C, V, S)
#Channels clipped to [0, 1] (only wind stays in [-1, 1])
CLIPPED = tuple(range(18))

#Task T1...T8
RULES = ("Combustion", "Wind", "Water", "Electricity", "Ice", "Fuel", "Lava", "Vapor")
N_RULES = 8
T1, T2, T3, T4, T5, T6, T7, T8 = range(8)
RULE_IDS = {f"T{i + 1}": i for i in range(N_RULES)}

#Priority in material changes, lava > fire > ice > other stuff
MATERIAL_PRIORITY = {T7: 3, T1: 2, T5: 1}

INF = float("inf")


@dataclass(frozen=True)
class MaterialProps:
    #Starting threshold for various phenomena
    #Temperature at which the material begins to burn
    ignition: tuple = (INF, 0.6, INF, INF, 0.65, INF, INF, INF)
    #How much a material can burn before the fire goes out
    starting_combustibility: tuple = (0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    #Thermal conductivity
    k: tuple = (0.5, 0.4, 0.3, 0.3, 0.3, 1.0, 0.6, 0.6)
    #Electrical conductivity
    conductor: tuple = (0, 0, 0, 0, 0, 1, 0, 0)
    #Water absorbency
    absorbency: tuple = (0.5, 0.8, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SimParams:
    t_amb: float = 0.3
    #To avoid division by 0
    eps: float = 1e-6

    #Base
    heat_rate: float = 0.2
    heat_relax: float = 0.02
    hum_rate: float = 0.1
    hum_decay: float = 0.01
    w_rate: float = 0.05
    w_decay: float = 0.002
    vap_rate: float = 0.2
    vap_decay: float = 0.005
    smoke_rate: float = 0.2
    smoke_decay: float = 0.05
    fire_decay: float = 0.1
    charge_decay: float = 0.5
    rest_decay: float = 0.34

    #Combustion
    t1_max_h_ignite: float = 0.4
    t1_c_burn: float = 0.1
    t1_heat: float = 0.3
    t1_smoke: float = 0.2
    t1_smoke_heat: float = 0.05
    t1_h_extinguish: float = 0.7
    #Heat given to the 8 neighbours, proportional to how many of them are burning
    t1_radiate: float = 1.2

    #Wind
    #It's the portion of T, V or S taken from the direction of the wind
    t2_rate: float = 0.6

    #Water
    t3_wet: float = 0.1
    t3_cool: float = 0.1
    t3_h_puddle: float = 0.95 #humidity to have a puddle
    t3_puddle_rate: float = 0.02 #growth rate per tick
    t3_absorb_rate: float = 0.01
    t3_thin: float = 0.05 #if the puddle is thinner than this, it disappears
    t3_t_evap: float = 0.6
    t3_evap: float = 0.1
    #contact with water wets a cell only up to here, below t3_h_puddle: only rain makes new puddles
    t3_h_wet_max: float = 0.9

    #Electricity
    t4_w_conduct: float = 0.2 #How much water we need on a cell to conduct
    t4_e_on: float = 0.5 
    t4_heat: float = 0.4
    t4_radiate: float = 0.8

    #Ice
    t5_w_freeze: float = 0.5
    t5_t_freeze: float = 0.15
    t5_t_melt: float = 0.25

    #Fuel
    t6_o_ignite: float = 0.1
    t6_t_ignite: float = 0.4
    t6_max_h_ignite: float = 0.7
    t6_o_burn: float = 0.2
    t6_heat: float = 0.5
    t6_smoke: float = 0.3
    t6_radiate: float = 1.5

    #Lava
    t7_cool: float = 0.03
    t7_t_solid: float = 0.6
    #If a neighbour cell has water > t7_w_contant then in the cell turn immediately lava in stone
    t7_w_contact: float = 0.2 
    t7_vapor: float = 0.3
    t7_t_melt: float = 0.95
    t7_radiate: float = 0.5

    
    #Vapor
    #Minimum threshold for vapor to condensate
    t8_v_cond: float = 0.02
    #Share of the vapor that condenses per tick
    t8_rate: float = 0.1
    #Max temperature to condensate
    t8_t_cond: float = 0.35
    #Humidity produced by each unit of condensed vapor
    t8_h_gain: float = 3.0

    #Events
    rain_h: float = 0.15 #if it rains at every tick h increases of this
    ignite_t: float = 0.8 #if we set a fire on, the t instantly becomes this

    materials: MaterialProps = field(default_factory=MaterialProps)

    def to_dict(self):
        return asdict(self)


def material_table(values, device, dtype=torch.float32):
    """
    values is a simple tuple containing the values of a given property for all materials
    We return it as a tensor, to be used with one-hot tensors"""
    return torch.tensor(values, device=device, dtype=dtype)


def rules_mask(spec, device="cpu"):
    """Return a simple boolean array that tells which task are active"""
    out = torch.zeros(N_RULES, dtype=torch.bool, device=device) #Base Case
    if spec in ("base", None):
        return out
    if spec == "joint_full":
        return torch.ones(N_RULES, dtype=torch.bool, device=device)
    names = [spec] if isinstance(spec, str) else list(spec)
    for name in names:
        out[RULE_IDS[name] if isinstance(name, str) else int(name)] = True
    return out