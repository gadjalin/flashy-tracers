from __future__ import annotations
from typing import Tuple, List, Dict, Union, Optional
from dataclasses import dataclass

import numpy as np
import re


@dataclass(frozen=True)
class Nucleus(object):
    A: int
    Z: int
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __neq__(self, other) -> bool:
        return not(self == other)


# Dictionary mapping element symbols and names to atomic numbers and mass number of most stable isotope
_SYMBOL_TO_ELEMENT = {
   "N0": (0,  1  ), "H": (1,  1  ), "He": (2,  4  ), "Li": (3,  7  ), "Be": (4,  9  ), "B": (5,  11 ), "C": (6,  12 ),
    "N": (7,  14 ), "O": (8,  16 ), "F": (9,  19 ), "Ne": (10, 20 ), "Na": (11, 23 ), "Mg": (12, 24 ), "Al": (13, 27 ),
   "Si": (14, 28 ), "P": (15, 31 ), "S": (16, 32 ), "Cl": (17, 35 ), "Ar": (18, 40 ), "K": (19, 39 ), "Ca": (20, 40 ),
   "Sc": (21, 45 ), "Ti": (22, 48 ), "V": (23, 51 ), "Cr": (24, 52 ), "Mn": (25, 55 ), "Fe": (26, 56 ), "Co": (27, 59 ),
   "Ni": (28, 58 ), "Cu": (29, 63 ), "Zn": (30, 64 ), "Ga": (31, 69 ), "Ge": (32, 74 ), "As": (33, 75 ), "Se": (34, 80 ),
   "Br": (35, 79 ), "Kr": (36, 84 ), "Rb": (37, 85 ), "Sr": (38, 88 ), "Y": (39, 89 ), "Zr": (40, 90 ), "Nb": (41, 93 ),
   "Mo": (42, 98 ), "Tc": (43, 97 ), "Ru": (44, 102), "Rh": (45, 103), "Pd": (46, 106), "Ag": (47, 107), "Cd": (48, 114),
   "In": (49, 115), "Sn": (50, 120), "Sb": (51, 121), "Te": (52, 130), "I": (53, 127), "Xe": (54, 132), "Cs": (55, 133),
   "Ba": (56, 138), "La": (57, 139), "Ce": (58, 140), "Pr": (59, 141), "Nd": (60, 142), "Pm": (61, 145), "Sm": (62, 152),
   "Eu": (63, 153), "Gd": (64, 158), "Tb": (65, 159), "Dy": (66, 164), "Ho": (67, 165), "Er": (68, 166), "Tm": (69, 169),
   "Yb": (70, 174), "Lu": (71, 175), "Hf": (72, 180), "Ta": (73, 181), "W": (74, 184), "Re": (75, 187), "Os": (76, 192),
   "Ir": (77, 193), "Pt": (78, 195), "Au": (79, 197), "Hg": (80, 202), "Tl": (81, 205), "Pb": (82, 208), "Bi": (83, 209),
   "Po": (84, 209), "At": (85, 210), "Rn": (86, 222), "Fr": (87, 223), "Ra": (88, 226), "Ac": (89, 227), "Th": (90, 232),
   "Pa": (91, 231), "U": (92, 238), "Np": (93, 237), "Pu": (94, 244), "Am": (95, 243), "Cm": (96, 247), "Bk": (97, 247),
   "Cf": (98, 251), "Es": (99, 252), "Fm": (100,257), "Md": (101,258), "No": (102,259), "Lr": (103,262), "Rf": (104,267),
   "Db": (105,262), "Sg": (106,269), "Bh": (107,264), "Hs": (108,269), "Mt": (109,278), "Ds": (110,281), "Rg": (111,282),
   "Cn": (112,285), "Nh": (113,286), "Fl": (114,289), "Mc": (115,289), "Lv": (116,293), "Ts": (117,294), "Og": (118,294)
}


# Reused from https://github.com/gadjalin/flashy.git
def find_isotope(iid: str) -> Nucleus:
    """
    Return a nucleus object corresponding to the given isotope id.

    Arguments
    ---
    iid : str
        The isotope id in the form e.g. Ni56.
        The string must start with the symbol or name of the element, and the full or shorten atomic weight of the isotope.
        For example, Pb07 must be parsed as Lead 207 (82, 207).

    Returns
    ---
    Nucleus
        A nucleus object

    Raises
    ---
    ValueError
        If the isotope id cannot be parsed;
    """

    # Special cases
    if iid.lower() in ['n', 'neut', 'neutron']:
        return Nucleus(A=1, Z=0, name='neut')
    if iid.lower() in ['p', 'prot', 'h']:
        return Nucleus(A=1, Z=1, name='h1')
    if iid.lower() in ['d', 'deut', 'deuterium']:
        return Nucleus(A=2, Z=1, name='h2')
    if iid.lower() in ['t', 'trit', 'tritium']:
        return Nucleus(A=3, Z=1, name='h3')

    match = re.match(r'^([^\d]*)(.*)$', iid)
    if match is None:
        raise ValueError(f'Unrecognised isotope id: {iid}')

    symbol = match[1]
    weight = match[2]
    try:
        Z = _SYMBOL_TO_ELEMENT[symbol.strip().title()][0]
    except KeyError:
        raise ValueError(f'Not a valid element name: {symbol}')

    A_parsed = int(weight)
    # Weight in nucleus identifier can sometimes be 2 digits even if A > 100
    if (A_parsed < 100):
        if (Z < 42): # Hydrogen to Niobium, A < 100
            A = A_parsed
        elif (Z >= 42 and Z <= 47): # Molybdenum to Silver, A ~ 100
            if (A_parsed > 80):
                A = A_parsed
            else:
                A = 100 + A_parsed
        elif (Z > 47 and Z < 78): # Cadmium to Iridium, 100 < A < 200
            A = 100 + A_parsed
        elif (Z >= 78 and Z <= 83): # Paladium to Bismuth, A ~ 200
            if (A_parsed > 80):
                A = 100 + A_parsed
            else:
                A = 200 + A_parsed
        elif (Z > 83): # Over Bismuth, A > 200
            A = 200 + A_parsed
    else:
        A = A_parsed

    return Nucleus(A=A, Z=Z, name=iid)

