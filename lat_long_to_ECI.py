import numpy as np
from celest.satellite import Time, Coordinate, Satellite
from celest.encounter import GroundPosition, windows

julian = [30462.50, 30462.50]
position = [[43, -79], [44, -80]]
c = Coordinate(position=position, frame="geo", julian=julian, offset=2430000)
location = GroundPosition(latitude=43.65, longitude=-79.38)
