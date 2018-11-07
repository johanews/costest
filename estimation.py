from OCC.StlAPI import *
from OCC.TopoDS import *
from OCC.GProp import *
from OCC.BRepGProp import *
from OCC.Bnd import *
from OCC.BRepBndLib import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import pandas as pd
import os

metal_density = 0.0077      # kg/cm^3

powder_cost = 1000          # kr/kg
labour_cost = 600           # kr/h

small_machine_cost = 600    # kr/h
large_machine_cost = 800    # kr/h


def read_stl_file(filename):
    """
    Read the content of a STL file and return
    the shape
    :param filename: the STL file
    :return: the TopoDS_Shape object
    """
    assert os.path.isfile(filename)
    stl_reader = StlAPI_Reader()
    shape = TopoDS_Shape()
    stl_reader.Read(shape, filename)
    assert not shape.IsNull()
    return shape


def calculate_volume(shape):
    """
    Calculate the volume of the given shape
    :param shape: the TopoDS_Shape object
    :return: the volume of the shape in cm^3
    """
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    return props.Mass() / 1000


def calculate_surface_area(shape):
    """
    Calculate the surface area of the given shape
    :param shape: the TopoDS_Shape object
    :return: the surface area of the shape in cm^2
    """
    props = GProp_GProps()
    brepgprop_SurfaceProperties(shape, props)
    return props.Mass() / 100


def calculate_dimensions(shape):
    """
    Calculate the z-height of the given shape
    :param shape: the TopoDS_Shape object
    :return: the z-height of the shape in cm
    """
    bbox = Bnd_Box()
    bbox.SetGap(1.e-5)
    brepbndlib_Add(shape, bbox, True)

    x = abs(bbox.Get()[3] - bbox.Get()[0])
    y = abs(bbox.Get()[4] - bbox.Get()[1])
    z = abs(bbox.Get()[5] - bbox.Get()[2])

    return x/10, y/10, z/10


def predict_time(data, attributes):
    """
    Predict the printing time of the object by
    performing a multiple linear regression on
    previous builds
    :param data: the data from previous builds
    :param attributes: the build's attributes
    :return: the estimated printing time
    """
    x = data[['Height', 'Volume']]
    y = data['Time']

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    reg = LinearRegression()
    reg.fit(x, y)

    return reg.predict(attributes)


def material_consumption(data, shape):
    volume = calculate_volume(shape)
    volume += estimate_support(data)
    return volume


def material_cost(data, shape):
    volume = material_consumption(data, shape)
    weight = volume * metal_density
    cost = weight * powder_cost
    return cost


def estimate_support(data):
    support_mean = data['Support'].mean()
    return support_mean / 1000


def total_labour_cost(shape):
    dims = calculate_dimensions(shape)
    area = dims[0] * dims[1]
    cost = labour_cost * (22 + (area * 0.5))
    return cost


def define_machine(shape):
    dims = calculate_dimensions(shape)
    if dims[0] < 125 and dims[1] < 125:
        return small_machine_cost
    else:
        return large_machine_cost


def machine_cost(data, shape):
    volume = calculate_volume(shape)
    height = calculate_dimensions(shape)[2]
    time = predict_time(data, [[volume, height]]) / 60
    return time * define_machine(shape)


def recycled_material_cost(data, shape):
    dims = calculate_dimensions(shape)
    total = dims[0] * dims[1] * dims[2]
    volume = material_consumption(data, shape)
    recycled = total - volume
    return recycled * powder_cost * 0.5


def estimate_cost(data, shape):
    return material_cost(data, shape) + total_labour_cost(shape) + \
           machine_cost(data, shape) - recycled_material_cost(data, shape)


def test():
    path = os.getcwd() + "/data.csv"
    data = pd.read_csv(path, delimiter=';', skipinitialspace=True)
    shape = read_stl_file("stlfiles/cube.stl")
    return estimate_cost(data, shape)


if __name__ == "__main__":
    print(test())
