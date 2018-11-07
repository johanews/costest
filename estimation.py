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


# Each side of the cube is 1 cm
# The radius and height of the cylinder is 0.75 cm and 1 cm respectively
# The height of the tower is 18.2 cm


def read_stl_file(filename):
    """
    Read the content of a STL file and return the
    shape
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


def predict_time():
    """
    Predict the printing time of the object by
    performing a multiple linear regression on
    previous builds
    :return: the estimated printing time
    """
    path = os.getcwd() + "/data.csv"
    ds = pd.read_csv(path, delimiter=';', skipinitialspace=True)

    x = ds[['Height', 'Powder', 'Recycled', 'Layers', 'Volume', 'Support']]
    y = ds['Time']

    #   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #   model = make_pipeline(PolynomialFeatures(3), LinearRegression())

    #   df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    #   print(df.head())


if __name__ == "__main__":
    predict_time()
