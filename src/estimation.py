from OCC.StlAPI import *
from OCC.TopoDS import *
from OCC.GProp import *
from OCC.BRepGProp import *
from OCC.Bnd import *
from OCC.BRepBndLib import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline

import pandas as pd
import numpy as np
import os
import pyodbc
import datetime

metal_density = 0.0077      # kg/cm^3

powder_cost = 1000          # kr/kg
labour_cost = 600           # kr/h

small_machine_cost = 600    # kr/h
large_machine_cost = 800    # kr/h

DB_list = [] # variable data list from DB

def read_stl_file(filename):
    """
    Read the content of the given STL file and
    return the shape
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


def predict_time(attributes):
    """
    Predict the printing time of the shape by
    performing a multiple linear regression on
    previous builds
    :param attributes: the shape's features
    :return: the estimated printing time in h
    """

    features_DB = get_DB_data('TOTAL_Z_HEIGHT__MM_, VOLUME_OF_PARTS')
    features_DB = np.array(features_DB)
    features_DB = features_DB.reshape(len(features_DB), 2)

    labels_DB = get_DB_data('TOTAL_PRINTING_TIME')
    labels_DB = labels_DB[0:len(features_DB)]

    x_train, x_test, y_train, y_test = train_test_split(features_DB, labels_DB, test_size=0.2, random_state=0)

    reg_test = LinearRegression()
    reg_test.fit(x_train, y_train)
    res = reg_test.predict(x_test)



    reg = LinearRegression()
    reg.fit(features_DB, labels_DB)

    print("Estimated print time: ", "%.0f" % reg.predict(attributes)[0], "minutes")
    return reg.predict(attributes)[0] / 60


def material_consumption(volume):
    """
    The total material consumption include the
    powder used for the object and the support
    structure together
    :param volume: the shape's volume
    :return: the material consumption in cm^3
    """
    volume += estimate_support()
    print("Estimated powder consumption: ", "%.0f" % volume, "cm^3")
    return volume


def material_cost(consumption):
    """
    Calculate the cost of the powder used for
    the shape and the support structure
    :param consumption: the powder consumption
    :return: the material cost in SEK
    """
    weight = consumption * metal_density
    cost = weight * powder_cost
    return cost


def estimate_support():
    """
    Estimate the volume of the support structure
    needed by taking the average of the data from
    previous builds
    :return: the estimated support volume in cm^3
    """
    support_mean_DB = np.array(get_DB_data('Volume_of_supports')).mean()
    return support_mean_DB / 1000


def operator_cost(dims):
    """
    Calculate the labour cost of the shape
    :param dims: the shape's dimensions
    :return: the labour cost in SEK
    """
    area = dims[0] * dims[1]
    cost = labour_cost * (22 + (area * 0.5))
    return cost


def printing_cost(hours, dims):
    """
    Estimate the actual cost of printing the
    shape based on the predicted printing time
    and the appropriate printer.
    :param hours: the estimated time
    :param dims: the shape's dimensions
    :return: the printing cost in SEK
    """
    cost = hours * printer_cost(dims[0], dims[1])
    return cost


def printer_cost(x, y):
    """
    Return the printer cost by determining
    the most appropriate printer to use for
    printing the shape
    :param x: the shape's x width
    :param y: the shape's y width
    :return: the printer cost in SEK
    """
    if x < 125 and y < 125:
        return small_machine_cost
    else:
        return large_machine_cost


def recycled_savings(dims, consumption):
    """
    Calculate the monetary value of the powder
    that can be reused in later builds.
    :param dims: the shape's dimensions
    :param consumption: the powder consumption
    :return: the value of recycled powder in SEK
    """
    total = dims[0] * dims[1] * dims[2]
    recycled = total - consumption
    return recycled * powder_cost * 0.5


def estimate_cost(shape):
    """
    Estimate the cost of a shape based on
    static data and previous builds.
    :param shape: the TopoDS_Shape object
    :return: the estimated cost in SEK
    """
    cost = 0

    volume = calculate_volume(shape)
    dims = calculate_dimensions(shape)
    cons = material_consumption(volume)

    hours = predict_time([[dims[2], volume]])

    cost += material_cost(cons)
    #print("Material: " + str(cost))
    cost += operator_cost(dims)
    #print("Operational: " + str(cost))
    cost += printing_cost(hours, dims)
    #print("Running time: " + str(cost))
    cost -= recycled_savings(dims, cons)
    #print("Recycled: " + str(cost))

    return round(cost, 2)


def main():
    file = "/Users/felixlissaker/IdeaProjects/costest/stlfiles/cube.stl"
    shape = read_stl_file(file)
    return estimate_cost(shape)

def get_DB_data(colName):
    DB_list.clear()

    server = 'evoserver.database.windows.net'
    database = 'SEPDB'
    username = 'eksmo'
    password = 'password-8'
    driver = '{ODBC Driver 13 for SQL Server}'
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+password)

    #cnxn = pyodbc.connect("Driver = ODBC Driver 13 for SQL Server;Server=tcp:evoserver.database.windows.net,1433;Database=SEPDB;Uid=eksmo@evoserver;Pwd={your_password_here};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;")

    cursor = cnxn.cursor()

    #Select Query
    tsql = "SELECT "+colName+" FROM SwereaData"
    with cursor.execute(tsql):
        row = cursor.fetchone()
        if ", " not in colName:
            while row:
                DB_list.append(row[0])
                row = cursor.fetchone()
        else:
            while row:
                if '-' not in row:
                    DB_list.append(row)
                row = cursor.fetchone()
    return DB_list

if __name__ == "__main__":
    print("Cost for shape: ", "%.0f" % main(), "SEK")
