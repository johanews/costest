from OCC.StlAPI import *
from OCC.TopoDS import *
from OCC.GProp import *
from OCC.BRepGProp import *
from OCC.Bnd import *
from OCC.BRepBndLib import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import os
import pyodbc

metal_density = 0.008      # kg/cm^3

powder_cost = 1000          # kr/kg
hourly_labour_cost = 600           # kr/h

small_machine_cost = 600    # kr/h
large_machine_cost = 800    # kr/h

DB_list = [] # variable data list from DB
support_switch = bool #Determains if the build has support or not.

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

    features_DB = np.array(get_DB_data('TOTAL_Z_HEIGHT__MM_, VOLUME_OF_PARTS'))
    features_DB = features_DB.reshape(len(features_DB), 2)

    labels_DB = get_DB_data('TOTAL_PRINTING_TIME')[0:len(features_DB)]

    x_train, x_test, y_train, y_test = train_test_split(features_DB, labels_DB, test_size=0.2, random_state=0)

    reg_test = LinearRegression()
    reg_test.fit(x_train, y_train)

    reg = LinearRegression()
    reg.fit(features_DB, labels_DB)

    #print("Estimated print time: ", "%.0f" % reg.predict(attributes)[0], "minutes")
    return reg.predict(attributes)[0]


def material_consumption(volume, support_switch):
    """
    The total material consumption can include support if needed.
    :param volume: the shape's volume
    :param support_switch: switches support on/off as bool.
    :return: the material consumption in cm^3
    """
    if support_switch == True:
        volume += estimate_support()
    #print("Powder consumption: ", "%.2f" % volume, "cm^3")
    return volume


def material_cost(consumption, metal_density):
    """
    Calculate the cost of the powder used for
    the shape and the support structure
    :param consumption: the powder consumption
    :param metal_density: which material density is used
    :return: the material cost in SEK
    """
    weight = consumption * metal_density #cm^3 * kg/cm^3 = kg
    #print("Weight of build is ", "%.0f" % (weight * 1000), "g")
    cost = weight * powder_cost #kg * SEK/kg = SEK
    return cost


def estimate_support():
    """
    Estimate the volume of the support structure
    needed by taking the average of the data from
    previous builds
    :return: the estimated support volume in cm^3
    """
    support_collum = get_DB_data('VOLUME_OF_SUPPORTS')
    support_list = []
    for build_support in support_collum:
        if 0 != build_support:
            support_list.append(build_support)
    support_mean = np.array(support_list).mean()
    return support_mean / 1000


def labour_cost(dims):
    """
    Calculate the labour cost of the shape
    :param dims: the shape's dimensions
    :return: the labour cost in SEK
    """
    area = dims[0] * dims[1]
    cost = hourly_labour_cost * (22 + (area * 0.5))
    return cost


def printing_cost(minutes, dims):
    """
    Estimate the actual cost of printing the
    shape based on the predicted printing time
    and the appropriate printer.
    :param hours: the estimated time
    :param dims: the shape's dimensions
    :return: the printing cost in SEK
    """
    cost = (minutes/60) * printer_cost(dims[0], dims[1]) #
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
    cons = material_consumption(volume, False)
    minutes = predict_time([[dims[2], volume]])

    cost += material_cost(cons, metal_density)
    #print("Material cost: ", "%.0f" % cost, "SEK")

    cost += labour_cost( dims )
    #print("Labour cost: ", "%.0f" % labour_cost( dims ), "SEK" )

    cost += printing_cost(minutes, dims)
    #print("Running time cost: ", "%.0f" % printing_cost(minutes, dims), "SEK")

    cost -= recycled_savings(dims, cons)
    #print("Recycled: ", "%.0f" % recycled_savings(dims,cons), "SEK")

    return round(cost, 1)


def main():
    file_names = ["D1.stl", "D2.stl"]
    cost_summary = {}
    for name_stl in file_names:
        file = "/Users/felixlissaker/IdeaProjects/costest/stlfiles/" + name_stl
        shape = read_stl_file(file)
        #print("Cost for", name_stl, "is: ", estimate_cost(shape), "SEK")

        cost_summary[name_stl] = estimate_cost(shape)
    print(cost_summary)
    
def get_DB_data(colName):
    DB_list.clear()

    server = 'evoserver.database.windows.net'
    database = 'SEPDB'
    username = 'eksmo'
    password = 'password-8'
    driver = '{ODBC Driver 13 for SQL Server}'
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+password)

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
    main()

