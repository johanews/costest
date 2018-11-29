from OCC.StlAPI import *
from OCC.TopoDS import *
from OCC.GProp import *
from OCC.BRepGProp import *
from OCC.Bnd import *
from OCC.BRepBndLib import *

from sklearn.linear_model import LinearRegression

from flask import Flask
from flask import Blueprint
from flask_cors import CORS
from flask_restful import Resource, Api
from flask_jsonpify import jsonify

from enum import Enum

import numpy as np
import pyodbc
import os

app = Flask(__name__)
api_bp = Blueprint('api', __name__)
api = Api(api_bp)

CORS(app)


class Machine(Enum):
    SMALL = [12.5, 12.5, 600]
    LARGE = [28.0, 28.0, 800]

DB_list = []  # variable data list from DB
support_switch = bool  # Determains if the build has support or not.

class CostEstimation(Resource):

    def __init__(self):
        self.metal_density = 0.0077      # kg/cm^3

        self.powder_cost = 1000          # kr/kg
        self.labour_cost = 600           # kr/h

        self.small_machine_cost = 600    # kr/h
        self.large_machine_cost = 800    # kr/h

        self.machine = None

    def read_stl_file(self, filename):
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

    def calculate_volume(self, shape):
        """
        Calculate the volume of the given shape
        :param shape: the TopoDS_Shape object
        :return: the volume of the shape in cm^3
        """
        props = GProp_GProps()
        brepgprop_VolumeProperties(shape, props)
        return props.Mass() / 1000

    def calculate_surface_area(self, shape):
        """
        Calculate the surface area of the given shape
        :param shape: the TopoDS_Shape object
        :return: the surface area of the shape in cm^2
        """
        props = GProp_GProps()
        brepgprop_SurfaceProperties(shape, props)
        return props.Mass() / 100

    def calculate_dimensions(self, shape):
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

    def predict_time(self, attributes):
        """
        Predict the printing time of the shape by
        performing a multiple linear regression on
        previous builds
        :param data: data form previous builds
        :param attributes: the shape's features
        :return: the estimated printing time in h
        """
        features_DB = np.array(get_DB_data('TOTAL_Z_HEIGHT__MM_, VOLUME_OF_PARTS'))
        features_DB = features_DB.reshape(len(features_DB), 2)

        labels_DB = get_DB_data('TOTAL_PRINTING_TIME')[0:len(features_DB)]

        reg = LinearRegression()
        reg.fit(features_DB, labels_DB)

        return reg.predict(attributes)[0] / 60

    def material_consumption(self, volume, include_support):
        """
        The total material consumption include the
        powder used for the object and the support
        structure together
        :param data: data from previous builds
        :param volume: the shape's volume
        :return: the material consumption in cm^3
        """
        if include_support:
            volume += self.estimate_support()
        return volume

    def material_cost(self, consumption):
        """
        Calculate the cost of the powder used for
        the shape and the support structure
        :param consumption: the powder consumption
        :return: the material cost in SEK
        """
        weight = consumption * self.metal_density
        cost = weight * self.powder_cost
        return cost

    def estimate_support(self):
        """
        Estimate the volume of the support structure
        needed by taking the average of the data from
        previous builds
        :param data: data from previous builds
        :return: the estimated support volume in cm^3
        """
        support_collum = get_DB_data('VOLUME_OF_SUPPORTS')
        support_list = []
        for build_support in support_collum:
            if 0 != build_support:
                support_list.append(build_support)
        support_mean = np.array(support_list).mean()
        return support_mean / 1000

    def operator_cost(self, dims):
        """
        Calculate the labour cost of the shape
        :param dims: the shape's dimensions
        :return: the labour cost in SEK
        """
        area = dims[0] * dims[1]
        cost = self.labour_cost * (22 + (area * 0.5))
        return cost

    def printing_cost(self, hours, dims):
        """
        Estimate the actual cost of printing the
        shape based on the predicted printing time
        and the appropriate printer.
        :param hours: the estimated time
        :param dims: the shape's dimensions
        :return: the printing cost in SEK
        """
        cost = hours * self.printer_cost(dims[0], dims[1])
        return cost

    def select_printer(self, dims):
        if dims[0] < 125 and dims[1] < 125:
            self.machine = Machine.SMALL
        else:
            self.machine = Machine.LARGE

    def recycled_savings(self, dims, consumption):
        """
        Calculate the monetary value of the powder
        that can be reused in later builds.
        :param dims: the shape's dimensions
        :param consumption: the powder consumption
        :return: the value of recycled powder in SEK
        """
        total = self.printer[0] * self.printer[1] * dims[2]
        recycled = total - consumption
        return recycled * self.powder_cost * 0.5

    def get_DB_data(colName):
        DB_list.clear()

        server = 'evoserver.database.windows.net'
        database = 'SEPDB'
        username = 'eksmo'
        password = 'password-8'
        driver = '{ODBC Driver 13 for SQL Server}'
        cnxn = pyodbc.connect(
            'DRIVER=' + driver + ';SERVER=' + server + ';PORT=1433;DATABASE=' + database + ';UID=' + username + ';PWD=' + password)

        cursor = cnxn.cursor()

        # Select Query
        tsql = "SELECT " + colName + " FROM SwereaData"
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

    def get(self, shape):
        """
        Estimate the cost of a shape based on
        static data and previous builds.
        :return: the estimated cost in SEK
        """
        # self.metal_density = variables[0]   # kg/cm^3

        # self.powder_cost = variables[1]     # kr/kg
        # self.labour_cost = variables[2]     # kr/h

        # self.small_machine_cost = variables[3]  # kr/h
        # self.large_machine_cost = variables[4]  # kr/h

        cost = 0

        volume = self.calculate_volume(shape)
        dims = self.calculate_dimensions(shape)
        cons = self.material_consumption(volume, False)

        self.select_printer(dims)

        hours = self.predict_time([[dims[2], volume]])

        cost += self.material_cost(cons)
        cost += self.operator_cost(dims)
        cost += self.printing_cost(hours, dims)
        cost += self.recycled_savings(dims, cons)

        result = {'cost': round(cost, 2)}
        return jsonify(result)


api.add_resource(CostEstimation, '/costest')
app.register_blueprint(api_bp)


if __name__ == "__main__":
    app.run()
