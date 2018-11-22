from OCC.StlAPI import *
from OCC.TopoDS import *
from OCC.GProp import *
from OCC.BRepGProp import *
from OCC.Bnd import *
from OCC.BRepBndLib import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api
from flask_jsonpify import jsonify

import numpy as np
import os

app = Flask(__name__)
api = Api(app)

CORS(app)


class CostEstimation(Resource):

    def __init__(self):
        self.metal_density = 0.0077      # kg/cm^3

        self.powder_cost = 1000          # kr/kg
        self.labour_cost = 600           # kr/h

        self.small_machine_cost = 600    # kr/h
        self.large_machine_cost = 800    # kr/h

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

    def predict_time(self, data, attributes):
        """
        Predict the printing time of the shape by
        performing a multiple linear regression on
        previous builds
        :param data: data form previous builds
        :param attributes: the shape's features
        :return: the estimated printing time in h
        """
        features = data[['Height', 'Volume']]
        labels = data['Time']

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

        reg_test = LinearRegression()
        reg_test.fit(x_train, y_train)
        res = reg_test.predict(x_test)

        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, res)))

        reg = LinearRegression()
        reg.fit(features, labels)

        return reg.predict(attributes)[0] / 60

    def material_consumption(self, data, volume):
        """
        The total material consumption include the
        powder used for the object and the support
        structure together
        :param data: data from previous builds
        :param volume: the shape's volume
        :return: the material consumption in cm^3
        """
        volume += self.estimate_support(data)
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

    def estimate_support(self, data):
        """
        Estimate the volume of the support structure
        needed by taking the average of the data from
        previous builds
        :param data: data from previous builds
        :return: the estimated support volume in cm^3
        """
        support_mean = data['Support'].mean()
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

    def printer_cost(self, x, y):
        """
        Return the printer cost by determining
        the most appropriate printer to use for
        printing the shape
        :param x: the shape's x width
        :param y: the shape's y width
        :return: the printer cost in SEK
        """
        if x < 125 and y < 125:
            return self.small_machine_cost
        else:
            return self.large_machine_cost

    def recycled_savings(self, dims, consumption):
        """
        Calculate the monetary value of the powder
        that can be reused in later builds.
        :param dims: the shape's dimensions
        :param consumption: the powder consumption
        :return: the value of recycled powder in SEK
        """
        total = dims[0] * dims[1] * dims[2]
        recycled = total - consumption
        return recycled * self.powder_cost * 0.5

    def get(self):
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

        # path = "/Users/johansjoberg/IdeaProjects/costest/data.csv"
        # data = pa.read_csv(path, delimiter=';')

        # volume = self.calculate_volume(shape)
        # dims = self.calculate_dimensions(shape)
        # cons = self.material_consumption(data, volume)

        # hours = self.predict_time(data, [[dims[2], volume]])

        # cost += self.material_cost(cons)
        # cost += self.operator_cost(dims)
        # cost += self.printing_cost(hours, dims)
        # cost -= self.recycled_savings(dims, cons)

        result = {'data': {'cost': round(cost, 2)}}
        return jsonify(result)


api.add_resource(CostEstimation, '/costest')


if __name__ == "__main__":
    app.run()
