import math
import matplotlib.pyplot as plt
import numpy
import numpy as np


class Point(object):
    def __init__(self, number, x, y):
        self.x = x
        self.y = y
        self.number = number

    def get_location(self):
        return self.x, self.y

    def get_number(self):
        return self.number

    def set_location(self, x, y):
        self.x = x
        self.y = y

    def set_number(self, number):
        self.number = number

    def get_distance_to(self, Point):
        x1 = self.x
        x2 = Point.get_location()[0]
        y1 = self.y
        y2 = Point.get_location()[1]
        d = math.sqrt((x1 - y1)**2 + (x2-y2)**2)
        return d


# Инициализация Точек
Point1 = Point(1, 2, 3)
Point2 = Point(2, 20, 5)
Point3 = Point(3, 18, 1)
Point4 = Point(4, 10, 20)
Point5 = Point(5, 7, 5)
Point6 = Point(6, 40, 50)
points_num = 6
# Инициализация массива
x_coord = []
y_coord = []
number_coord = []


def filling_coordinate_array(Point):
    x_coord.append(Point.get_location()[0])
    y_coord.append(Point.get_location()[1])
    number_coord.append(Point.get_number())


for i in range(1, points_num + 1):
    point = globals()[f"Point{i}"]
    filling_coordinate_array(point)

x_coord = np.array(x_coord)
y_coord = np.array(y_coord)
number_coord = np.array(number_coord)

# print(x_coord, '\n', y_coord)
# График
plt.scatter(x_coord, y_coord, marker="o")
for i in range(len(x_coord)):
    point = globals()[f"Point{i + 1}"]
    plt.annotate(str(point.get_number()), (x_coord[i] + 0.2, y_coord[i] + 0.2))
plt.grid(linestyle='--')
# Соединяем точки линиями
for i in range(len(x_coord)):
    for j in range(i + 1, len(x_coord)):
        point = globals()[f"Point{i + 1}"]
        point1 = globals()[f"Point{j + 1}"]
        d = round(point.get_distance_to(point1), 2) # дистанция между двумя точками
        plt.plot([x_coord[i], x_coord[j]], [y_coord[i], y_coord[j]], linestyle='-', linewidth= 10/d, color='green')
        text_x = (x_coord[i] + x_coord[j]) / 2
        text_y = (y_coord[i] + y_coord[j]) / 2
        plt.text(text_x, text_y, f"{d}")

plt.xlim(0, 50)
plt.ylim(0, 50)
plt.show()