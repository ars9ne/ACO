import math
import matplotlib.pyplot as plt
import numpy as np

#константы
alpha = 1
beta = 5
dist_const = 10

class Point(object):
    def __init__(self, number, x, y, transition_probability = None):
        self.x = x
        self.y = y
        self.number = number
        self.transition_probability = None

    def get_transition_probability(self):
        return self.transition_probability

    def set_transition_probability(self, transition_probability):
        self.transition_probability = transition_probability

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
        d = math.sqrt((x2 - x1)**2 + (y2-y1)**2)
        return d


# Инициализация Точек
Point1 = Point(1, 2, 3)
Point2 = Point(2, 20, 5)
Point3 = Point(3, 18, 1)
Point4 = Point(4, 10, 20)
Point5 = Point(5, 7, 5)
points_num = 5


# Инициализация массива
x_coord = []
y_coord = []
number_coord = []
distances_array = []
pheromone_array = [[0.2 if i != j else 0.0 for i in range(points_num)] for j in range(points_num)]


def filling_coordinate_array(Point):
    x_coord.append(Point.get_location()[0])
    y_coord.append(Point.get_location()[1])
    number_coord.append(Point.get_number())

def filling_distances_array():
    global distances_array
    distances_array = [[0 for _ in range(points_num)] for _ in range(points_num)]

    for i in range(points_num):
        for j in range(points_num):
            if i != j:
                point1 = globals()[f"Point{i + 1}"]
                point2 = globals()[f"Point{j + 1}"]
                distances_array[i][j] = point1.get_distance_to(point2)

def zero_below_diagonal(matrix):
    modified_matrix = np.array(matrix, dtype=float)  # Копируем и преобразуем в numpy массив для удобства
    for i in range(len(modified_matrix)):
        for j in range(i+1, len(modified_matrix)):
            modified_matrix[j][i] = 0.0
    return modified_matrix

def print_2darray(array):
    array = np.round(array, decimals=2)
    formatted_rows = []
    for i in range(len(array)):
        formatted_rows.append(" ".join(map(str, array[i])))
    return "\n".join(formatted_rows)


filling_distances_array()
distances_array = zero_below_diagonal(distances_array)
pheromone_array = zero_below_diagonal(pheromone_array)

print(f"Исходные данные \n"
      f"Массив длин путей: \n{print_2darray(distances_array)} \n "
      f"\nМассив феромонов на путях: \n{print_2darray(pheromone_array)}")
distances_array = np.array(distances_array)

probability_array = np.zeros(points_num)
def calculate_transition_probability(P1, P2):
    index_P1 = P1.get_number() - 1
    index_P2 = P2.get_number() - 1

    a = pheromone_array[index_P1][index_P2] / distances_array[index_P1][index_P2] if distances_array[index_P1][index_P2] > 0 else 0
    psum = sum(pheromone_array[index_P1][j] / distances_array[index_P1][j] for j in range(points_num) if j != index_P1 and distances_array[index_P1][j] > 0)

    probability = a / psum if psum > 0 else 0
    return probability

probability = calculate_transition_probability(Point1, Point5)
print(probability)


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