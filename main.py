import math
import matplotlib.pyplot as plt
import numpy
import numpy as np

#константы
alpha = 1  # Важность феромона
beta = 5   # Важность дистанции
num_ants = 10  # Количество муравьёв
evaporation_rate = 0.5  # Скорость испарения феромонов
pheromone_deposit = 1  # Количество откладываемого феромона
iterations = 10000  # Количество итераций алгоритма
best_route = None
best_route_length = float('inf')

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


points = [
    Point(0, 2, 3),
    Point(1, 20, 5),
    Point(2, 18, 1),
    Point(3, 10, 20),
    Point(4, 7, 5),
    Point(5, 39, 30),
    Point(6, 30, 25),
    Point(7, 31, 41),
    Point(8, 33, 29),
    Point(9, 10, 26),
    Point(10, 32, 31),
    Point(11, 8, 23),
    Point(12, 37, 49),
    Point(13, 24, 35),
    Point(14, 18, 9),
    Point(15, 50, 21),
    Point(16, 32, 44),
    Point(17, 0, 34),
    Point(18, 49, 2),
    Point(19, 25, 30),
    Point(20, 37, 9),
    Point(21, 42, 15),
    Point(22, 17, 12),
    Point(23, 44, 17),
    Point(24, 21, 34),
    Point(25, 6, 43)

]

points_num = len(points)


# Инициализация массива
x_coord = []
y_coord = []
number_coord = []
distances_array = np.zeros((points_num, points_num))
pheromone_array = np.ones((points_num, points_num)) * 0.1


def filling_coordinate_array(Point):
    x_coord.append(Point.get_location()[0])
    y_coord.append(Point.get_location()[1])
    number_coord.append(Point.get_number())

def filling_distances_array():
    global distances_array
    distances_array = np.full((points_num, points_num), float('inf'))

    for i in range(points_num):
        for j in range(points_num):
            if i != j:
                distances_array[i][j] = points[i].get_distance_to(points[j])

filling_distances_array()
print("Инициализированный массив расстояний:")
print(distances_array)

def zero_below_diagonal(matrix):
    modified_matrix = np.array(matrix, dtype=float)
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

def calculate_route_length(route):
    length = 0
    for i in range(len(route) - 1):
        length += distances_array[route[i]][route[i + 1]]
    length += distances_array[route[-1]][route[0]]  # Возвращение в начальную точку
    return length
test_route = [0, 1, 2, 3, 4]
print("Тестовая длина маршрута:", calculate_route_length(test_route))

def select_next_point(current_point, visited):
    probabilities = []
    for i in range(points_num):
        if i not in visited:
            distance = distances_array[current_point][i]
            if distance > 0:
                pheromone_level = pheromone_array[current_point][i]
                prob = (pheromone_level ** alpha) * ((1 / distance) ** beta)
                probabilities.append(prob)
            else:
                probabilities.append(0)
        else:
            probabilities.append(0)

    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]
        next_point = np.random.choice(range(points_num), p=probabilities)
    else:
        #  если все вероятности равны нулб выбираем случайную непосещенную точку
        not_visited = [i for i in range(points_num) if i not in visited]
        next_point = np.random.choice(not_visited)
    return next_point


for iteration in range(iterations):
    all_routes = []
    for ant in range(num_ants):
        start_point = np.random.randint(0, points_num)
        route = [start_point]
        current_point = start_point
        while len(route) < points_num:
            next_point = select_next_point(current_point, route)
            route.append(next_point)
            current_point = next_point
        all_routes.append(route)

    # Обновление феромонов
    pheromone_array *= (1 - evaporation_rate)
    for route in all_routes:
        for i in range(len(route) - 1):
            pheromone_array[route[i]][route[i+1]] += pheromone_deposit
            pheromone_array[route[i+1]][route[i]] += pheromone_deposit

    for route in all_routes:
        route_length = calculate_route_length(route)
        if route_length < best_route_length:
            best_route = route
            best_route_length = route_length



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


#лист вероятностей перехода в следующую точку
init_probability_list = np.zeros(points_num)
for i in range(points_num):
    init_probability_list[i] = calculate_transition_probability(points[0], points[i])

for point in points:
    filling_coordinate_array(point)



x_coord = np.array(x_coord)
y_coord = np.array(y_coord)
number_coord = np.array(number_coord)

# print(x_coord, '\n', y_coord)
# График
plt.scatter(x_coord, y_coord, marker="o")
for i in range(len(x_coord)):
    plt.annotate(str(points[i].get_number()), (x_coord[i] + 0.2, y_coord[i] + 0.2))

plt.grid(linestyle='--')

for i in range(len(x_coord)):
    for j in range(i + 1, len(x_coord)):
        d = round(points[i].get_distance_to(points[j]), 2)
        plt.plot([x_coord[i], x_coord[j]], [y_coord[i], y_coord[j]], linestyle='-', linewidth=10/d, color='green')
        text_x = (x_coord[i] + x_coord[j]) / 2
        text_y = (y_coord[i] + y_coord[j]) / 2
        plt.text(text_x, text_y, f"{d}")

plt.xlim(0, 50)
plt.ylim(0, 50)
plt.show()
print("Лучший маршрут:", best_route)
print("Длина лучшего маршрута:", best_route_length)
# Подготовка данных для графика
x_coord = [points[i].get_location()[0] for i in best_route] + [points[best_route[0]].get_location()[0]]
y_coord = [points[i].get_location()[1] for i in best_route] + [points[best_route[0]].get_location()[1]]

# Создание графика
plt.figure(figsize=(10, 6))
plt.plot(x_coord, y_coord, marker='o', linestyle='-', color='blue')
plt.scatter(x_coord, y_coord, color='red')
plt.grid(True)

for i in range(len(best_route)):
    plt.annotate(f"{points[best_route[i]].get_number()}", (x_coord[i], y_coord[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title("Лучший маршрут алгоритма")
plt.xlabel("X координата")
plt.ylabel("Y координата")
plt.axis('equal')
plt.show()
