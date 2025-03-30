import random
from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution

# THIS IS THE ORIGINAL CODE GIVEN FROM WHICH I MADE CHANGES - KEPT PURELY FOR REFERENCE

POLYGON_COUNT = 10
MAX = 255 * 200 * 200
TARGET = Image.open("8b.png")
TARGET.load()  # read image and close the file


def make_polygon():
    # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190
    R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    A = random.randint(30, 60)
    x1, x2, x3 = random.randint(10, 189), random.randint(10, 189), random.randint(10, 189)
    y1, y2, y3 = random.randint(10, 189), random.randint(10, 189), random.randint(10, 189)
    return [(R, G, B, A), (x1, y1), (x2, y2), (x3, y3)]


def initialise():
    return [make_polygon() for i in range(POLYGON_COUNT)]


def mutate(solution, rate):
    chance = random.random()
    if chance < 0.5:
        # mutate points
        polygon = random.choice(solution)
        coords = [x for point in polygon[1:] for x in point]
        coords = [x + (random.randint(-10, 10) if random.random() < rate else 0) for x in coords]
        coords = [max(0, min(int(x), 200)) for x in coords]
        polygon[1:] = list(zip(coords[::2], coords[1::2]))

    else:
        # reorder polygons
        new_solution = solution[:]
        random.shuffle(new_solution)
        solution = new_solution
    return solution


def select(population):
    return [random.choice(population) for i in range(2)]


def combine(*parents):
    return [a if random.random() < 0.5 else b for a, b in zip(*parents)]


# Checks how close to target image the solution is
def evaluate(solution):
    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)  # Calculates the pixel-wise absolute difference
    hist = diff.convert("L").histogram()  # Converts to greyscale & then histogram
    count = sum(i * n for i, n in enumerate(hist))  # Computes the weighted sum of pixel differences
    return (MAX - count) / MAX  # Creates fitness score


# Creates solution image
def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])
    return image


def run(generations=500, population_size=100, seed=31):
    # for reproducibility
    random.seed(seed)

    # initialization
    population = Population.generate(initialise, evaluate, population_size, maximize=True, concurrent_workers=4)
    evolution = (Evolution().survive(fraction=0.5)
                 .breed(parent_picker=select, combiner=combine)
                 .mutate(mutate_function=mutate, rate=0.1)
                 .evaluate())

    for i in range(generations):
        population = population.evolve(evolution)
        print("i =", i, " best =", population.current_best.fitness, " worst =",
              population.current_worst.fitness)
    draw(population[0].chromosome).save("solution.png")


def read_config(path):
    # read JSON or ini file, return a dictionary
    ...


if __name__ == "__main__":
    # params = read_config(sys.argv[1])
    # run(**params)
    run()
