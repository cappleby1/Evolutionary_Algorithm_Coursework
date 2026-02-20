import random, copy
from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution

POLYGON_COUNT = 50
MAX = 255 * 200 * 200

image = input("Enter a, b or c to select targetb: ")

match image:
    case "a":
        TARGET = Image.open("8a.png")
    case "b":
        TARGET = Image.open("8b.png")
    case "c":
        TARGET = Image.open("8c.png")
    case _:
        print("Invalid file name entered. Please select one of the valid options (8a.png, 8b.png, 8c.png).")

TARGET.load()

def menu():


    generations = int(input("Enter number of generations: "))
    pop_size = int(input("Enter population size: "))

    random_seed = str(input("Would you like a random seed? (y/n): "))
    if random_seed == "y":
        seed = random.randint(1, MAX)
    else:
        seed = 35

    return TARGET, generations, pop_size, seed


def make_polygon():
    # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190
    R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    A = random.randint(30, 60)
    x1, x2, x3, x4 = random.randint(10, 189), random.randint(10, 189), random.randint(10, 189), random.randint(10, 189)
    y1, y2, y3, y4 = random.randint(10, 189), random.randint(10, 189), random.randint(10, 189), random.randint(10, 189)
    return [(R, G, B, A), (x1, y1), (x2, y2), (x3, y3), (x4, y4)]


# Create image for size required
def initialise():
    return [make_polygon() for i in range(POLYGON_COUNT)]


def mutate(solution, rate):
    solution = copy.deepcopy(solution)
    chance = random.random()
    if chance < 0.5:
        # mutate points
        polygon = random.choice(solution)
        coords = [x for point in polygon[1:] for x in point]
        coords = [x + (random.randint(-10, 10) if random.random() < rate else 0) for x in coords]
        coords = [max(0, min(int(x), 200)) for x in coords]
        polygon[1:] = list(zip(coords[::2], coords[1::2]))

    # add polygon
    elif 0.5 < chance < 0.7:
        if len(solution) < 100:
            solution.append(make_polygon())
        else:
            delete = random.randint(0, (len(solution) - 1))
            solution.pop(delete)

    # remove polygon
    elif 0.7 < chance < 0.75:
        if len(solution) > 0:
            delete = random.randint(0, (len(solution) - 1))
            solution.pop(delete)
        else:
            solution.append(make_polygon())

    else:
        # reorder polygons
        new_solution = solution[:]
        random.shuffle(new_solution)
        solution = new_solution
    return solution


def select(population):
    subset1 = random.choices(population, k=5)
    subset2 = random.choices(population, k=5)

    highest1 = max(subset1, key=lambda x: x.fitness)
    highest2 = max(subset2, key=lambda x: x.fitness)

    return [highest1, highest2]


def combine(*parents):
    return [a if random.random() < 0.5 else b for a, b in zip(*parents)]


# Checks how close solution is to the target image
def evaluate(solution):
    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)  # Calculates pixel-wise absolute difference between images
    hist = diff.convert("L").histogram()  # Converts to greyscale & histogram of pixel intensity
    count = sum(i * n for i, n in enumerate(hist))  # Computes weighted sum of pixel differences
    return (MAX - count) / MAX  # Normalise to fitness value


# Creates solution image
def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])
    return image


def run():
    TARGET, generations, pop_size, seed = menu()

    random.seed(seed)

    # initialization
    population = Population.generate(initialise, evaluate, pop_size, maximize=True, concurrent_workers=4)
    evolution = (Evolution().survive(fraction=0.5)
                 .breed(parent_picker=select, combiner=combine)
                 .mutate(mutate_function=mutate, rate=0.1)
                 .evaluate())

    for i in range(generations):

        total = 0

        population = population.evolve(evolution)

        for j in population:
            total += j.fitness

        average = total / len(population)

        population = population.evolve(evolution)

        print("generation =", i, " best =", population.current_best.fitness, " worst =",

              population.current_worst.fitness, "average=", average)

    best = population[0]

    for i in population:

        if i.fitness > best.fitness:
            best = i

    draw(best.chromosome).save("solution.png")


def read_config(path):
    # read JSON or ini file, return a dictionary
    ...


if __name__ == "__main__":
    run()
