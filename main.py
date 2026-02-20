import random, copy
from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution

POLYGON_COUNT = 50
MAX = 255 * 200 * 200
max_generations = 2000
current_generation = 0

def menu():

    image = input("Enter a, b or c to select target: ")

    match image:
        case "a":
            TARGET = Image.open("8a.png")
        case "b":
            TARGET = Image.open("8b.png")
        case "c":
            TARGET = Image.open("8c.png")
        case _:
            raise ValueError("Invalid selection. Choose a, b, or c.")

    TARGET.load()

    max_generations = int(input("Enter number of generations: "))
    pop_size = int(input("Enter population size: "))

    random_seed = str(input("Would you like a random seed? (y/n): "))
    if random_seed == "y":
        seed = random.randint(1, MAX)
    else:
        seed = 35

    return TARGET, max_generations, pop_size, seed


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
    chance = random.random()

    # shallow copy the solution list
    new_solution = solution[:]

    # Cool mutation as generations increases - larger jumps to begin with then smaller
    progress = current_generation / max_generations
    cooling = 0.999 ** current_generation   # exponential cooling
    max_shift = max(1, int(10 * cooling))   # always at least 1

    if progress < 0.5:
        cooling = 1
    else:
        cooling = 1 - ((progress - 0.5) * 2)

    if chance < 0.5 and new_solution:
        # choose index instead of object
        idx = random.randrange(len(new_solution))

        # copy only the polygon we modify
        polygon = new_solution[idx]
        new_polygon = polygon[:]  # shallow copy polygon

        new_points = []
        # decide mutation mode
        micro_mode = random.random() < 0.3  # 30% chance of precision mutation

        for (x, y) in new_polygon[1:]:
            if random.random() < rate:
                if micro_mode:
                    # tiny refinement shift
                    shift = 2
                else:
                    # normal cooling shift
                    shift = max_shift

                x += random.randint(-shift, shift)
                y += random.randint(-shift, shift)

            # clamp to image boundaries
            x = max(0, min(200, x))
            y = max(0, min(200, y))

            new_points.append((x, y))

        new_polygon[1:] = new_points
        new_solution[idx] = new_polygon

    elif chance < 0.7:
        if len(new_solution) < 100:
            new_solution.append(make_polygon())
        elif new_solution:
            new_solution.pop(random.randrange(len(new_solution)))

    elif chance < 0.75:
        if new_solution:
            new_solution.pop(random.randrange(len(new_solution)))
        else:
            new_solution.append(make_polygon())

    else:
        random.shuffle(new_solution)

    return new_solution


def select(population):

    # Stops identical parents
    subset1 = random.sample(population, k=7)
    parent1 = max(subset1, key=lambda x: x.fitness)

    remaining = [p for p in population if p != parent1]
    subset2 = random.sample(remaining, k=7)
    parent2 = max(subset2, key=lambda x: x.fitness)

    return [parent1, parent2]


def combine(*parents):
    return [a if random.random() < 0.5 else b for a, b in zip(*parents)]


# Creates solution image
def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])
    return image


def run():
    global max_generations, current_generation

    TARGET, max_generations, pop_size, seed = menu()

    random.seed(seed)

    # Checks how close solution is to the target image
    def evaluate(solution):
        image = draw(solution)
        diff = ImageChops.difference(image, TARGET)  # Calculates pixel-wise absolute difference between images
        hist = diff.convert("L").histogram()  # Converts to greyscale & histogram of pixel intensity
        count = sum(i * n for i, n in enumerate(hist))  # Computes weighted sum of pixel differences
        return (MAX - count) / MAX  # Normalise to fitness value

    # initialization
    population = Population.generate(initialise, evaluate, pop_size, maximize=True, concurrent_workers=8)
    evolution = (Evolution().survive(fraction=0.5)
                 .breed(parent_picker=select, combiner=combine)
                 .mutate(mutate_function=mutate, rate=0.1, elitist=True)
                 .evaluate(lazy=True))
    
    for i in range(max_generations):

        current_generation = i

        population = population.evolve(evolution)

        total = sum(j.fitness for j in population)
        average = total / len(population)

        print("generation =", i,
            "best =", population.current_best.fitness,
            "worst =", population.current_worst.fitness,
            "average =", average)

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
