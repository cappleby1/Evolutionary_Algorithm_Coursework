import random, copy
from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution

POLYGON_COUNT = 50
MAX = 255 * 200 * 200
current_generation = 0
max_generations = 2000
best_fitness_so_far = 0
generations_without_improvement = 0
stagnation_threshold = 30  # e.g., 30 generations of no improvement

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
    global current_generation, max_generations
    global best_fitness_so_far, generations_without_improvement

    new_solution = solution[:]

    # --- Cooling ---
    progress = current_generation / max_generations
    cooling = 0.999 ** current_generation  # exponential cooling
    max_shift = max(1, int(10 * cooling))

    # Adaptive macro mutation: increase if no improvement
    if generations_without_improvement >= 30:
        macro_prob = 0.2          # 20% chance of big mutation
        macro_shift = max_shift * 2
    else:
        macro_prob = 0.05         # normal 5%
        macro_shift = max_shift

    micro_mode = random.random() < 0.3    # 30% chance precision nudging
    macro_mode = random.random() < macro_prob

    chance = random.random()

    # ----- Mutate points -----
    if chance < 0.5 and new_solution:
        idx = random.randrange(len(new_solution))
        polygon = new_solution[idx]
        new_polygon = polygon[:]

        new_points = []
        for (x, y) in new_polygon[1:]:
            if random.random() < rate:
                if macro_mode:
                    shift = max(5, int(macro_shift))  # big jump
                elif micro_mode:
                    shift = 2  # small refinement
                else:
                    shift = max_shift  # normal cooling-based mutation

                x += random.randint(-shift, shift)
                y += random.randint(-shift, shift)

            # Clamp coordinates
            x = max(0, min(200, x))
            y = max(0, min(200, y))
            new_points.append((x, y))

        new_polygon[1:] = new_points
        new_solution[idx] = new_polygon

    # ----- Add/remove polygons -----
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

    # ----- Shuffle / macro polygon move -----
    else:
        random.shuffle(new_solution)
        # Rare: move a polygon to a new position
        if new_solution and random.random() < 0.1:
            idx = random.randrange(len(new_solution))
            poly = new_solution.pop(idx)
            new_pos = random.randrange(len(new_solution)+1)
            new_solution.insert(new_pos, poly)

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
    global current_generation, max_generations
    global best_fitness_so_far, generations_without_improvement

    # --- Menu and seed ---
    TARGET, max_generations, pop_size, seed = menu()
    random.seed(seed)

    # --- Fitness evaluation function ---
    def evaluate(solution):
        image = draw(solution)
        diff = ImageChops.difference(image, TARGET)
        hist = diff.convert("L").histogram()
        count = sum(i * n for i, n in enumerate(hist))
        return (MAX - count) / MAX

    # --- Initialize population ---
    population = Population.generate(
        initialise,
        evaluate,
        pop_size,
        maximize=True,
        concurrent_workers=4
    )

    # --- Evolution pipeline ---
    evolution = (
        Evolution()
        .survive(fraction=0.5)
        .breed(parent_picker=select, combiner=combine)
        .mutate(mutate_function=mutate, rate=0.1, elitist=True)
        .evaluate(lazy=False)  # immediate evaluation
    )

    # --- Initialize stagnation tracking ---
    best_fitness_so_far = 0
    generations_without_improvement = 0

    # --- Main loop ---
    for i in range(max_generations):
        current_generation = i

        # Evolve population
        population = population.evolve(evolution)

        # Track best fitness and stagnation
        best = population.current_best.fitness
        if best > best_fitness_so_far:
            best_fitness_so_far = best
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        average = sum(j.fitness for j in population) / len(population)
        worst = population.current_worst.fitness

        # Normal print (no f-strings)
        print("generation =", i, "best =", best, "worst =", worst, "average =", average)

    # --- Save final best solution ---
    best_solution = max(population, key=lambda x: x.fitness)
    draw(best_solution.chromosome).save("solution.png")
    print("Best solution saved with fitness =", best_solution.fitness)


def read_config(path):
    # read JSON or ini file, return a dictionary
    ...


if __name__ == "__main__":
    run()
