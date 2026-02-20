import random, copy
from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution

POLYGON_COUNT = 70
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

    new_solution = solution[:]  # shallow copy

    # --- Cooling schedule ---
    cooling = max(0.2, 0.995 ** current_generation)
    max_shift = max(1, int(10 * cooling))

    # --- Determine macro/micro probabilities ---
    progress_ratio = current_generation / max_generations

    # Reduce macro probability late in evolution
    if progress_ratio < 0.7:
        macro_prob = 0.3  # early: big jumps
        multi_poly_prob = 0.05
    else:
        macro_prob = 0.05  # late: mostly fine tuning
        multi_poly_prob = 0.01

    micro_mode = random.random() < 0.3          # 30% precision nudging
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
                    shift = max(5, int(max_shift))
                elif micro_mode:
                    shift = 2
                else:
                    shift = max_shift

                x += random.randint(-shift, shift)
                y += random.randint(-shift, shift)

            # clamp coordinates
            x = max(0, min(200, x))
            y = max(0, min(200, y))
            new_points.append((x, y))

        new_polygon[1:] = new_points
        new_solution[idx] = new_polygon

    # ----- Rare single polygon replacement (mostly early) -----
    if new_solution and random.random() < (0.05 * (1 - progress_ratio)):
        replace_idx = random.randrange(len(new_solution))
        new_solution[replace_idx] = make_polygon()

    # ----- Rare multi-polygon replacement (early only) -----
    if new_solution and random.random() < (multi_poly_prob * (1 - progress_ratio)):
        num_replace = random.randint(1, 3)
        for _ in range(num_replace):
            idx = random.randrange(len(new_solution))
            new_solution[idx] = make_polygon()

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
        # Reduce shuffling late in evolution
        if random.random() < (0.1 * (1 - progress_ratio)):
            random.shuffle(new_solution)
            if new_solution and random.random() < 0.05:
                idx = random.randrange(len(new_solution))
                poly = new_solution.pop(idx)
                new_pos = random.randrange(len(new_solution)+1)
                new_solution.insert(new_pos, poly)

    return new_solution


def select(population):

    # Stops identical parents
    subset1 = random.sample(population, k=5)
    parent1 = max(subset1, key=lambda x: x.fitness)

    remaining = [p for p in population if p != parent1]
    subset2 = random.sample(remaining, k=5)
    parent2 = max(subset2, key=lambda x: x.fitness)

    return [parent1, parent2]


def combine(*parents):
    p1, p2 = parents  # parents are lists of polygons

    # Optional: assign simple fitness weights (you can pass in real fitness if available)
    f1 = getattr(p1, "fitness", 1.0)  # fallback to 1.0 if no fitness attribute
    f2 = getattr(p2, "fitness", 1.0)
    total = f1 + f2
    w1 = f1 / total if total > 0 else 0.5  # probability to choose from parent 1

    # --- Segment crossover ---
    cut = random.randint(0, min(len(p1), len(p2)))
    child = p1[:cut] + p2[cut:]

    # --- Weighted random swaps for diversity ---
    for i in range(len(child)):
        if random.random() < 0.1:
            poly_from_p1 = p1[i] if i < len(p1) else None
            poly_from_p2 = p2[i] if i < len(p2) else None

            if poly_from_p1 and poly_from_p2:
                child[i] = poly_from_p1 if random.random() < w1 else poly_from_p2
            elif poly_from_p1:
                child[i] = poly_from_p1
            elif poly_from_p2:
                child[i] = poly_from_p2
            # else: keep child[i] as-is

    return child

# Creates solution image
def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])
    return image


def run():
    TARGET, max_generations, pop_size, seed = menu()
    random.seed(seed)

    best_solution = None
    best_fitness_so_far = 0
    generations_without_improvement = 0

    def evaluate(solution):
        image = draw(solution)
        diff = ImageChops.difference(image, TARGET)
        hist = diff.convert("L").histogram()
        count = sum(i * n for i, n in enumerate(hist))
        return (MAX - count) / MAX

    # Initialize population
    population = Population.generate(initialise, evaluate, pop_size, maximize=True, concurrent_workers=8)
    evolution = (Evolution().survive(fraction=0.5)
                 .breed(parent_picker=select, combiner=combine)
                 .mutate(mutate_function=mutate, rate=0.1, elitist=True)
                 .evaluate(lazy=True))

    for i in range(max_generations):
        population = population.evolve(evolution)

        # Compute fitness stats
        best = population.current_best.fitness
        worst = population.current_worst.fitness
        total = 0
        for j in population:
            total += j.fitness
        average = total / len(population)

        # Update best solution visually
        if best > best_fitness_so_far:
            best_fitness_so_far = best
            best_solution = population.current_best.chromosome[:]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Normal print statements
        print("generation =", i, "best =", best, "worst =", worst, "average =", average)

        # Save intermediate images every 50 generations
        if i % 50 == 0 or i == max_generations - 1:
            draw(best_solution).save("solution_gen_" + str(i) + ".png")

    # Save final best solution
    draw(best_solution).save("solution_final.png")
    print("Final solution saved with fitness =", best_fitness_so_far)


def read_config(path):
    # read JSON or ini file, return a dictionary
    ...


if __name__ == "__main__":
    run()
