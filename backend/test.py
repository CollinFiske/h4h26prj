# test.py
# Run: python3 test.py

import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
from matplotlib.colors import ListedColormap

# -----------------------
# CONFIG
# -----------------------
WIDTH = 25
HEIGHT = 18

BUILDING_DENSITY = 0.18
FIRE_DENSITY = 0.05
AIR_DENSITY = 0.08
NUM_GOALS = 3

AIR_EXTRA_COST = 5
DISABILITY = None


# -----------------------
# Runtime input
# -----------------------
def ask_disability():
    while True:
        ans = input("Is the person disabled? (yes/no): ").strip().lower()
        if ans in ("yes", "y"):
            return True
        if ans in ("no", "n"):
            return False
        print("Please type 'yes' or 'no'.")


# -----------------------
# Random Grid Generator
# -----------------------
def generate_random_grid():
    grid = []
    for _ in range(HEIGHT):
        row = []
        for _ in range(WIDTH):
            r = random.random()
            if r < BUILDING_DENSITY:
                row.append("#")
            elif r < BUILDING_DENSITY + FIRE_DENSITY:
                row.append("F")
            elif r < BUILDING_DENSITY + FIRE_DENSITY + AIR_DENSITY:
                row.append("A")
            else:
                row.append(".")
        grid.append(row)

    # Place exactly ONE start
    while True:
        sx = random.randrange(WIDTH)
        sy = random.randrange(HEIGHT)
        if grid[sy][sx] == ".":
            grid[sy][sx] = "S"
            start = (sx, sy)
            break

    # Place goals
    goals = []
    while len(goals) < NUM_GOALS:
        gx = random.randrange(WIDTH)
        gy = random.randrange(HEIGHT)
        if grid[gy][gx] == ".":
            grid[gy][gx] = "G"
            goals.append((gx, gy))

    return grid, start, set(goals)


# -----------------------
# A* Multi-Goal
# -----------------------
def in_bounds(x, y):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT


def is_blocked(grid, x, y):
    c = grid[y][x]
    if c in ("#", "F"):
        return True
    if DISABILITY and c == "A":
        return True
    return False


def step_cost(grid, x, y):
    cost = 1
    if (not DISABILITY) and grid[y][x] == "A":
        cost += AIR_EXTRA_COST
    return cost


def heuristic(x, y, goals):
    return min(abs(x - gx) + abs(y - gy) for (gx, gy) in goals)


def a_star(grid, start, goals):
    sx, sy = start
    pq = [(heuristic(sx, sy, goals), 0, sx, sy)]
    best_g = {(sx, sy): 0}
    parent = {}

    moves = [(1,0), (-1,0), (0,1), (0,-1)]

    while pq:
        f, g, x, y = heapq.heappop(pq)

        if (x, y) in goals:
            end = (x, y)
            path = [end]
            while end in parent:
                end = parent[end]
                path.append(end)
            path.reverse()
            return (x, y), path, g

        if g != best_g.get((x, y), None):
            continue

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue
            if is_blocked(grid, nx, ny):
                continue

            ng = g + step_cost(grid, nx, ny)
            if (nx, ny) not in best_g or ng < best_g[(nx, ny)]:
                best_g[(nx, ny)] = ng
                parent[(nx, ny)] = (x, y)
                heapq.heappush(pq, (ng + heuristic(nx, ny, goals), ng, nx, ny))

    return None, None, None


# -----------------------
# Plotting (SIDE BY SIDE, NO OVERLAP)
# Base grid colors only for: free/building/fire/air
# Start/goals/path are OVERLAYS (so you never "see multiple starts")
# -----------------------
# Base array values:
# 0 free (white)
# 1 building (blue)
# 2 fire (red)
# 3 air (orange)
BASE_CMAP = ListedColormap(["white", "blue", "red", "orange"])


def build_base_array(grid):
    arr = np.zeros((HEIGHT, WIDTH), dtype=int)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            c = grid[y][x]
            if c == "#":
                arr[y, x] = 1
            elif c == "F":
                arr[y, x] = 2
            elif c == "A":
                arr[y, x] = 3
            else:
                arr[y, x] = 0  # includes '.', 'S', 'G'
    return arr


def overlay_markers(ax, start, goals, path=None):
    # Start: purple dot
    ax.scatter([start[0]], [start[1]], s=120, marker="o", edgecolors="black", linewidths=1.0)

    # Goals: green stars
    if goals:
        gx = [g[0] for g in goals]
        gy = [g[1] for g in goals]
        ax.scatter(gx, gy, s=160, marker="*", edgecolors="black", linewidths=1.0)

    # Path: yellow line
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, linewidth=3)  # default color; if you want forced yellow, tell me


def show_side_by_side(grid, start, goals, path=None, cost=None, chosen_goal=None):
    base = build_base_array(grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Left: Original
    axes[0].imshow(base, cmap=BASE_CMAP, interpolation="nearest")
    axes[0].set_title("Original Grid")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(True, linewidth=0.5)
    overlay_markers(axes[0], start, goals, path=None)

    # Right: Final (same base, path overlay)
    title = "Final Path"
    if chosen_goal is not None:
        title += f" → {chosen_goal}"
    if cost is not None:
        title += f" | cost={cost}"

    axes[1].imshow(base, cmap=BASE_CMAP, interpolation="nearest")
    axes[1].set_title(title)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].grid(True, linewidth=0.5)
    overlay_markers(axes[1], start, goals, path=path)

    # Bottom info bar (outside plots)
    air_rule = "blocked" if DISABILITY else f"allowed (+{AIR_EXTRA_COST} cost)"
    info = [
        f"Disabled: {'YES' if DISABILITY else 'NO'}",
        "Blue=Buildings (blocked)   Red=Fire (blocked)   Orange=Air",
        f"Air rule: {air_rule}",
        "Purple dot=Start   Green stars=Safe centers   Line=Path (right)",
    ]
    fig.text(0.5, 0.01, "   |   ".join(info), ha="center", va="bottom", fontsize=10)

    plt.show()


# -----------------------
# Main
# -----------------------
def main():
    global DISABILITY
    DISABILITY = ask_disability()

    grid, start, goals = generate_random_grid()

    # Debug proof: there is exactly one start
    s_count = sum(row.count("S") for row in grid)
    print("\nNumber of S in grid:", s_count)
    print("START:", start)
    print("SAFE CENTERS:", goals)

    chosen_goal, path, cost = a_star(grid, start, goals)

    if chosen_goal is None:
        print("No reachable safe center.")
        show_side_by_side(grid, start, goals, path=None, cost=None, chosen_goal=None)
        return

    print("\nChosen safe center:", chosen_goal)
    print("Total cost:", cost)
    print("Path:", path)

    show_side_by_side(grid, start, goals, path=path, cost=cost, chosen_goal=chosen_goal)


if __name__ == "__main__":
    main()