
export const INITIAL_CODE = `#!/usr/bin/env python3
"""
Evolved 2‑D physics engine for a bouncing particle.
Uses genetic programming + novelty search.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------- Constants -------------------
GRAVITY = -9.81
DT = 0.01
STEPS = 100
RESTITUTION = 0.8
GROUND = 0.0

POP_SIZE = 50
MAX_GEN = 200
ELITE_COUNT = 5
MUTATION_RATE = 0.95
ARCHIVE_SIZE = 50

# ------------------- Ground Truth -------------------
def verlet_simulation():
    """Verlet integration reference trajectory."""
    x0, y0 = 0.0, 1.0
    vx0, vy0 = 1.0, 0.0
    x_prev = x0 - vx0 * DT
    y_prev = y0 - vy0 * DT
    traj = []
    for _ in range(STEPS):
        a_y = GRAVITY
        x_new = 2 * x0 - x_prev
        y_new = 2 * y0 - y_prev + a_y * DT * DT
        vx0 = (x_new - x_prev) / (2 * DT)
        vy0 = (y_new - y_prev) / (2 * DT)
        if y_new < GROUND:
            y_new = GROUND + abs(y_new - GROUND) * RESTITUTION
            vy0 = -vy0 * RESTITUTION
        traj.append([x_new, y_new])
        x_prev, y_prev = x0, y0
        x0, y0 = x_new, y_new
    return np.array(traj)

TRUE_TRAJECTORY = verlet_simulation()

# ------------------- Code blocks -------------------
BLOCKS = {
    'gravity': ['    a_y -= 9.81'],
    'air_drag': [
        '    v = math.hypot(vx, vy)',
        '    drag_coeff = 0.02',
        '    a_x -= drag_coeff * vx * v',
        '    a_y -= drag_coeff * vy * v',
    ],
    'wind': [
        '    wind_speed = 5.0',
        '    a_x += wind_speed * math.sin(2 * math.pi * x)'
    ],
    'magnus': [
        '    lift_coeff = 0.01',
        '    a_y += lift_coeff * vx'
    ],
    'ground_friction': [
        '    if y <= 0.0:',
        '        a_x -= 0.5 * vx'
    ],
    'euler': [
        '    vx = vx + a_x * dt',
        '    vy = vy + a_y * dt',
        '    x = x + vx * dt',
        '    y = y + vy * dt'
    ],
    'rk4': [
        '    # RK4 integration',
        '    k1x = vx',
        '    k1y = vy',
        '    k1vx = a_x',
        '    k1vy = a_y',
        '    k2x = vx + 0.5 * k1vx * dt',
        '    k2y = vy + 0.5 * k1vy * dt',
        '    k2vx = a_x',
        '    k2vy = a_y',
        '    k3x = vx + 0.5 * k2vx * dt',
        '    k3y = vy + 0.5 * k2vy * dt',
        '    k3vx = a_x',
        '    k3vy = a_y',
        '    k4x = vx + k3vx * dt',
        '    k4y = vy + k3vy * dt',
        '    k4vx = a_x',
        '    k4vy = a_y',
        '    x = x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0',
        '    y = y + dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0',
        '    vx = vx + dt * (k1vx + 2 * k2vx + 2 * k3vx + k4vx) / 6.0',
        '    vy = vy + dt * (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6.0'
    ],
    'bounce': [
        '    if y < 0.0:',
        '        y = 0.0 + abs(y - 0.0) * 0.8',
        '        vy = -vy * 0.8'
    ],
}
force_blocks = ['gravity', 'air_drag', 'wind', 'magnus', 'ground_friction']

# ------------------- Individual class -------------------
class Individual:
    def __init__(self, blocks):
        self.blocks = blocks
        self.build_and_compile()

    def build_and_compile(self):
        lines = [
            'def update(state, dt):',
            '    x, y, vx, vy = state',
            '    a_x = 0.0',
            '    a_y = 0.0'
        ]
        for bid in self.blocks:
            lines.extend(BLOCKS[bid])
        lines.append('    return [x, y, vx, vy]')
        self.code = '\\n'.join(lines)
        ns = {}
        exec(self.code, globals(), ns)
        self.func = ns['update']

    def mutate(self):
        """Simple mutation: replace a random force or integration block."""
        idx = random.randint(0, len(self.blocks) - 1)
        bid = self.blocks[idx]
        if bid in force_blocks and bid != 'gravity':
            new = random.choice(force_blocks)
            while new == bid:
                new = random.choice(force_blocks)
            self.blocks[idx] = new
        elif bid in ('euler', 'rk4'):
            new = random.choice(['euler', 'rk4'])
            while new == bid:
                new = random.choice(['euler', 'rk4'])
            self.blocks[idx] = new
        # gravity and bounce stay fixed
        self.build_and_compile()

    def __str__(self):
        return self.code

# ------------------- Utility functions -------------------
def create_random_individual():
    """Generate a syntactically valid individual."""
    blocks = ['gravity']
    optional = [b for b in force_blocks if b != 'gravity']
    n_opt = random.randint(0, len(optional))
    blocks += random.sample(optional, n_opt)
    blocks.append(random.choice(['euler', 'rk4']))
    blocks.append('bounce')
    return Individual(blocks)

archive_behaviors = []

def evaluate(ind):
    """Simulate and score an individual."""
    state = [0.0, 1.0, 1.0, 0.0]
    traj = []
    t0 = time.time()
    for _ in range(STEPS):
        state = ind.func(state, DT)
        traj.append(state.copy())
    t1 = time.time()
    runtime = t1 - t0
    traj = np.array(traj)          # (STEPS, 4)
    pos  = traj[:, :2]             # (STEPS, 2)
    mse  = np.mean((pos - TRUE_TRAJECTORY[:, :2])**2)
    accuracy = 1.0 / (1.0 + mse)
    efficiency = 1.0 / (1.0 + runtime)
    behavior = pos.flatten()
    if not archive_behaviors:
        novelty = 0.0
    else:
        dists = [np.linalg.norm(b - behavior) for b in archive_behaviors]
        novelty = np.mean(dists)
    score = 0.4 * accuracy + 0.3 * efficiency + 0.3 * novelty
    return {
        'individual': ind,
        'score': score,
        'accuracy': accuracy,
        'efficiency': efficiency,
        'novelty': novelty,
        'behavior': behavior
    }

def tournament_selection(pop, k=5):
    """Select one individual by tournament."""
    chosen = random.sample(pop, k)
    chosen.sort(key=lambda e: e['score'], reverse=True)
    return chosen[0]['individual']

def crossover(p1, p2):
    """Line‑based safe crossover."""
    idx = random.randint(1, len(p1.blocks)-1)
    child_blocks = p1.blocks[:idx] + p2.blocks[idx:]
    return Individual(child_blocks)

# ------------------- Main evolution loop -------------------
def main():
    population = [create_random_individual() for _ in range(POP_SIZE)]
    best_overall = None
    best_score_overall = -1.0

    for gen in range(1, MAX_GEN + 1):
        evaluated = [evaluate(ind) for ind in population]
        evaluated.sort(key=lambda e: e['score'], reverse=True)
        elite = evaluated[:ELITE_COUNT]

        if elite[0]['score'] > best_score_overall:
            best_score_overall = elite[0]['score']
            best_overall = elite[0]

        for e in evaluated:
            archive_behaviors.append(e['behavior'])
            if len(archive_behaviors) > ARCHIVE_SIZE:
                archive_behaviors.pop(0)

        if gen % 20 == 0 or gen == 1 or gen == MAX_GEN:
            print(f"Gen {gen:3d} | Best score: {elite[0]['score']:.4f} | "
                  f"Accuracy: {elite[0]['accuracy']:.4f} | Novelty: {elite[0]['novelty']:.4f}")

        if elite[0]['accuracy'] > 0.92 and len(archive_behaviors) > 20:
            print(f"Early stop: accuracy > 0.92 and archive > 20 at generation {gen}")
            break

        children = []
        while len(children) < POP_SIZE - ELITE_COUNT:
            p1 = tournament_selection(evaluated)
            p2 = tournament_selection(evaluated)
            child = crossover(p1, p2)
            if random.random() < MUTATION_RATE:
                child.mutate()
            children.append(child)
        population = [e['individual'] for e in elite] + children

    # ------------------- Output -------------------
    best_ind = best_overall['individual']
    print("\\n=== Best Evolved \`update(state,dt)\` ===")
    print(best_ind.code)

    with open('evolved_physics.py', 'w') as f:
        f.write('import math\\n\\n')
        f.write(best_ind.code + '\\n')

    # Trajectory of the best individual
    state = [0.0, 1.0, 1.0, 0.0]
    traj = []
    for _ in range(STEPS):
        state = best_ind.func(state, DT)
        traj.append(state.copy())
    traj = np.array(traj)

    # Plot comparison
    plt.figure(figsize=(6,4))
    plt.plot(TRUE_TRAJECTORY[:,0], TRUE_TRAJECTORY[:,1], 'r-', label='Verlet (Truth)')
    plt.plot(traj[:,0], traj[:,1], 'b--', label='Evolved')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png')
    plt.show()

    # GIF animation
    fig, ax = plt.subplots()
    line_truth, = ax.plot([], [], 'r-')
    line_evolved, = ax.plot([], [], 'b-')
    ax.set_xlim(np.min(TRUE_TRAJECTORY[:,0]) - 1, np.max(TRUE_TRAJECTORY[:,0]) + 1)
    ax.set_ylim(np.min(TRUE_TRAJECTORY[:,1]) - 1, np.max(TRUE_TRAJECTORY[:,1]) + 1)
    ax.set_title('Bouncing particle')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    def init():
        line_truth.set_data([], [])
        line_evolved.set_data([], [])
        return line_truth, line_evolved

    def animate(i):
        line_truth.set_data(TRUE_TRAJECTORY[:i+1,0], TRUE_TRAJECTORY[:i+1,1])
        line_evolved.set_data(traj[:i+1,0], traj[:i+1,1])
        return line_truth, line_evolved

    ani = FuncAnimation(fig, animate, frames=STEPS, init_func=init, blit=True, interval=50)
    ani.save('bounce.gif', writer=PillowWriter(fps=20))
    print("\\nSaved GIF as 'bounce.gif' and plot as 'trajectory_comparison.png'.")

if __name__ == "__main__":
    main()
`;

export const MOCK_OUTPUT = `Gen   1 | Best score: 0.3451 | Accuracy: 0.1532 | Novelty: 0.4823
Gen  20 | Best score: 0.7812 | Accuracy: 0.6789 | Novelty: 0.8123
Gen  40 | Best score: 0.8923 | Accuracy: 0.8511 | Novelty: 0.6521
Gen  60 | Best score: 0.9415 | Accuracy: 0.9155 | Novelty: 0.5134
Gen  62 | Best score: 0.9567 | Accuracy: 0.9241 | Novelty: 0.4988
Early stop: accuracy > 0.92 and archive > 20 at generation 62

=== Best Evolved \`update(state,dt)\` ===
...
`;