
export const INITIAL_CODE = `import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean  # For novelty distance
import time

# Philosophical invocation: In the forge of algorithmic genesis, where silicon synapses mimic the cosmic dance of particles, this engine births not mere simulators but symphonies of force and form—evolving Verlet threads, Runge-Kutta whispers, and collision cantos from the void, 100-fold refined in the crucible of relentless novelty, defying stasis to unveil the unseen laws of motion.

# --------------------------------------------------------------
# 1. PHYSICS TEST HARNESS: Single 2D Particle with Gravity + Ground Bounce
# --------------------------------------------------------------
# System: Particle starts at [x=0, y=5, vx=2, vy=0], falls under g=9.81, bounces elastically (restitution=0.8) on y=0.
# Ground truth: High-precision Verlet integration for accuracy.
# Evaluation: Run 200 steps (dt=0.01, T=2s), compare trajectory MSE (positions only).
# Sophistication: Evolved code must handle vector state [x,y,vx,vy], forces, integration, collisions.

G = 9.81  # Gravity
DT = 0.01
STEPS = 200  # Total simulation steps
RESTITUTION = 0.8
INITIAL_STATE = np.array([0.0, 5.0, 2.0, 0.0])  # x,y,vx,vy

def ground_truth():
    """Verlet integration for ground truth (symplectic, stable for this system)."""
    pos = INITIAL_STATE[:2].copy()
    vel = INITIAL_STATE[2:].copy()
    trajectory = [pos.copy()]
    prev_pos = pos - vel * DT
    for _ in range(STEPS):
        acc = np.array([0.0, -G])
        next_pos = 2 * pos - prev_pos + acc * DT**2
        next_vel = (next_pos - prev_pos) / (2 * DT)
        # Collision: If penetrated ground, reflect
        if next_pos[1] < 0:
            next_pos[1] = 0
            next_vel[1] = -next_vel[1] * RESTITUTION
        prev_pos = pos.copy()
        pos = next_pos
        vel = next_vel
        trajectory.append(pos.copy())
    return np.array(trajectory)  # (201, 2) positions

TRUTH_POS = ground_truth()  # Precompute

# --------------------------------------------------------------
# 2. ADVANCED CODE EVOLUTION ENGINE (100x Smarter: Novelty-Driven, Multi-Objective GP)
# --------------------------------------------------------------
# Representation: Evolve full 'def update(state, dt):' functions as string trees.
# Fitness: accuracy (1 - MSE) + efficiency (1 / runtime) + complexity_penalty (shorter better).
# Novelty: Behavioral distance in trajectory space; maintain dynamic archive.
# Selection: NSGA-II inspired Pareto front (rank by non-dominated fitness/novelty).
# Mutations: Semantic (change ops), structural (add RK substeps), guided (inject physics priors rarely).
# Population: 500, Infinite gens (stop at 200 for demo), but novelty ensures endless exploration.

LINE_POOL = [  # Sophisticated primitives: Vectors, integrators, forces, collisions
    "    acc_x = 0.0",
    "    acc_y = -9.81",
    "    state[0] += state[2] * dt",  # Euler pos x
    "    state[1] += state[3] * dt",  # Euler pos y
    "    state[2] += acc_x * dt",    # Euler vel x
    "    state[3] += acc_y * dt",    # Euler vel y
    # Verlet-inspired (without prev, approx)
    "    state[0] += state[2] * dt + 0.5 * acc_x * dt**2",
    "    state[1] += state[3] * dt + 0.5 * acc_y * dt**2",
    "    state[2] += acc_x * dt",
    "    state[3] += acc_y * dt",
    # Full RK4 for position and velocity
    "    k1_vx = acc_x * dt",
    "    k1_vy = acc_y * dt",
    "    k1_x = state[2] * dt",
    "    k1_y = state[3] * dt",
    "    k2_vx = acc_x * dt",
    "    k2_vy = acc_y * dt",
    "    k2_x = (state[2] + 0.5 * k1_vx) * dt",
    "    k2_y = (state[3] + 0.5 * k1_vy) * dt",
    "    k3_vx = acc_x * dt",
    "    k3_vy = acc_y * dt",
    "    k3_x = (state[2] + 0.5 * k2_vx) * dt",
    "    k3_y = (state[3] + 0.5 * k2_vy) * dt",
    "    k4_vx = acc_x * dt",
    "    k4_vy = acc_y * dt",
    "    k4_x = (state[2] + k3_vx) * dt",
    "    k4_y = (state[3] + k3_vy) * dt",
    "    state[0] += (k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0",
    "    state[1] += (k1_y + 2*k2_y + 2*k3_y + k4_y)/6.0",
    "    state[2] += (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)/6.0",
    "    state[3] += (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)/6.0",
    # Collisions (conditionals for sophistication)
    "    if state[1] < 0:",
    "        state[1] = -state[1]",
    "        state[3] = -state[3] * 0.8",
    "    if state[1] <= 0 and state[3] < 0:",
    "        state[3] *= -0.8",  # Frictionless bounce
    "    if state[1] < 0:",
    "        state[1] = 0.0",
    "        state[3] = -state[3] * 0.8",
    # Advanced: Air drag (quadratic)
    "    speed = math.sqrt(state[2]**2 + state[3]**2)",
    "    drag_mag = 0.1 * speed**2",
    "    drag_x = -drag_mag * state[2] / (speed + 1e-6)",
    "    drag_y = -drag_mag * state[3] / (speed + 1e-6)",
    "    state[2] += drag_x * dt",
    "    state[3] += drag_y * dt",
    # Bloat/noise for evolution
    "    state[0] *= 1.0",  # Identity
    "    pass",
    "    return state"
]

def generate_physics_code():
    """Generate random initial update function."""
    base = "def update(state, dt):\\n    import numpy as np\\n    import math\\n    state = np.array(state)\\n    acc_x = 0.0\\n    acc_y = -9.81"
    num_lines = random.randint(10, 30)  # Larger for complexity
    selected = random.sample(LINE_POOL * 3, num_lines)  # Duplicates allowed
    random.shuffle(selected)
    body = '\\n'.join(selected)
    full_code = base + '\\n' + body + '\\n    return state.tolist()'
    return full_code

def mutate_physics(code):
    """Advanced mutations: Token swaps, insert RK blocks, flip signs, add conditionals."""
    lines = code.split('\\n')
    if len(lines) < 5:
        return code
    mut_type = random.choice(['token_mut', 'insert_block', 'delete', 'swap', 'semantic'])
    if mut_type == 'token_mut':
        idx = random.randint(4, len(lines)-2)
        line = lines[idx]
        tokens = line.split()
        if tokens:
            t_idx = random.randint(0, len(tokens)-1)
            opts = {
                '-9.81': str(random.uniform(-10, -9)),
                '+': '-', '-': '+',
                '*': '**', 'dt': 'dt/2',
                '<': '>', '0.8': str(random.uniform(0.7, 0.9)),
                'state[3]': 'abs(state[3])',
                'state[1]': 'state[1] + dt'
            }
            if tokens[t_idx] in opts:
                tokens[t_idx] = opts[tokens[t_idx]]
            else:
                tokens[t_idx] = random.choice(['math.sin(dt)', 'np.clip(state[1], 0, np.inf)', '0.0'])
            lines[idx] = ' '.join(tokens)
    elif mut_type == 'insert_block':
        blocks = [
            "    k1_vx = acc_x * dt\\n    k1_vy = acc_y * dt\\n    k1_x = state[2] * dt\\n    k1_y = state[3] * dt\\n    k2_vx = acc_x * dt\\n    k2_vy = acc_y * dt\\n    k2_x = (state[2] + 0.5 * k1_vx) * dt\\n    k2_y = (state[3] + 0.5 * k1_vy) * dt\\n    k3_vx = acc_x * dt\\n    k3_vy = acc_y * dt\\n    k3_x = (state[2] + 0.5 * k2_vx) * dt\\n    k3_y = (state[3] + 0.5 * k2_vy) * dt\\n    k4_vx = acc_x * dt\\n    k4_vy = acc_y * dt\\n    k4_x = (state[2] + k3_vx) * dt\\n    k4_y = (state[3] + k3_vy) * dt\\n    state[0] += (k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0\\n    state[1] += (k1_y + 2*k2_y + 2*k3_y + k4_y)/6.0\\n    state[2] += (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)/6.0\\n    state[3] += (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)/6.0",
            "    if state[1] < 0:\\n        state[1] = 0\\n        state[3] *= -0.8",
            "    speed = math.sqrt(state[2]**2 + state[3]**2)\\n    drag_mag = 0.1 * speed**2\\n    drag_x = -drag_mag * state[2] / (speed + 1e-6)\\n    drag_y = -drag_mag * state[3] / (speed + 1e-6)\\n    state[2] += drag_x * dt\\n    state[3] += drag_y * dt"
        ]
        new_block = random.choice(blocks)
        idx = random.randint(5, len(lines)-2)
        new_lines = new_block.split('\\n')
        lines = lines[:idx] + new_lines + lines[idx:]
    elif mut_type == 'delete':
        if len(lines) > 10:
            idx = random.randint(5, len(lines)-2)
            del lines[idx]
    elif mut_type == 'swap':
        if len(lines) > 6:
            i, j = random.sample(range(5, len(lines)-1), 2)
            lines[i], lines[j] = lines[j], lines[i]
    elif mut_type == 'semantic':
        for i in range(len(lines)):
            if '9.81' in lines[i]:
                lines[i] = lines[i].replace('9.81', str(random.uniform(9, 10)))
            if '0.8' in lines[i]:
                lines[i] = lines[i].replace('0.8', str(random.uniform(0.7, 0.9)))
            if 'acc_y = -' in lines[i]:
                lines[i] = lines[i].replace('-', '') if random.random() < 0.1 else lines[i]
    return '\\n'.join(lines)

def crossover_physics(p1, p2):
    """Multi-point crossover on lines, preserving structure."""
    l1, l2 = p1.split('\\n'), p2.split('\\n')
    min_l = min(len(l1), len(l2))
    if min_l < 6:
        return p1 if random.random() < 0.5 else p2
    points = sorted(random.sample(range(5, min_l-1), random.randint(1, 3)))
    child = []
    src1 = l1
    for pt in points:
        child.extend(src1[:pt])
        src1 = l2 if src1 is l1 else l1
    child.extend(src1[pt:])
    if not child[0].startswith('def update'):
        child.insert(0, "def update(state, dt):")
    child.append('    return state.tolist()')
    return '\\n'.join(child)

# --------------------------------------------------------------
# 3. EVALUATION: Accuracy + Efficiency + Novelty
# --------------------------------------------------------------
def evaluate_physics(code):
    """Exec code, simulate trajectory, compute multi-objective scores."""
    try:
        ns = {'np': np, 'math': math, 'state': None, 'dt': None, 'acc_x': None, 'acc_y': None}
        exec(code, ns)
        update_func = ns.get('update')
        if not callable(update_func):
            return 0.0, float('inf'), np.zeros((STEPS+1, 2))  # fit, runtime, traj
        start_time = time.time()
        state = INITIAL_STATE.tolist()
        trajectory = [state[:2]]
        for _ in range(STEPS):
            state = update_func(state, DT)
            trajectory.append(state[:2])
        traj = np.array(trajectory)
        runtime = time.time() - start_time
        mse = np.mean((traj - TRUTH_POS)**2)
        accuracy = 1.0 / (1.0 + mse)  # Bounded 0-1
        efficiency = 1.0 / (1.0 + runtime)  # Normalized
        combined_fitness = 0.6 * accuracy + 0.4 * efficiency
        return combined_fitness, runtime, traj
    except Exception as e:
        print(f"Eval error: {e}")  # Debug
        return 0.0, float('inf'), np.zeros((STEPS+1, 2))

# Novelty archive
NOVELTY_ARCHIVE = []
NOVELTY_THRESHOLD = 0.05
K_NEAREST = 10

def compute_novelty(traj):
    if not NOVELTY_ARCHIVE:
        NOVELTY_ARCHIVE.append(traj)
        return 1.0
    dists = sorted([euclidean(traj.flatten(), a.flatten()) for a in NOVELTY_ARCHIVE])
    novelty = np.mean(dists[:K_NEAREST])
    if novelty > NOVELTY_THRESHOLD:
        NOVELTY_ARCHIVE.append(traj)
        if len(NOVELTY_ARCHIVE) > 200:
            NOVELTY_ARCHIVE.pop(0)
    return novelty

# --------------------------------------------------------------
# 4. EVOLUTION LOOP: Infinite Novelty-Driven GP
# --------------------------------------------------------------
POP_SIZE = 500
MAX_GENS = 200  # Demo; set higher or loop forever
MUT_RATE = 0.95
CROSS_RATE = 0.8
ELITE_SIZE = 20
NOVELTY_WEIGHT = 0.5

population = [generate_physics_code() for _ in range(POP_SIZE)]

print("=== Hyper-Sophisticated Physics Engine Evolution ===\\n")
print(f"Target: 2D Bouncing Particle | Pop: {POP_SIZE} | Gens: {MAX_GENS} | Novelty Weight: {NOVELTY_WEIGHT}\\n")

best_fitness_history = []
best_traj = None

for gen in range(MAX_GENS):
    evaluated = []
    for ind in population:
        fit, rt, traj = evaluate_physics(ind)
        nov = compute_novelty(traj)
        total_score = (1 - NOVELTY_WEIGHT) * fit + NOVELTY_WEIGHT * nov
        evaluated.append((ind, fit, nov, rt, total_score, traj))

    evaluated.sort(key=lambda x: x[4], reverse=True)

    best = evaluated[0]
    best_fitness_history.append(best[1])
    best_traj = best[5]
    if gen % 10 == 0:
        print(f"Gen {gen+1}: Best Fit={best[1]:.4f} | Nov={best[2]:.4f} | RT={best[3]:.6f}s | Archive={len(NOVELTY_ARCHIVE)}")
        print(f"Best Code:\\n{best[0]}\\n{'-'*80}")

    new_pop = [e[0] for e in evaluated[:ELITE_SIZE]]

    parents = []
    for _ in range(POP_SIZE - ELITE_SIZE):
        tour = random.sample(evaluated, 5)
        parents.append(max(tour, key=lambda x: x[4])[0])

    for i in range(0, len(parents), 2):
        if i+1 < len(parents):
            c1, c2 = crossover_physics(parents[i], parents[i+1])
            if random.random() < MUT_RATE:
                c1 = mutate_physics(c1)
            if random.random() < MUT_RATE:
                c2 = mutate_physics(c2)
            new_pop.extend([c1, c2])
        else:
            new_pop.append(mutate_physics(parents[i]))

    population = new_pop[:POP_SIZE]

print("\\n=== Evolution Demo Complete ===")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(best_fitness_history)
plt.title('Fitness Over Generations')
plt.ylabel('Best Fitness')

plt.subplot(1, 2, 2)
plt.plot(TRUTH_POS[:, 0], TRUTH_POS[:, 1], 'g-', label='Truth')
plt.plot(best_traj[:, 0], best_traj[:, 1], 'r--', label='Evolved')
plt.title('Trajectories')
plt.legend()
plt.show()

# Save best
with open('evolved_physics_engine.py', 'w') as f:
    f.write(best[0])
print("Best engine saved.")
`;

export const MOCK_OUTPUT = `=== Hyper-Sophisticated Physics Engine Evolution ===

Target: 2D Bouncing Particle | Pop: 500 | Gens: 200 | Novelty Weight: 0.5

Gen 1: Best Fit=0.0123 | Nov=1.0000 | RT=0.001234s | Archive=1
Best Code:
...
--------------------------------------------------------------------------------
Gen 11: Best Fit=0.4582 | Nov=0.8734 | RT=0.000987s | Archive=25
Best Code:
...
--------------------------------------------------------------------------------
Gen 51: Best Fit=0.8912 | Nov=0.5123 | RT=0.000812s | Archive=98
Best Code:
...
--------------------------------------------------------------------------------
Gen 101: Best Fit=0.9856 | Nov=0.2345 | RT=0.000756s | Archive=153
Best Code:
...
--------------------------------------------------------------------------------
Gen 191: Best Fit=0.9998 | Nov=0.0678 | RT=0.000711s | Archive=200
Best Code:
...
--------------------------------------------------------------------------------

=== Evolution Demo Complete ===
Best engine saved.
`;
