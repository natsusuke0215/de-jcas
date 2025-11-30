import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    L, W = x
    if L < 0.1 or W < 0.1:
        return 1e9
    H = 1000 / (L * W)
    return 2 * (L*W + L*H + W*H)

def de_variant(strategy, pop_size=20, dim=2, F=0.5, CR=0.7,
               max_gen=20, bounds=(1, 50)):

    lower, upper = bounds
    pop = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([cost_function(x) for x in pop])
    convergence = []

    for gen in range(max_gen):
        for i in range(pop_size):
            idx = list(range(pop_size))
            idx.remove(i)
            r = np.random.choice(idx, 5, replace=False)
            if strategy == "rand/1":
                v = pop[r[0]] + F * (pop[r[1]] - pop[r[2]])

            elif strategy == "best/1":
                best = pop[np.argmin(fitness)]
                v = best + F * (pop[r[0]] - pop[r[1]])

            elif strategy == "rand/2":
                v = pop[r[0]] + F*((pop[r[1]] - pop[r[2]]) + (pop[r[3]] - pop[r[4]]))

            elif strategy == "best/2":
                best = pop[np.argmin(fitness)]
                v = best + F*((pop[r[0]] - pop[r[1]]) + (pop[r[2]] - pop[r[3]]))

            elif strategy == "current-to-best/1":
                best = pop[np.argmin(fitness)]
                v = pop[i] + F*(best - pop[i]) + F*(pop[r[0]] - pop[r[1]])

            else:
                raise ValueError("Invalid strategy")

            u = pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            u = np.clip(u, lower, upper)

            trial_f = cost_function(u)
            if trial_f < fitness[i]:
                pop[i] = u
                fitness[i] = trial_f

        convergence.append(np.min(fitness))

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx], convergence

def jde(pop_size=20, dim=2, max_gen=20, bounds=(1,50),
        tau1=0.1, tau2=0.1):

    lower, upper = bounds
    pop = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([cost_function(x) for x in pop])

    F_list  = np.random.uniform(0.1, 0.9, pop_size)
    CR_list = np.random.uniform(0.0, 1.0, pop_size)

    convergence = []

    for gen in range(max_gen):
        for i in range(pop_size):

            F_i = F_list[i]
            CR_i = CR_list[i]
            
            if np.random.rand() < tau1:
                F_i = 0.1 + 0.9 * np.random.rand()
            if np.random.rand() < tau2:
                CR_i = np.random.rand()

            idx = list(range(pop_size))
            idx.remove(i)
            r1, r2, r3 = np.random.choice(idx, 3, replace=False)

            v = pop[r1] + F_i * (pop[r2] - pop[r3])

            u = pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR_i or j == j_rand:
                    u[j] = v[j]

            u = np.clip(u, lower, upper)

            trial_f = cost_function(u)
            if trial_f < fitness[i]:
                pop[i] = u
                fitness[i] = trial_f
                F_list[i] = F_i
                CR_list[i] = CR_i

        convergence.append(np.min(fitness))

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx], convergence


strategies = ["rand/1", "best/1", "rand/2", "best/2", "current-to-best/1"]
results = {}

print(f"\n{'Strategy':<20} | {'Best Cost':<12} | {'Best Solution (L, W)'}")
print("-" * 65)

for s in strategies:
    best_sol, best_val, conv = de_variant(s)
    results[s] = conv
    print(f"{s:<20} | {best_val:<12.4f} | {best_sol}")

best_sol, best_val, jde_conv = jde()
results["jDE"] = jde_conv
print(f"{'jDE':<20} | {best_val:<12.4f} | {best_sol}")

plt.figure(figsize=(10, 6)) 

for name, conv in results.items():
    plt.plot(conv, label=name, linewidth=1.5)

plt.xlabel("Generation", fontsize=12)
plt.ylabel("Best Cost", fontsize=12)
plt.title("Convergence Curves of DE Variants + jDE", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('convergence_plot.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')

plt.show()
