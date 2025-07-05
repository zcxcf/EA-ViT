import csv
import os
import random

from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_args_parser
from dataloader.image_datasets import build_image_dataset
from models.model_stage2 import EAViTStage2, ModifiedBlock


random.seed(42)

args = get_args_parser()

dataset_train, dataset_val, nb_classes = build_image_dataset(args)
trainDataLoader = DataLoader(dataset_train, args.batch_size, shuffle=True, num_workers=args.num_workers)

model = EAViTStage2(embed_dim=768, depth=12, mlp_ratio=4, num_heads=12, num_classes=nb_classes,
                    drop_path_rate=args.drop_path, qkv_bias=True, block=ModifiedBlock)
device = args.device

stage1_checkpoint_path = args.stage1_checkpoint_path
checkpoint = torch.load(stage1_checkpoint_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()

NUM_GENES = 12 + 12 * 8 + 12 * 12 + 12 + 12  # Number of binary genes in the vector (representing network structure)
GENERATIONS = 301  # Number of generations
MUTATION_PROBABILITY = 0.3  # Probability of mutation
CROSSOVER_PROBABILITY = 0.95  # Probability of crossover
POPULATION_SIZE = 5  # Population size

creator.create("FitnessMulti", base.Fitness, weights=(-0.5, 1.0))  # Minimize MACs, maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMulti)


def load_population_from_csv(csv_path, pop_size):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("CSV is empty")

    gen_counts = {}
    for r in rows:
        g = int(r['Generation'])
        gen_counts[g] = gen_counts.get(g, 0) + 1

    gens_sorted = sorted(gen_counts.keys())
    last_gen = gens_sorted[-1]

    if gen_counts[last_gen] < pop_size:
        last_gen -= 1
        if last_gen not in gen_counts or gen_counts[last_gen] < pop_size:
            raise ValueError("can not restore")

    target_rows = [r for r in rows if int(r['Generation']) == last_gen][:pop_size]

    pop = []
    for r in target_rows:
        encoding = list(map(int, r['Encoding']))  # '01010...' → [0,1,0,1,0]
        ind = creator.Individual(encoding)
        macs = float(r['MACs'])
        acc = float(r['Accuracy'])
        ind.fitness.values = (macs, acc)
        pop.append(ind)

    start_gen = last_gen + 1
    print(f"restore {len(pop)} population from Generation {last_gen}，")
    return pop, start_gen

def create_individual():
    r = random.randint(0, 1)
    if r > 0.5:
        return [1 for _ in range(NUM_GENES)]
    else:
        return [0 for _ in range(NUM_GENES)]

def create_individual_random():
    return [random.randint(0, 1) for _ in range(NUM_GENES)]

def evaluate(vector):
    latency = torch.rand(1).to(device)

    model.configure_latency(latency=latency, tau=1)
    vector[0] = 1

    embed_sum = int(sum(vector[:12]))
    embed_mask = torch.tensor([1] * embed_sum + [0] * (12 - embed_sum)).to(device)

    depth_attn_mask = torch.tensor(vector[12:24]).to(device)
    depth_mlp_mask = torch.tensor(vector[24:36]).to(device)

    mask_attn = []
    mask_mlp = []

    for i in range(12):
        attn_sum = int(sum(vector[36 + i * 12: 36 + (i + 1) * 12]))
        mask_attn.append(torch.tensor([1] * attn_sum + [0] * (12 - attn_sum)).to(device))

    for i in range(12):
        mlp_sum = int(sum(vector[180 + i * 8: 180 + (i + 1) * 8]))
        mask_mlp.append(torch.tensor([1] * mlp_sum + [0] * (8 - mlp_sum)).to(device))

    model.set_mask(embed_mask, mask_attn, mask_mlp, depth_attn_mask, depth_mlp_mask)

    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (img, label) in enumerate(trainDataLoader):
            img = img.to(device)
            label = label.to(device)

            preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

            _, predicted = torch.max(preds, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            break

        accuracy = correct / total
        # print("macs", total_macs.item())
        # print("val acc", accuracy)

    return (total_macs.item(), accuracy)


def plot_pareto_front(population, generation):
    plt.figure(figsize=(10, 6))

    macs = [ind.fitness.values[0] for ind in population]
    acc = [ind.fitness.values[1] for ind in population]

    front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    f_macs = [ind.fitness.values[0] for ind in front]
    f_acc = [ind.fitness.values[1] for ind in front]

    plt.scatter(macs, acc, c='blue', alpha=0.5, label='Population')
    plt.scatter(f_macs, f_acc, c='red', marker='x', label='Pareto Front')

    plt.xlabel("MACs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"Pareto Front @ Generation {generation}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

def save_population(population, generation, file_path="population.csv"):
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:

            writer.writerow(['Generation', 'MACs', 'Accuracy', 'Encoding', 'IsPareto'])

        for ind in population:
            macs, acc = ind.fitness.values
            encoding = ''.join(map(str, ind))
            is_pareto = ind in pareto_front
            writer.writerow([generation, macs, acc, encoding, int(is_pareto)])

def assign_macs_global_crowding(population):
    N = len(population)
    macs_vals = np.array([ind.fitness.values[0] for ind in population])

    idx = np.argsort(macs_vals)
    vmin, vmax = macs_vals[idx[0]], macs_vals[idx[-1]]

    cds = np.zeros(N)

    if vmax > vmin:
        cds[idx[0]]  = (macs_vals[idx[1]] - vmin) / (vmax - vmin)
        cds[idx[-1]] = (vmax - macs_vals[idx[-2]]) / (vmax - vmin)
    for k in range(1, N-1):
        i = idx[k]
        cds[i] = ((macs_vals[idx[k+1]] - macs_vals[idx[k-1]])
                  / (vmax - vmin + 1e-12))

    for ind, cd in zip(population, cds):
        ind.macs_crowding = cd

def select_by_partition_incremental(
        pop, toolbox, cxpb, mutpb,
        quotas, bins, max_iters, pop_size, min_mac_diff=0.001):
    combined = pop
    for it in range(max_iters + 1):

        source = pop if it == 0 else combined
        offspring = algorithms.varAnd(source, toolbox, cxpb, mutpb)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        combined = combined + offspring if it > 0 else pop + offspring


        assign_macs_global_crowding(combined)

        fronts = tools.sortNondominated(combined, len(combined))

        selected = []
        for i, quota in enumerate(quotas):
            low, high = bins[i], bins[i+1]
            seen_codes = set()
            sel_bin = []

            for front in fronts:
                cands = [ind for ind in front
                         if (i < len(quotas)-1 and low <= ind.fitness.values[0] < high)
                         or (i == len(quotas)-1 and low <= ind.fitness.values[0] <= high)]
                if not cands:
                    continue

                U = sel_bin + cands
                assign_macs_global_crowding(U)

                cands_sorted = sorted(
                    cands,
                    key=lambda ind: ind.macs_crowding,
                    reverse=True
                )

                for ind in cands_sorted:
                    code = ''.join(map(str, ind))
                    mac   = ind.fitness.values[0]
                    if code in seen_codes:
                        continue
                    if any(abs(mac - s.fitness.values[0]) < min_mac_diff for s in sel_bin):
                        continue
                    seen_codes.add(code)
                    sel_bin.append(ind)
                    if len(sel_bin) >= quota:
                        break
                if len(sel_bin) >= quota:
                    break

            selected.extend(sel_bin)

        if len(selected) >= pop_size:
            return selected[:pop_size]

    final = selected[:]
    need = pop_size - len(final)
    if need > 0:
        immigrants = [toolbox.individual_random() for _ in range(need)]
        fits = toolbox.map(toolbox.evaluate, immigrants)
        for ind, fit in zip(immigrants, fits):
            ind.fitness.values = fit
        final.extend(immigrants)
    return final

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("individual_random", tools.initIterate, creator.Individual, create_individual_random)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_PROBABILITY)
toolbox.register("evaluate", evaluate)


def main():
    csv_path = args.nsga_path

    if os.path.isfile(csv_path):
        try:
            pop, start_gen = load_population_from_csv(csv_path, POPULATION_SIZE)
        except Exception as e:
            print("restore failure：", e)
            pop = toolbox.population(n=POPULATION_SIZE)
            start_gen = 0
    else:
        pop = toolbox.population(n=POPULATION_SIZE)
        start_gen = 0

    unevaluated = [ind for ind in pop if not ind.fitness.valid]
    if unevaluated:
        fits = toolbox.map(toolbox.evaluate, unevaluated)
        for ind, fit in zip(unevaluated, fits):
            ind.fitness.values = fit

    hof = tools.ParetoFront()

    fitnesses = toolbox.map(toolbox.evaluate, pop)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in tqdm(range(start_gen, GENERATIONS)):
        random.shuffle(pop)

        pop = select_by_partition_incremental(
            pop, toolbox,
            cxpb=CROSSOVER_PROBABILITY,
            mutpb=MUTATION_PROBABILITY,
            quotas=[2, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4, 4],
            bins=np.linspace(0.0, 1.0, 21),
            max_iters=3,
            pop_size=POPULATION_SIZE,
            min_mac_diff=0.002
        )

        hof.update(pop)

        save_population(pop, gen, file_path=csv_path)


if __name__ == "__main__":
    main()



