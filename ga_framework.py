# filepath: ga_dissertation/ga_framework.py
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import pandas as pd
import os


class GeneticAlgorithm:
    def __init__(self, problem_name, is_maximization=True):
        """Initialize GA with problem name and optimization direction"""
        self.problem_name = problem_name

        # Reset creator to avoid conflicts if multiple GAs are created
        if hasattr(creator, "FitnessFunc"):
            del creator.FitnessFunc
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Create fitness and individual classes
        if is_maximization:
            creator.create("FitnessFunc", base.Fitness, weights=(1.0,))
        else:
            creator.create("FitnessFunc", base.Fitness, weights=(-1.0,))

        creator.create("Individual", list, fitness=creator.FitnessFunc)

        # Initialize toolbox
        self.toolbox = base.Toolbox()

        # Create directory for results if it doesn't exist
        self.results_dir = os.path.join("results", problem_name)
        os.makedirs(self.results_dir, exist_ok=True)

        self.plots_dir = os.path.join("plots", problem_name)
        os.makedirs(self.plots_dir, exist_ok=True)

    def setup_ga(
        self,
        evaluate_function,
        n_vars,
        var_bounds,
        mutation_sigma=1.0,
        mutation_prob=0.2,
        crossover_alpha=0.5,
    ):
        """Set up the genetic algorithm components"""

        # Define genes and individuals
        if isinstance(var_bounds[0], (list, tuple)):
            # If different bounds for each variable
            self.toolbox.register(
                "attr_var", self._generate_var_with_bounds, var_bounds
            )
            self.toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                self.toolbox.attr_var,
                n=n_vars,
            )
        else:
            # Same bounds for all variables
            self.toolbox.register(
                "attr_float", random.uniform, var_bounds[0], var_bounds[1]
            )
            self.toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                self.toolbox.attr_float,
                n=n_vars,
            )

        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Register operators
        self.toolbox.register("evaluate", evaluate_function)
        self.toolbox.register("mate", tools.cxBlend, alpha=crossover_alpha)
        self.toolbox.register(
            "mutate", tools.mutGaussian, mu=0, sigma=mutation_sigma, indpb=mutation_prob
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        return self.toolbox

    def _generate_var_with_bounds(self, bounds):
        """Generate a variable within its specific bounds"""
        return random.uniform(bounds[0], bounds[1])

    def run_ga(
        self,
        n_generations=50,
        population_size=100,
        crossover_prob=0.7,
        mutation_prob=0.2,
    ):
        """Run the genetic algorithm and return statistics"""

        population = self.toolbox.population(n=population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", min)
        stats.register("max", max)

        # Hall of fame to keep track of best individuals
        hof = tools.HallOfFame(10)

        # Run algorithm
        pop, log = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=n_generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        # Save results
        self._save_results(
            pop,
            log,
            hof,
            f"pop{population_size}_gen{n_generations}_cxpb{crossover_prob}_mutpb{mutation_prob}",
        )

        return pop, log, hof

    def _save_results(self, population, logbook, halloffame, config_name):
        """Save results to files"""
        # Save statistics
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_maxs = logbook.select("max")
        fit_avgs = logbook.select("avg")
        fit_stds = logbook.select("std")

        df = pd.DataFrame(
            {
                "generation": gen,
                "min_fitness": fit_mins,
                "max_fitness": fit_maxs,
                "avg_fitness": fit_avgs,
                "std_fitness": fit_stds,
            }
        )

        df.to_csv(
            os.path.join(self.results_dir, f"stats_{config_name}.csv"), index=False
        )

        # Save best individuals
        best_ind_df = pd.DataFrame([list(ind) for ind in halloffame])
        best_ind_df.to_csv(
            os.path.join(self.results_dir, f"best_individuals_{config_name}.csv"),
            index=False,
        )

        # Plot fitness evolution
        self._plot_fitness_evolution(gen, fit_mins, fit_maxs, fit_avgs, config_name)

    def _plot_fitness_evolution(
        self, generations, min_fitness, max_fitness, avg_fitness, config_name
    ):
        """Create and save plots of fitness evolution"""
        plt.figure(figsize=(10, 6))
        plt.plot(generations, min_fitness, "g-", label="Minimum Fitness")
        plt.plot(generations, max_fitness, "r-", label="Maximum Fitness")
        plt.plot(generations, avg_fitness, "b-", label="Average Fitness")
        plt.title(f"Fitness Evolution - {self.problem_name} ({config_name})")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(
            os.path.join(self.plots_dir, f"fitness_evolution_{config_name}.png")
        )
        plt.close()

    def parameter_sweep(self, parameter_grid, n_generations=50):
        """Run GA with different parameter configurations and compare results"""
        results = []

        # Parameter sweep
        for pop_size in parameter_grid.get("population_size", [100]):
            for crossover_prob in parameter_grid.get("crossover_prob", [0.7]):
                for mutation_prob in parameter_grid.get("mutation_prob", [0.2]):
                    # Run GA with these parameters
                    print(
                        f"\nRunning GA with: pop_size={pop_size}, cx_prob={crossover_prob}, mut_prob={mutation_prob}"
                    )

                    pop, log, hof = self.run_ga(
                        n_generations=n_generations,
                        population_size=pop_size,
                        crossover_prob=crossover_prob,
                        mutation_prob=mutation_prob,
                    )

                    # Store results
                    best_fitness = hof[0].fitness.values[0]
                    avg_fitness = log.select("avg")[-1]
                    std_fitness = log.select("std")[-1]

                    results.append(
                        {
                            "population_size": pop_size,
                            "crossover_prob": crossover_prob,
                            "mutation_prob": mutation_prob,
                            "best_fitness": best_fitness,
                            "avg_fitness": avg_fitness,
                            "std_fitness": std_fitness,
                            "best_solution": list(hof[0]),
                        }
                    )

        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            os.path.join(self.results_dir, "parameter_sweep_results.csv"), index=False
        )

        # Create comparison plots
        self._plot_parameter_comparisons(results_df)

        return results_df

    def _plot_parameter_comparisons(self, results_df):
        """Create plots comparing different parameter settings"""
        # Plot best fitness vs population size
        plt.figure(figsize=(12, 8))
        for cr in results_df["crossover_prob"].unique():
            subset = results_df[results_df["crossover_prob"] == cr]
            plt.plot(
                subset["population_size"],
                subset["best_fitness"],
                marker="o",
                label=f"Crossover prob: {cr}",
            )

        plt.xlabel("Population Size")
        plt.ylabel("Best Fitness")
        plt.title(f"{self.problem_name}: Effect of Population Size on Best Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "population_size_effect.png"))
        plt.close()

        # Plot best fitness vs mutation probability
        plt.figure(figsize=(12, 8))
        for ps in results_df["population_size"].unique():
            subset = results_df[results_df["population_size"] == ps]
            plt.plot(
                subset["mutation_prob"],
                subset["best_fitness"],
                marker="o",
                label=f"Population size: {ps}",
            )

        plt.xlabel("Mutation Probability")
        plt.ylabel("Best Fitness")
        plt.title(
            f"{self.problem_name}: Effect of Mutation Probability on Best Fitness"
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "mutation_prob_effect.png"))
        plt.close()
