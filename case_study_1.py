# filepath: ga_dissertation/case_study_1.py
from ga_framework import GeneticAlgorithm
from data_processor import DataProcessor
import numpy as np

# 1. Define the case study
CASE_STUDY = "CaseStudy1"

# 2. Set up data processor
data_processor = DataProcessor("case_studies")

# 3. Load and process data 
data = data_processor.load_case_study_data("Argo1.csv")
# Extract only numeric columns and exclude the first column (Name)
numeric_data = data.select_dtypes(include=['number'])
data_processed = numeric_data.drop(['BUGS'], axis=1, errors='ignore')  # Remove target variable from features

# 4. Define problem parameters based on case study
N_VARIABLES = data_processed.shape[1]  # Number of numeric features
# Binary representation (0 = exclude feature, 1 = include feature)
VARIABLE_BOUNDS = (0, 1)  

# 5. Define the fitness function
def fitness_function(individual):
    """
    Fitness function for software bug prediction case study.
    Uses selected features (based on genetic algorithm) to predict bugs.
    
    The individual is an array of binary values indicating which features to use.
    """
    # Select features based on the individual's genes (1 = include feature, 0 = exclude)
    selected_indices = [i for i, gene in enumerate(individual) if gene > 0.5]
    
    if not selected_indices:  # Ensure at least one feature is selected
        return (float('inf'),)  # Return a poor fitness score
        
    feature_names = data_processed.columns[selected_indices]
        
    # Extract selected features from your dataset
    X = data_processed[feature_names]  # Selected features
    y = numeric_data['BUGS']  # Target variable
    
    # Simple train/test split
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Return error (to be minimized)
    return (mse,)  # Note the comma to return a tuple

# 6. Initialize Genetic Algorithm
ga = GeneticAlgorithm(CASE_STUDY, is_maximization=False)  # False for minimization

# 7. Setup GA parameters
ga.setup_ga(
    evaluate_function=fitness_function,
    n_vars=N_VARIABLES,
    var_bounds=VARIABLE_BOUNDS,
    mutation_sigma=0.5,
    mutation_prob=0.2,
    crossover_alpha=0.5
)

# 8. Define parameter grid for experiments
parameter_grid = {
    'population_size': [50, 100, 200],
    'crossover_prob': [0.5, 0.7, 0.9],
    'mutation_prob': [0.1, 0.2, 0.3]
}

# 9. Run parameter sweep
print(f"Running parameter sweep for {CASE_STUDY}...")
results = ga.parameter_sweep(parameter_grid, n_generations=50)

# 10. Print best results
best_row = results.loc[results['best_fitness'].idxmin()]  # Use idxmax() for maximization
print("\nBest Parameters:")
print(f"Population Size: {best_row['population_size']}")
print(f"Crossover Probability: {best_row['crossover_prob']}")
print(f"Mutation Probability: {best_row['mutation_prob']}")
print(f"Best Fitness: {best_row['best_fitness']}")
print(f"Best Solution: {best_row['best_solution']}")

# 11. Run one final time with best parameters
print("\nRunning GA with best parameters...")
pop, log, hof = ga.run_ga(
    n_generations=100,  # More generations for final run
    population_size=int(best_row['population_size']),
    crossover_prob=best_row['crossover_prob'],
    mutation_prob=best_row['mutation_prob']
)

print("\nOptimization complete!")
print(f"Best solution: {hof[0]}")
print(f"Best fitness: {hof[0].fitness.values[0]}")