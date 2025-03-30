import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ResultsAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.case_studies = [d for d in os.listdir(results_dir) 
                           if os.path.isdir(os.path.join(results_dir, d))]
        
        # Create output directory for comparison results
        self.comparison_dir = os.path.join("plots", "comparisons")
        os.makedirs(self.comparison_dir, exist_ok=True)
    
    def load_all_results(self):
        """Load parameter sweep results from all case studies"""
        all_results = {}
        
        for case_study in self.case_studies:
            file_path = os.path.join(self.results_dir, case_study, "parameter_sweep_results.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['case_study'] = case_study
                all_results[case_study] = df
            else:
                print(f"Warning: No results found for {case_study}")
        
        return all_results
    
    def compare_convergence(self):
        """Compare convergence rates across case studies"""
        plt.figure(figsize=(12, 8))
        
        for case_study in self.case_studies:
            # Get best parameter configuration for this case study
            param_results = pd.read_csv(os.path.join(self.results_dir, case_study, "parameter_sweep_results.csv"))
            
            # Find configuration with best fitness (assuming lower is better, use idxmax for maximization)
            best_config_idx = param_results['best_fitness'].idxmin()
            best_config = param_results.loc[best_config_idx]
            
            # Find corresponding stats file
            config_name = f"pop{int(best_config['population_size'])}_gen50_cxpb{best_config['crossover_prob']}_mutpb{best_config['mutation_prob']}"
            stats_file = os.path.join(self.results_dir, case_study, f"stats_{config_name}.csv")
            
            if os.path.exists(stats_file):
                stats = pd.read_csv(stats_file)
                
                # Plot convergence
                plt.plot(stats['generation'], stats['best_fitness'] if 'best_fitness' in stats else stats['max_fitness'],
                         label=f"{case_study}")
        
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Convergence Comparison Across Case Studies")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.comparison_dir, "convergence_comparison.png"))
        plt.close()
    
    def compare_parameter_sensitivity(self):
        """Compare parameter sensitivity across case studies"""
        all_results = self.load_all_results()
        
        if not all_results:
            print("No results to analyze")
            return
            
        # Create a combined dataframe
        combined_df = pd.concat([df for df in all_results.values()])
        
        # Plot effect of population size
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='case_study', y='best_fitness', hue='population_size', data=combined_df)
        plt.title("Effect of Population Size Across Case Studies")
        plt.xlabel("Case Study")
        plt.ylabel("Best Fitness")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, "population_size_comparison.png"))
        plt.close()
        
        # Plot effect of mutation probability
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='case_study', y='best_fitness', hue='mutation_prob', data=combined_df)
        plt.title("Effect of Mutation Probability Across Case Studies")
        plt.xlabel("Case Study")
        plt.ylabel("Best Fitness")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, "mutation_prob_comparison.png"))
        plt.close()
        
        # Plot effect of crossover probability
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='case_study', y='best_fitness', hue='crossover_prob', data=combined_df)
        plt.title("Effect of Crossover Probability Across Case Studies")
        plt.xlabel("Case Study")
        plt.ylabel("Best Fitness")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, "crossover_prob_comparison.png"))
        plt.close()
    
    def analyze_best_configurations(self):
        """Find and compare the best configurations for each case study"""
        all_results = self.load_all_results()
        
        if not all_results:
            print("No results to analyze")
            return
            
        best_configs = []
        
        for case_study, df in all_results.items():
            # Find the best configuration (lowest fitness value for minimization, highest for maximization)
            best_idx = df['best_fitness'].idxmin()  # Use idxmax() for maximization
            best_row = df.loc[best_idx].copy()
            best_row['case_study'] = case_study
            best_configs.append(best_row)
        
        # Combine into a single dataframe
        best_df = pd.DataFrame(best_configs)
        
        # Save to CSV
        best_df.to_csv(os.path.join(self.comparison_dir, "best_configurations.csv"), index=False)
        
        # Create summary visualizations
        plt.figure(figsize=(12, 6))
        
        # Bar chart of best fitness values
        plt.subplot(1, 2, 1)
        sns.barplot(x='case_study', y='best_fitness', data=best_df)
        plt.title("Best Fitness Achieved per Case Study")
        plt.xlabel("Case Study")
        plt.ylabel("Best Fitness")
        plt.xticks(rotation=45)
        
        # Heatmap of best parameters
        plt.subplot(1, 2, 2)
        param_cols = ['population_size', 'crossover_prob', 'mutation_prob']
        heatmap_data = best_df.set_index('case_study')[param_cols]
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Best Parameters per Case Study")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, "best_configurations_summary.png"))
        plt.close()
        
        return best_df
    
    def run_all_analyses(self):
        """Run all comparative analyses"""
        print("Comparing convergence rates...")
        self.compare_convergence()
        
        print("Analyzing parameter sensitivity...")
        self.compare_parameter_sensitivity()
        
        print("Finding best configurations...")
        best_configs = self.analyze_best_configurations()
        
        print("Analysis complete! Results saved to:", self.comparison_dir)
        return best_configs

# Example usage
if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    best_configs = analyzer.run_all_analyses()
    print("\nBest configurations for each case study:")
    print(best_configs)