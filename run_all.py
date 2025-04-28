# filepath: ga_dissertation/run_all.py
import os
import sys
import importlib
import time
from comparative_analysis import ResultsAnalyzer

def run_case_studies():
    """Run all case study scripts in the project"""
    # Find all case study scripts
    case_study_files = [f for f in os.listdir(".") 
    if f.startswith("case_study_") and f.endswith(".py")]
    
    if not case_study_files:
        print("No case study scripts found!")
        return
    
    print(f"Found {len(case_study_files)} case study scripts:")
    for i, script in enumerate(case_study_files):
        print(f"{i+1}. {script}")
    
    # Run each script
    for script in case_study_files:
        print(f"\n{'='*60}")
        print(f"Running {script}...")
        print(f"{'='*60}")
        
        # Get module name (remove .py extension)
        module_name = script[:-3]
        
        try:
            # Import and run the module
            importlib.import_module(module_name)
            print(f"Completed {script} successfully.")
        except Exception as e:
            print(f"Error running {script}: {e}")
    
    print("\nAll case studies completed.")

def run_comparative_analysis():
    """Run comparative analysis on all results"""
    print(f"\n{'='*60}")
    print("Running comparative analysis...")
    print(f"{'='*60}")
    
    # Check if results directory exists and has content
    if not os.path.exists("results"):
        print("No results directory found. Skipping comparative analysis.")
        return None
        
    case_studies = [d for d in os.listdir("results") 
    if os.path.isdir(os.path.join("results", d))]
    
    if not case_studies:
        print("No case study results found. Skipping comparative analysis.")
        return None
        
    # Check if at least one case study has parameter_sweep_results.csv
    has_results = False
    for case_study in case_studies:
        if os.path.exists(os.path.join("results", case_study, "parameter_sweep_results.csv")):
            has_results = True
            break
            
    if not has_results:
        print("No parameter sweep results found. Skipping comparative analysis.")
        return None
    
    analyzer = ResultsAnalyzer()
    best_configs = analyzer.run_all_analyses()
    
    return best_configs

def generate_summary_report(best_configs):
    """Generate a summary report of findings"""
    report_path = "summary_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Genetic Algorithm Dissertation - Summary of Results\n\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Best Parameter Configurations\n\n")
        f.write("| Case Study | Population Size | Crossover Prob | Mutation Prob | Best Fitness |\n")
        f.write("|------------|----------------|---------------|--------------|-------------|\n")
        
        for _, row in best_configs.iterrows():
            f.write(f"| {row['case_study']} | {int(row['population_size'])} | {row['crossover_prob']} | {row['mutation_prob']} | {row['best_fitness']:.6f} |\n")
        
        f.write("\n## Key Findings\n\n")
        f.write("- The analysis reveals that ...\n")
        f.write("- Parameter sensitivity varies across case studies ...\n")
        f.write("- Convergence rates show ...\n")
        f.write("\n(Complete these findings based on your actual results)\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("- Further investigation into ...\n")
        f.write("- More fine-grained parameter tuning for ...\n")
        f.write("- Application of findings to real-world problems ...\n")
    
    print(f"Summary report generated: {report_path}")

if __name__ == "__main__":
    # Run all case studies
    run_case_studies()
    
    # Run comparative analysis
    best_configs = run_comparative_analysis()
    
    # Generate summary report
    if best_configs is not None:
        generate_summary_report(best_configs)
    
    print("\nProject execution completed!")