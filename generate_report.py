# filepath: ga_dissertation/generate_report.py
import pandas as pd
import matplotlib.pyplot as plt
from comparative_analysis import ResultsAnalyzer

def generate_final_report():
    """Generate a comprehensive final report with all results"""
    analyzer = ResultsAnalyzer()
    best_configs = analyzer.analyze_best_configurations()
    
    # Create report
    with open("final_report.md", "w") as f:
        f.write("# Genetic Algorithm Optimization Results\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of applying genetic algorithms to optimize ")
        f.write("multiple case studies. The analysis focused on parameter tuning and comparative performance.\n\n")
        
        # Best Configurations
        f.write("## Optimal Parameter Configurations\n\n")
        f.write(best_configs.to_markdown(index=False))
        f.write("\n\n")
        
        # Key Insights
        f.write("## Key Insights\n\n")
        f.write("1. **Parameter Sensitivity**: [Complete with actual findings]\n")
        f.write("2. **Convergence Analysis**: [Complete with actual findings]\n")
        f.write("3. **Problem Characteristics**: [Complete with actual findings]\n\n")
        
        # Case Study Specific Results
        f.write("## Results by Case Study\n\n")
        for case_study in analyzer.case_studies:
            f.write(f"### {case_study}\n\n")
            f.write("- Optimal solution achieved: [Details from results]\n")
            f.write("- Convergence behavior: [Observations from plots]\n")
            f.write("- Parameter sensitivity: [Observations from parameter sweep]\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The genetic algorithm performance across different case studies shows ")
        f.write("[summary of overall findings]. The optimal parameters vary by problem, ")
        f.write("indicating the importance of parameter tuning for specific problem domains.\n\n")
        
        # References
        f.write("## References\n\n")
        f.write("1. DEAP documentation: https://deap.readthedocs.io/\n")
        f.write("2. [Add your reference sources]\n")

if __name__ == "__main__":
    generate_final_report()