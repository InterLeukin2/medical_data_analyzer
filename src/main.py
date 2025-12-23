"""
Main Entry Point
===============

This is the main entry point for the medical data analyzer application that supports
CSV, Excel, JSON, and JSONL input formats.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from analyzer import MedicalDataAnalyzer
from visualizer import DataVisualizer
from report_generator import ReportGenerator


def main():
    """
    Main function to run the medical data analyzer
    """
    parser = argparse.ArgumentParser(description='Medical Data Analyzer')
    parser.add_argument('input_file', help='Input data file (CSV, Excel, JSON, or JSONL)')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--viz-dir', default='visualizations', help='Visualization directory')
    parser.add_argument('--report-dir', default='reports', help='Report directory')
    parser.add_argument('--language', choices=['en', 'zh'], default='en', 
                        help='Language for reports and visualizations')
    parser.add_argument('--no-charts', action='store_true', 
                        help='Skip chart generation')
    parser.add_argument('--no-report', action='store_true', 
                        help='Skip report generation')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    print(f"Analyzing medical data from: {args.input_file}")
    print(f"Output language: {'English' if args.language == 'en' else 'Chinese'}")
    
    try:
        # Initialize analyzer
        analyzer = MedicalDataAnalyzer(
            args.input_file,
            config={
                'output_dir': args.output_dir,
                'visualization_dir': args.viz_dir,
                'report_dir': args.report_dir
            }
        )
        
        # Load and process data
        analyzer.load_data()
        
        # If there's a JSON column, extract it
        if 'response' in analyzer.df.columns:
            print("Extracting JSON fields...")
            analyzer.extract_json_fields()
        
        # Remove duplicates
        print("Removing duplicates...")
        analyzer.remove_duplicates()
        
        # Translate difficulty sources if using English
        if args.language == 'en':
            print("Translating difficulty sources...")
            analyzer.translate_difficulty_sources()
            df_to_use = analyzer.translated_df
        else:
            df_to_use = analyzer.deduplicated_df
        
        # Create visualizations
        if not args.no_charts:
            print("Generating visualizations...")
            visualizer = DataVisualizer(df_to_use, args.viz_dir)
            visualizer.create_all_visualizations(args.language)
        
        # Generate report
        if not args.no_report:
            print("Generating report...")
            report_generator = ReportGenerator(df_to_use, args.viz_dir, args.report_dir)
            report_generator.generate_pdf_report(args.language, not args.no_charts)
        
        # Generate summary
        print("Generating analysis summary...")
        summary = analyzer.get_analysis_summary(df_to_use)
        summary_file = analyzer.save_summary(summary)
        
        print("\nAnalysis completed successfully!")
        print(f"Summary: {summary_file}")
        if not args.no_charts:
            print(f"Visualizations saved to: {args.viz_dir}")
        if not args.no_report:
            print(f"Reports saved to: {args.report_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()