"""
Medical Data Analyzer
=====================

A comprehensive tool for analyzing medical data with support for multiple formats
and customizable analysis parameters.

Features:
- Extract data from various file formats (CSV, Excel, JSON, JSONL)
- Perform statistical analysis on medical data
- Generate visualizations (charts, graphs)
- Create comprehensive PDF reports
- Support for multilingual reports (English, Chinese)
- Deduplication of data based on unique identifiers
"""

import pandas as pd
import json
import os
import sys
from typing import Dict, List, Optional, Union
from datetime import datetime


class MedicalDataAnalyzer:
    """
    Main class for analyzing medical data
    """
    
    def __init__(self, data_file: str, config: Optional[Dict] = None):
        """
        Initialize the analyzer with a data file and optional configuration
        
        Args:
            data_file (str): Path to the input data file
            config (dict, optional): Configuration dictionary
        """
        self.data_file = data_file
        self.config = config or {}
        self.df = None
        self.deduplicated_df = None
        self.translated_df = None
        
        # Default configuration
        self.default_config = {
            'unique_id_column': 'uniq_vid',
            'difficulty_source_column': 'primary_difficulty_source',
            'level_column': 'final_level',
            'score_suffix': '_score',
            'output_dir': 'output',
            'visualization_dir': 'visualizations',
            'report_dir': 'reports',
            'translations': {
                '信息质量': 'Information Quality',
                '混合': 'Mixed',
                '诊断挑战': 'Diagnostic Challenge',
                '治疗决策': 'Treatment Decision'
            }
        }
        
        # Update default config with user config
        self.default_config.update(self.config)
        self.config = self.default_config
        
        # Create output directories
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['visualization_dir'], exist_ok=True)
        os.makedirs(self.config['report_dir'], exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the specified file
        
        Returns:
            pd.DataFrame: Loaded data
        """
        if self.data_file.endswith('.csv'):
            self.df = pd.read_csv(self.data_file)
        elif self.data_file.endswith('.xlsx') or self.data_file.endswith('.xls'):
            self.df = pd.read_excel(self.data_file)
        elif self.data_file.endswith('.json'):
            self.df = pd.read_json(self.data_file)
        elif self.data_file.endswith('.jsonl'):
            self.df = pd.read_json(self.data_file, lines=True)
        else:
            raise ValueError("Unsupported file format. Please provide CSV, Excel, JSON, or JSONL file.")
        
        print(f"Loaded {len(self.df)} records from {self.data_file}")
        return self.df
    
    def extract_json_fields(self, json_column: str = 'response') -> pd.DataFrame:
        """
        Extract fields from JSON column and flatten the data
        
        Args:
            json_column (str): Name of the column containing JSON data
            
        Returns:
            pd.DataFrame: DataFrame with extracted and flattened data
        """
        if self.df is None:
            self.load_data()
        
        extracted_data = []
        
        for idx, row in self.df.iterrows():
            try:
                # Parse JSON data
                json_data = json.loads(row[json_column])
                
                # Flatten the JSON and add to list
                flat_data = self._flatten_json(json_data)
                flat_data['id'] = row.get('id', idx)
                extracted_data.append(flat_data)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not parse JSON in row {idx}")
                continue
        
        self.df = pd.DataFrame(extracted_data)
        output_file = os.path.join(self.config['output_dir'], 'extracted_data.csv')
        self.df.to_csv(output_file, index=False)
        print(f"Extracted data saved to {output_file}")
        return self.df
    
    def _flatten_json(self, data: dict, parent_key: str = '', sep: str = '_') -> dict:
        """
        Flatten nested JSON data
        
        Args:
            data (dict): JSON data to flatten
            parent_key (str): Parent key for recursion
            sep (str): Separator for nested keys
            
        Returns:
            dict: Flattened dictionary
        """
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_json(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate records based on unique identifier
        
        Returns:
            pd.DataFrame: Deduplicated DataFrame
        """
        if self.df is None:
            self.load_data()
        
        unique_id_col = self.config['unique_id_column']
        if unique_id_col not in self.df.columns:
            print(f"Warning: Column '{unique_id_col}' not found. Skipping deduplication.")
            self.deduplicated_df = self.df.copy()
            return self.deduplicated_df
        
        original_count = len(self.df)
        self.deduplicated_df = self.df.drop_duplicates(subset=[unique_id_col], keep='first')
        deduplicated_count = len(self.deduplicated_df)
        
        print(f"Removed {original_count - deduplicated_count} duplicate records")
        print(f"Remaining records: {deduplicated_count}")
        
        # Save deduplicated data
        output_file = os.path.join(self.config['output_dir'], 'deduplicated_data.csv')
        self.deduplicated_df.to_csv(output_file, index=False)
        print(f"Deduplicated data saved to {output_file}")
        
        return self.deduplicated_df
    
    def translate_difficulty_sources(self) -> pd.DataFrame:
        """
        Translate difficulty source values to English
        
        Returns:
            pd.DataFrame: DataFrame with translated difficulty sources
        """
        if self.deduplicated_df is None:
            self.remove_duplicates()
        
        self.translated_df = self.deduplicated_df.copy()
        difficulty_col = self.config['difficulty_source_column']
        
        if difficulty_col in self.translated_df.columns:
            translations = self.config['translations']
            self.translated_df[difficulty_col] = self.translated_df[difficulty_col].map(
                lambda x: translations.get(x, x) if pd.notnull(x) else x
            )
            
            # Save translated data
            output_file = os.path.join(self.config['output_dir'], 'translated_data.csv')
            self.translated_df.to_csv(output_file, index=False)
            print(f"Translated data saved to {output_file}")
        
        return self.translated_df
    
    def get_score_columns(self, df: pd.DataFrame = None) -> List[str]:
        """
        Get all score-related columns
        
        Args:
            df (pd.DataFrame, optional): DataFrame to analyze
            
        Returns:
            List[str]: List of score column names
        """
        if df is None:
            df = self.deduplicated_df if self.deduplicated_df is not None else self.df
        
        if df is None:
            return []
        
        score_suffix = self.config['score_suffix']
        return [col for col in df.columns if col.endswith(score_suffix)]
    
    def get_analysis_summary(self, df: pd.DataFrame = None) -> Dict:
        """
        Generate analysis summary statistics
        
        Args:
            df (pd.DataFrame, optional): DataFrame to analyze
            
        Returns:
            Dict: Summary statistics
        """
        if df is None:
            df = self.deduplicated_df if self.deduplicated_df is not None else self.df
        
        if df is None:
            return {}
        
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'level_column': self.config['level_column'],
            'difficulty_source_column': self.config['difficulty_source_column'],
            'score_columns': self.get_score_columns(df)
        }
        
        # Add level distribution if level column exists
        level_col = self.config['level_column']
        if level_col in df.columns:
            summary['level_distribution'] = df[level_col].value_counts().to_dict()
        
        # Add difficulty source distribution if column exists
        difficulty_col = self.config['difficulty_source_column']
        if difficulty_col in df.columns:
            summary['difficulty_source_distribution'] = df[difficulty_col].value_counts().to_dict()
        
        return summary
    
    def save_summary(self, summary: Dict = None) -> str:
        """
        Save analysis summary to a file
        
        Args:
            summary (Dict, optional): Summary to save
            
        Returns:
            str: Path to the saved summary file
        """
        if summary is None:
            summary = self.get_analysis_summary()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(self.config['output_dir'], f'analysis_summary_{timestamp}.json')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis summary saved to {summary_file}")
        return summary_file


def main():
    """
    Main function to demonstrate usage
    """
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <data_file>")
        print("Example: python analyzer.py data/sample_data.xlsx")
        return
    
    data_file = sys.argv[1]
    
    # Initialize analyzer
    analyzer = MedicalDataAnalyzer(data_file)
    
    # Load and process data
    analyzer.load_data()
    
    # If there's a JSON column, extract it
    if 'response' in analyzer.df.columns:
        analyzer.extract_json_fields()
    
    # Remove duplicates
    analyzer.remove_duplicates()
    
    # Translate difficulty sources
    analyzer.translate_difficulty_sources()
    
    # Generate summary
    summary = analyzer.get_analysis_summary()
    analyzer.save_summary(summary)
    
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()