# Medical Data Analyzer

A comprehensive tool for analyzing medical data with support for multiple formats and customizable analysis parameters.

## Features

- Extract data from various file formats (CSV, Excel, JSON, JSONL)
- Perform statistical analysis on medical data
- Generate visualizations (charts, graphs)
- Create comprehensive PDF reports
- Support for multilingual reports (English, Chinese)
- Deduplication of data based on unique identifiers

## New Features

- **Support for JSON and JSONL formats**: The analyzer now supports JSON and JSONL input files in addition to CSV and Excel
- **Interactive file naming**: Provides dialog boxes for renaming output charts and reports before saving
- **Detailed statistics by level**: Added detailed statistics tables for each level showing mean, standard deviation, min and max values
- **Enhanced visualizations**: Charts now display percentage values in addition to absolute numbers
- **Automated environment setup**: Added scripts to automatically create and configure virtual environments
- **Improved chart styling**: Level Distribution and Difficulty Source Distribution charts now have white background with black borders and light grey grid lines
- **Removed Score Distributions by Level chart**: Simplified visualization set by removing this chart
- **PDF reports formatted for A4 paper**: Reports are now optimized for A4 paper size

## Installation

1. Clone the repository
2. You can set up the environment in two ways:

   **Option A: Automatic setup with script**
   ```bash
   # Run the setup script to create virtual environment and install dependencies
   python setup_env.py
   # OR use the bash script (on macOS/Linux):
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

   **Option B: Manual setup**
   ```bash
   # Create and activate virtual environment
   python -m venv myvenv
   source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
   
   # Install required dependencies:
   pip install -r requirements.txt
   ```

## Usage

First, activate the virtual environment:
```
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
```

Then run the analyzer with:
```
python src/main.py data/your_data_file.csv
```

### Command Line Arguments

- `input_file`: Input data file (CSV, Excel, JSON, or JSONL)
- `--output-dir`: Output directory (default: 'output')
- `--viz-dir`: Visualization directory (default: 'visualizations')
- `--report-dir`: Report directory (default: 'reports')
- `--language`: Language for reports and visualizations (choices: 'en', 'zh', default: 'en')
- `--no-charts`: Skip chart generation
- `--no-report`: Skip report generation

## Dependencies

- Python 3.x
- pandas
- matplotlib
- seaborn
- numpy
- Other requirements listed in requirements.txt