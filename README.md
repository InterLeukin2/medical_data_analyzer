# Medical Data Analyzer

A comprehensive tool for analyzing medical data with support for multiple formats and customizable analysis parameters.

## Features

- Extract data from various file formats (CSV, Excel)
- Perform statistical analysis on medical data
- Generate visualizations (charts, graphs)
- Create comprehensive PDF reports
- Support for multilingual reports (English, Chinese)
- Deduplication of data based on unique identifiers

## Installation

1. Clone the repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the analyzer with:
```
python src/main.py data/your_data_file.csv
```

### Command Line Arguments

- `input_file`: Input data file (CSV or Excel)
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