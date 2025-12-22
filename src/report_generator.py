"""
Report Generator Module
======================

This module provides functions to generate comprehensive PDF reports for medical data analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from typing import Dict, List, Optional


class ReportGenerator:
    """
    Class for generating comprehensive PDF reports
    """
    
    def __init__(self, df: pd.DataFrame, visualization_dir: str = "visualizations", 
                 report_dir: str = "reports"):
        """
        Initialize the report generator
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            visualization_dir (str): Directory containing visualization files
            report_dir (str): Directory to save reports
        """
        self.df = df
        self.visualization_dir = visualization_dir
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        
        # Configuration
        self.config = {
            'level_column': 'final_level',
            'difficulty_source_column': 'primary_difficulty_source',
            'score_suffix': '_score'
        }
    
    def get_score_columns(self) -> List[str]:
        """
        Get all score-related columns
        
        Returns:
            List[str]: List of score column names
        """
        score_suffix = self.config['score_suffix']
        return [col for col in self.df.columns if col.endswith(score_suffix)]
    
    def generate_pdf_report(self, language: str = "en", include_charts: bool = True) -> str:
        """
        Generate a comprehensive PDF report
        
        Args:
            language (str): Language for the report ("en" for English, "zh" for Chinese)
            include_charts (bool): Whether to include charts in the report
            
        Returns:
            str: Path to the generated report
        """
        # Create a PDF report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        lang_suffix = "_en" if language == "en" else "_zh"
        report_filename = f'comprehensive_analysis_report{lang_suffix}_{timestamp}.pdf'
        report_filepath = os.path.join(self.report_dir, report_filename)
        
        with PdfPages(report_filepath) as pdf:
            # Title Page
            self._create_title_page(pdf, language)
            
            # Add charts if requested
            if include_charts:
                self._add_charts_to_report(pdf, language)
            
            # Add Statistics Tables
            self._add_statistics_tables(pdf, language)
        
        print(f"Comprehensive {language} report generated: {report_filepath}")
        return report_filepath
    
    def _create_title_page(self, pdf: PdfPages, language: str):
        """
        Create the title page for the report
        
        Args:
            pdf (PdfPages): PDF document
            language (str): Language for the title page
        """
        fig = plt.figure(figsize=(10, 8))
        
        # Set title based on language
        if language == "zh":
            title = '医疗数据分析综合报告'
            generated_text = f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            cases_text = f'总病例数: {len(self.df)}'
        else:
            title = 'Comprehensive Medical Data Analysis Report'
            generated_text = f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            cases_text = f'Total Cases: {len(self.df)}'
        
        fig.text(0.5, 0.7, title, ha='center', va='center', fontsize=20, weight='bold')
        fig.text(0.5, 0.5, generated_text, ha='center', va='center', fontsize=14)
        fig.text(0.5, 0.3, cases_text, ha='center', va='center', fontsize=12)
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
    def _add_charts_to_report(self, pdf: PdfPages, language: str):
        """
        Add charts to the report
        
        Args:
            pdf (PdfPages): PDF document
            language (str): Language for chart titles
        """
        # Define chart files and titles
        chart_files = [
            ('level_distribution', 'Level Distribution', '病例等级分布'),
            ('difficulty_source_distribution', 'Difficulty Source Distribution', '主要困难来源分布'),
            ('difficulty_source_by_level', 'Difficulty Source Distribution by Level', '各等级困难来源分布'),
            ('scores_heatmap', 'Average Scores by Level', '各等级平均得分'),
            ('score_distributions_boxplot', 'Score Distributions by Level', '各等级得分分布'),
            ('radar_chart', 'Average Scores by Level - Radar Chart', '各等级平均得分 - 雷达图')
        ]
        
        lang_suffix = "" if language == "en" else "_zh"
        
        for base_filename, en_title, zh_title in chart_files:
            try:
                chart_filename = f'{base_filename}{lang_suffix}.png'
                img_path = os.path.join(self.visualization_dir, chart_filename)
                
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.imshow(img)
                    ax.axis('off')
                    
                    # Set title based on language
                    title = en_title if language == "en" else zh_title
                    ax.set_title(title, fontsize=16, pad=20, weight='bold')
                    
                    pdf.savefig()
                    plt.close()
            except Exception as e:
                print(f"Could not add {chart_filename} to report: {e}")
    
    def _add_statistics_tables(self, pdf: PdfPages, language: str):
        """
        Add statistics tables to the report
        
        Args:
            pdf (PdfPages): PDF document
            language (str): Language for table titles
        """
        # Level Distribution Table (if available)
        level_col = self.config['level_column']
        if level_col in self.df.columns:
            level_data = self.df.dropna(subset=[level_col])
            if len(level_data) > 0:
                self._add_level_distribution_table(pdf, level_data, language)
        
        # Difficulty Source Distribution Table (if available)
        difficulty_col = self.config['difficulty_source_column']
        if difficulty_col in self.df.columns:
            difficulty_data = self.df.dropna(subset=[difficulty_col])
            if len(difficulty_data) > 0:
                self._add_difficulty_source_table(pdf, difficulty_data, language)
        
        # Score Statistics Table
        score_columns = self.get_score_columns()
        if len(score_columns) > 0:
            self._add_score_statistics_table(pdf, score_columns, language)
    
    def _add_level_distribution_table(self, pdf: PdfPages, level_data: pd.DataFrame, language: str):
        """
        Add level distribution table to the report
        
        Args:
            pdf (PdfPages): PDF document
            level_data (pd.DataFrame): Data with level information
            language (str): Language for table title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        level_col = self.config['level_column']
        level_counts = level_data[level_col].value_counts().sort_index()
        level_dist_data = []
        
        for level in level_counts.index:
            count = level_counts[level]
            percentage = count / len(level_data) * 100
            level_dist_data.append([level, count, f"{percentage:.2f}%"])
        
        # Set column labels based on language
        if language == "zh":
            col_labels = ['等级', '数量', '百分比']
            table_title = '等级分布统计'
        else:
            col_labels = ['Level', 'Count', 'Percentage']
            table_title = 'Level Distribution Statistics'
        
        table = ax.table(cellText=level_dist_data,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax.set_title(table_title, fontsize=16, pad=20, weight='bold')
        pdf.savefig()
        plt.close()
    
    def _add_difficulty_source_table(self, pdf: PdfPages, difficulty_data: pd.DataFrame, language: str):
        """
        Add difficulty source distribution table to the report
        
        Args:
            pdf (PdfPages): PDF document
            difficulty_data (pd.DataFrame): Data with difficulty source information
            language (str): Language for table title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        difficulty_col = self.config['difficulty_source_column']
        difficulty_counts = difficulty_data[difficulty_col].value_counts()
        difficulty_dist_data = []
        
        for difficulty in difficulty_counts.index:
            count = difficulty_counts[difficulty]
            percentage = count / len(difficulty_data) * 100
            difficulty_dist_data.append([difficulty, count, f"{percentage:.2f}%"])
        
        # Set column labels based on language
        if language == "zh":
            col_labels = ['困难来源', '数量', '百分比']
            table_title = '主要困难来源分布统计'
        else:
            col_labels = ['Difficulty Source', 'Count', 'Percentage']
            table_title = 'Difficulty Source Distribution Statistics'
        
        table = ax.table(cellText=difficulty_dist_data,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax.set_title(table_title, fontsize=16, pad=20, weight='bold')
        pdf.savefig()
        plt.close()
    
    def _add_score_statistics_table(self, pdf: PdfPages, score_columns: List[str], language: str):
        """
        Add score statistics table to the report
        
        Args:
            pdf (PdfPages): PDF document
            score_columns (List[str]): List of score column names
            language (str): Language for table title
        """
        fig, ax = plt.subplots(figsize=(14, min(10, len(score_columns) + 2)))
        ax.axis('tight')
        ax.axis('off')
        
        score_stats_data = []
        for col in score_columns:
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            
            # Format column name
            formatted_col = col.replace('_score', '').replace('_', ' ').title()
            
            score_stats_data.append([
                formatted_col,
                f"{mean_val:.3f}" if not np.isnan(mean_val) else "N/A",
                f"{std_val:.3f}" if not np.isnan(std_val) else "N/A",
                f"{min_val:.3f}" if not np.isnan(min_val) else "N/A",
                f"{max_val:.3f}" if not np.isnan(max_val) else "N/A"
            ])
        
        # Set column labels based on language
        if language == "zh":
            col_labels = ['维度', '均值', '标准差', '最小值', '最大值']
            table_title = '得分维度统计'
        else:
            col_labels = ['Dimension', 'Mean', 'Std Dev', 'Min', 'Max']
            table_title = 'Score Dimensions Statistics'
        
        table = ax.table(cellText=score_stats_data,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title(table_title, fontsize=16, pad=20, weight='bold')
        pdf.savefig()
        plt.close()


def main():
    """
    Main function to demonstrate usage
    """
    # This would typically be called from the main analyzer
    print("ReportGenerator module ready for use")


if __name__ == "__main__":
    main()