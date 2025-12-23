"""
Data Visualization Module
=========================

This module provides functions to create various visualizations for medical data analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List, Optional
from matplotlib import rcParams

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure fonts to avoid Chinese character issues
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    """
    Class for creating visualizations of medical data
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "visualizations"):
        """
        Initialize the visualizer
        
        Args:
            df (pd.DataFrame): DataFrame to visualize
            output_dir (str): Directory to save visualizations
        """
        self.df = df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
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
    
    def create_level_distribution_chart(self, language: str = "en") -> Optional[str]:
        """
        Create level distribution chart
        
        Args:
            language (str): Language for labels ("en" for English, "zh" for Chinese)
            
        Returns:
            str: Path to the saved chart, or None if failed
        """
        level_col = self.config['level_column']
        if level_col not in self.df.columns:
            print(f"Column '{level_col}' not found in data")
            return None
        
        level_data = self.df.dropna(subset=[level_col])
        if len(level_data) == 0:
            print("No data available for level distribution chart")
            return None
        
        plt.figure(figsize=(12, 8))  # Increased figure size for better clarity
        level_counts = level_data[level_col].value_counts().sort_index()
        total_count = len(level_data)
        bars = plt.bar(level_counts.index, level_counts.values, color='skyblue', edgecolor='black', linewidth=0.5)
        
        # Set labels based on language
        if language == "zh":
            plt.title('病例等级分布', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('等级', fontsize=16)
            plt.ylabel('病例数量', fontsize=16)
        else:
            plt.title('Distribution of Cases by Level', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('Level', fontsize=16)
            plt.ylabel('Number of Cases', fontsize=16)
        
        # Add a light grey grid
        plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Add value labels and percentage on bars
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_count) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + max(level_counts.values) * 0.01,  # Add a small offset
                     f'{int(height)}\n({percentage:.1f}%)',
                     ha='center', va='bottom', fontsize=12)  # Increased font size
        
        # Set y-axis to start from 0 and add a little extra space at the top for the labels
        plt.ylim(0, max(level_counts.values) * 1.15)  # Add 15% more space for labels
        
        # Set white background
        ax = plt.gca()
        ax.set_facecolor('white')
        # Add black border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        # Save chart
        lang_suffix = "_zh" if language == "zh" else ""
        filename = f'level_distribution{lang_suffix}.png'
        
        # Try to use dialog for renaming if tkinter is available
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            renamed_filename = simpledialog.askstring("重命名文件", "请输入文件名:", initialvalue=filename)
            root.destroy()
            
            if renamed_filename is None:  # User cancelled
                print("File save cancelled by user")
                plt.close()
                return None
            
            if not renamed_filename.endswith('.png'):
                renamed_filename += '.png'
                
            filepath = os.path.join(self.output_dir, renamed_filename)
        except ImportError:
            # If tkinter is not available, use the default filename
            print("Warning: tkinter not available, using default filename")
            filepath = os.path.join(self.output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created level distribution chart: {filepath}")
        return filepath
    
    def create_difficulty_source_chart(self, language: str = "en") -> Optional[str]:
        """
        Create difficulty source distribution chart
        
        Args:
            language (str): Language for labels ("en" for English, "zh" for Chinese)
            
        Returns:
            str: Path to the saved chart, or None if failed
        """
        difficulty_col = self.config['difficulty_source_column']
        if difficulty_col not in self.df.columns:
            print(f"Column '{difficulty_col}' not found in data")
            return None
        
        difficulty_data = self.df.dropna(subset=[difficulty_col])
        if len(difficulty_data) == 0:
            print("No data available for difficulty source distribution chart")
            return None
        
        plt.figure(figsize=(12, 8))  # Increased figure size for better clarity
        difficulty_counts = difficulty_data[difficulty_col].value_counts()
        total_count = len(difficulty_data)
        bars = plt.bar(difficulty_counts.index, difficulty_counts.values, color='lightcoral', edgecolor='black', linewidth=0.5)
        
        # Set labels based on language
        if language == "zh":
            plt.title('主要困难来源分布', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('主要困难来源', fontsize=16)
            plt.ylabel('病例数量', fontsize=16)
        else:
            plt.title('Distribution of Cases by Primary Difficulty Source', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('Primary Difficulty Source', fontsize=16)
            plt.ylabel('Number of Cases', fontsize=16)
        
        # Add a light grey grid
        plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels and percentage on bars
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_count) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + max(difficulty_counts.values) * 0.01,  # Add a small offset
                     f'{int(height)}\n({percentage:.1f}%)',
                     ha='center', va='bottom', fontsize=12)  # Increased font size
        
        # Set y-axis to start from 0 and add a little extra space at the top for the labels
        plt.ylim(0, max(difficulty_counts.values) * 1.15)  # Add 15% more space for labels
        
        # Set white background
        ax = plt.gca()
        ax.set_facecolor('white')
        # Add black border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        # Save chart
        lang_suffix = "_zh" if language == "zh" else ""
        filename = f'difficulty_source_distribution{lang_suffix}.png'
        
        # Try to use dialog for renaming if tkinter is available
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            renamed_filename = simpledialog.askstring("重命名文件", "请输入文件名:", initialvalue=filename)
            root.destroy()
            
            if renamed_filename is None:  # User cancelled
                print("File save cancelled by user")
                plt.close()
                return None
            
            if not renamed_filename.endswith('.png'):
                renamed_filename += '.png'
                
            filepath = os.path.join(self.output_dir, renamed_filename)
        except ImportError:
            # If tkinter is not available, use the default filename
            print("Warning: tkinter not available, using default filename")
            filepath = os.path.join(self.output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created difficulty source distribution chart: {filepath}")
        return filepath
    
    def create_difficulty_by_level_chart(self, language: str = "en") -> Optional[str]:
        """
        Create difficulty source by level chart (cross-tabulation heatmap)
        
        Args:
            language (str): Language for labels ("en" for English, "zh" for Chinese)
            
        Returns:
            str: Path to the saved chart, or None if failed
        """
        level_col = self.config['level_column']
        difficulty_col = self.config['difficulty_source_column']
        
        if level_col not in self.df.columns or difficulty_col not in self.df.columns:
            print(f"Required columns '{level_col}' or '{difficulty_col}' not found in data")
            return None
        
        cross_data = self.df.dropna(subset=[level_col, difficulty_col])
        if len(cross_data) == 0:
            print("No data available for difficulty source by level chart")
            return None
        
        plt.figure(figsize=(14, 10))  # Increased figure size for better clarity
        difficulty_level_crosstab = pd.crosstab(cross_data[level_col], cross_data[difficulty_col])
        
        # Set labels based on language
        if language == "zh":
            cbar_label = '病例数量'
            title = '各等级困难来源分布'
            xlabel = '主要困难来源'
            ylabel = '等级'
        else:
            cbar_label = 'Number of Cases'
            title = 'Difficulty Source Distribution by Level'
            xlabel = 'Primary Difficulty Source'
            ylabel = 'Level'
        
        sns.heatmap(difficulty_level_crosstab, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': cbar_label})
        plt.title(title, fontsize=18, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        
        # Save chart
        lang_suffix = "_zh" if language == "zh" else ""
        filename = f'difficulty_source_by_level{lang_suffix}.png'
        
        # Try to use dialog for renaming if tkinter is available
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            renamed_filename = simpledialog.askstring("重命名文件", "请输入文件名:", initialvalue=filename)
            root.destroy()
            
            if renamed_filename is None:  # User cancelled
                print("File save cancelled by user")
                plt.close()
                return None
            
            if not renamed_filename.endswith('.png'):
                renamed_filename += '.png'
                
            filepath = os.path.join(self.output_dir, renamed_filename)
        except ImportError:
            # If tkinter is not available, use the default filename
            print("Warning: tkinter not available, using default filename")
            filepath = os.path.join(self.output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created difficulty source by level chart: {filepath}")
        return filepath
    
    def create_scores_heatmap(self, language: str = "en") -> Optional[str]:
        """
        Create scores heatmap by level
        
        Args:
            language (str): Language for labels ("en" for English, "zh" for Chinese)
            
        Returns:
            str: Path to the saved chart, or None if failed
        """
        level_col = self.config['level_column']
        score_columns = self.get_score_columns()
        
        if level_col not in self.df.columns or len(score_columns) == 0:
            print(f"Required columns '{level_col}' or score columns not found in data")
            return None
        
        score_data = self.df.dropna(subset=[level_col])
        if len(score_data) == 0:
            print("No data available for scores heatmap")
            return None
        
        plt.figure(figsize=(12, 8))
        score_by_level = score_data.groupby(level_col)[score_columns].mean()
        
        # Set labels based on language
        if language == "zh":
            cbar_label = '平均得分'
            title = '各等级平均得分'
            xlabel = '得分维度'
            ylabel = '等级'
        else:
            cbar_label = 'Average Score'
            title = 'Average Scores by Level'
            xlabel = 'Score Dimensions'
            ylabel = 'Level'
        
        sns.heatmap(score_by_level, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': cbar_label})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.tight_layout()
        
        # Save chart
        lang_suffix = "_zh" if language == "zh" else ""
        filename = f'scores_heatmap{lang_suffix}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created scores heatmap: {filename}")
        return filepath
    
    def create_score_boxplots(self, language: str = "en") -> Optional[str]:
        """
        Create score distribution boxplots by level
        
        Args:
            language (str): Language for labels ("en" for English, "zh" for Chinese)
            
        Returns:
            str: Path to the saved chart, or None if failed
        """
        level_col = self.config['level_column']
        score_columns = self.get_score_columns()
        
        if level_col not in self.df.columns or len(score_columns) == 0:
            print(f"Required columns '{level_col}' or score columns not found in data")
            return None
        
        boxplot_data = self.df.dropna(subset=[level_col])
        if len(boxplot_data) == 0:
            print("No data available for score distribution boxplots")
            return None
        
        # Limit to first 12 score columns to avoid overly large plots
        selected_scores = score_columns[:12] if len(score_columns) > 12 else score_columns
        
        # Calculate grid size - use fewer columns to make plots larger
        n_scores = len(selected_scores)
        n_cols = min(3, n_scores)  # Reduce number of columns to make plots larger
        n_rows = (n_scores + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))  # Increase size per subplot
        
        # Handle case where there's only one subplot
        if n_scores == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(selected_scores):
            if i < len(axes):
                boxplot_data.boxplot(column=col, by=level_col, ax=axes[i])
                
                # Set title based on language
                col_title = col.replace("_", " ").title()
                if language == "zh":
                    axes[i].set_title(f'{col_title} 分布', fontweight='bold', fontsize=14)
                    axes[i].set_xlabel('等级', fontsize=12)
                    axes[i].set_ylabel('得分', fontsize=12)
                else:
                    axes[i].set_title(f'{col_title} Distribution', fontweight='bold', fontsize=14)
                    axes[i].set_xlabel('Level', fontsize=12)
                    axes[i].set_ylabel('Score', fontsize=12)
                
                # Add grid for better readability
                axes[i].grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Hide any unused subplots
        for j in range(len(selected_scores), len(axes)):
            fig.delaxes(axes[j])
        
        # Set suptitle based on language
        if language == "zh":
            plt.suptitle('各等级得分分布', fontsize=18, fontweight='bold', y=0.98)
        else:
            plt.suptitle('Score Distributions by Level', fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save chart
        lang_suffix = "_zh" if language == "zh" else ""
        filename = f'score_distributions_boxplot{lang_suffix}.png'
        
        # Try to use dialog for renaming if tkinter is available
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            renamed_filename = simpledialog.askstring("重命名文件", "请输入文件名:", initialvalue=filename)
            root.destroy()
            
            if renamed_filename is None:  # User cancelled
                print("File save cancelled by user")
                plt.close()
                return None
            
            if not renamed_filename.endswith('.png'):
                renamed_filename += '.png'
                
            filepath = os.path.join(self.output_dir, renamed_filename)
        except ImportError:
            # If tkinter is not available, use the default filename
            print("Warning: tkinter not available, using default filename")
            filepath = os.path.join(self.output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created score distributions boxplot: {filepath}")
        return filepath
    
    def create_radar_chart(self, language: str = "en") -> Optional[str]:
        """
        Create radar chart showing average scores by level
        
        Args:
            language (str): Language for labels ("en" for English, "zh" for Chinese)
            
        Returns:
            str: Path to the saved chart, or None if failed
        """
        level_col = self.config['level_column']
        score_columns = self.get_score_columns()
        
        if level_col not in self.df.columns or len(score_columns) == 0:
            print(f"Required columns '{level_col}' or score columns not found in data")
            return None
        
        radar_data = self.df.dropna(subset=[level_col])
        if len(radar_data) == 0:
            print("No data available for radar chart")
            return None
        
        # Calculate average scores by level
        avg_scores_by_level = radar_data.groupby(level_col)[score_columns].mean()
        
        # Number of variables
        categories = [col.replace('_score', '').replace('_', ' ').title() for col in score_columns]
        N = len(categories)
        
        # If too many categories, limit to first 15 for readability
        if N > 15:
            categories = categories[:15]
            avg_scores_by_level = avg_scores_by_level.iloc[:, :15]
            N = 15
        
        # Create the radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], categories, color='grey', size=8)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
        plt.ylim(0, 1)
        
        # Plot data for each level
        colors = plt.cm.Set1(np.linspace(0, 1, len(avg_scores_by_level)))
        for i, (level, row) in enumerate(avg_scores_by_level.iterrows()):
            values = row.values.flatten().tolist()
            values += values[:1]  # Complete the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Level {level}', color=colors[i])
            ax.fill(angles, values, alpha=0.2, color=colors[i])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set title based on language
        if language == "zh":
            plt.title('各等级平均得分 - 雷达图', size=14, pad=20, fontweight='bold')
        else:
            plt.title('Average Scores by Level - Radar Chart', size=14, pad=20, fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        lang_suffix = "_zh" if language == "zh" else ""
        filename = f'radar_chart{lang_suffix}.png'
        
        # Try to use dialog for renaming if tkinter is available
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            renamed_filename = simpledialog.askstring("重命名文件", "请输入文件名:", initialvalue=filename)
            root.destroy()
            
            if renamed_filename is None:  # User cancelled
                print("File save cancelled by user")
                plt.close()
                return None
            
            if not renamed_filename.endswith('.png'):
                renamed_filename += '.png'
                
            filepath = os.path.join(self.output_dir, renamed_filename)
        except ImportError:
            # If tkinter is not available, use the default filename
            print("Warning: tkinter not available, using default filename")
            filepath = os.path.join(self.output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created radar chart: {filepath}")
        return filepath
    
    def create_all_visualizations(self, language: str = "en") -> List[str]:
        """
        Create all available visualizations
        
        Args:
            language (str): Language for labels ("en" for English, "zh" for Chinese)
            
        Returns:
            List[str]: List of paths to created charts
        """
        charts = []
        
        # Try to create each chart
        level_chart = self.create_level_distribution_chart(language)
        if level_chart:
            charts.append(level_chart)
        
        difficulty_chart = self.create_difficulty_source_chart(language)
        if difficulty_chart:
            charts.append(difficulty_chart)
        
        cross_chart = self.create_difficulty_by_level_chart(language)
        if cross_chart:
            charts.append(cross_chart)
        
        heatmap_chart = self.create_scores_heatmap(language)
        if heatmap_chart:
            charts.append(heatmap_chart)
        
        radar_chart = self.create_radar_chart(language)
        if radar_chart:
            charts.append(radar_chart)
        
        print(f"Created {len(charts)} charts in {language} language")
        return charts


def main():
    """
    Main function to demonstrate usage
    """
    # This would typically be called from the main analyzer
    print("DataVisualizer module ready for use")


if __name__ == "__main__":
    main()