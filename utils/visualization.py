import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from config.config import Config
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def setup_plot_style():
    """Настройка стиля графиков"""
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlepad'] = 15
    plt.rcParams['axes.labelpad'] = 10


def plot_forecast_comparison(train, test, forecasts, title, save_path=None):
    """Визуализация сравнения прогнозов разных моделей"""
    setup_plot_style()
    plt.figure(figsize=(14, 8))

    # Обучающие и тестовые данные
    plt.plot(train.index, train.values, 'b-', label='Обучающие данные', linewidth=2)
    plt.plot(test.index, test.values, 'g-', label='Тестовые данные (факт)', linewidth=2)

    # Прогнозы моделей
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#54A0FF']
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        label = f'{model_name} (sMAPE={forecast["metrics"]["sMAPE (%)"]:.2f}%)'
        plt.plot(test.index, forecast['values'], '--', label=label, linewidth=2.5, color=color)

    plt.title(f'Сравнение прогнозов для {title}', fontsize=16)
    plt.xlabel('Дата', fontsize=14)
    plt.ylabel('Значение', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_model_comparison(metrics_df, title, save_path=None):
    """Визуализация сравнения моделей по метрикам"""
    setup_plot_style()
    plt.figure(figsize=(14, 8))

    x = np.arange(len(metrics_df.index))
    width = 0.25

    plt.bar(x - width, metrics_df['RMSE'], width, label='RMSE', alpha=0.8)
    plt.bar(x, metrics_df['MAE'], width, label='MAE', alpha=0.8)
    plt.bar(x + width, metrics_df['sMAPE (%)'], width, label='sMAPE (%)', alpha=0.8)

    plt.xticks(x, metrics_df.index, rotation=45, ha='right', fontsize=12)
    plt.title(f'Сравнение моделей для {title}', fontsize=16)
    plt.ylabel('Значение метрики', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_aggregated_results(summary_m3, summary_m4, results_dir):
    """Визуализация агрегированных результатов"""
    setup_plot_style()

    # Since we're using Prophet for M3/M4 analysis, add a Model column to indicate this
    if summary_m3 is not None and not summary_m3.empty:
        summary_m3 = summary_m3.copy()
        summary_m3['Model'] = 'Prophet'

    if summary_m4 is not None and not summary_m4.empty:
        summary_m4 = summary_m4.copy()
        summary_m4['Model'] = 'Prophet'

    # 1. Сравнение моделей по sMAPE для M3 и M4
    if summary_m3 is not None and summary_m4 is not None and not summary_m3.empty and not summary_m4.empty:
        plt.figure(figsize=(15, 10))

        m3_means = summary_m3.groupby('Model')['sMAPE'].mean().sort_values()
        m4_means = summary_m4.groupby('Model')['sMAPE'].mean().sort_values()

        x = np.arange(len(m3_means.index))
        width = 0.35

        plt.bar(x - width / 2, m3_means.values, width, label='M3', alpha=0.8, color='#FF6B6B')
        plt.bar(x + width / 2, m4_means.values, width, label='M4', alpha=0.8, color='#4ECDC4')

        plt.xlabel('Модель', fontsize=14)
        plt.ylabel('Средний sMAPE', fontsize=14)
        plt.title('Сравнение моделей по среднему sMAPE между M3 и M4', fontsize=16, fontweight='bold')
        plt.xticks(x, m3_means.index, rotation=45, ha='right', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plt.savefig(f"{results_dir}/model_comparison_smase_m3_vs_m4.png", bbox_inches='tight', dpi=300)
        plt.close()

    # 2. Распределение sMAPE по моделям для каждого набора данных
    if summary_m3 is not None and not summary_m3.empty:
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=summary_m3, x='Model', y='sMAPE', palette='viridis')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.xlabel('Модель', fontsize=14)
        plt.ylabel('sMAPE', fontsize=14)
        plt.title('Распределение sMAPE по моделям для набора данных M3', fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/smase_distribution_m3.png", bbox_inches='tight', dpi=300)
        plt.close()

    if summary_m4 is not None and not summary_m4.empty:
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=summary_m4, x='Model', y='sMAPE', palette='viridis')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.xlabel('Модель', fontsize=14)
        plt.ylabel('sMAPE', fontsize=14)
        plt.title('Распределение sMAPE по моделям для набора данных M4', fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/smase_distribution_m4.png", bbox_inches='tight', dpi=300)
        plt.close()


def plot_synthetic_series(series_dict, save_path=None):
    """Визуализация синтетических временных рядов"""
    setup_plot_style()
    plt.figure(figsize=(15, 12))

    for i, (name, series) in enumerate(series_dict.items(), 1):
        plt.subplot(len(series_dict), 1, i)
        series.plot()
        plt.title(name, fontsize=14)
        plt.grid(True)
        plt.ylabel('Значение')

    plt.xlabel('Дата')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()