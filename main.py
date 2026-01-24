from config.config import Config
from analysis.synthetic_data_analysis import SyntheticDataAnalysis
from analysis.m3_m4_analysis import M3M4Analysis
import argparse
import sys


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Прогнозирование временных рядов с использованием различных моделей')

    parser.add_argument('--mode', type=str, choices=['synthetic', 'm3', 'm4', 'm3m4', 'all'], default='all',
                        help='Режим анализа: synthetic - синтетические данные, m3 - данные M3, m4 - данные M4, m3m4 - данные M3/M4, all - все')

    parser.add_argument('--max-series', type=int, default=10,
                        help='Максимальное количество рядов для анализа в каждом наборе данных M3/M4')

    parser.add_argument('--test-size', type=int, default=12,
                        help='Размер тестовой выборки')

    return parser.parse_args()


def main():
    """Главная функция для запуска проекта"""
    # Настройка конфигурации
    Config.setup_directories()

    # Парсинг аргументов
    args = parse_arguments()

    # Установка параметров
    Config.TEST_SIZE = args.test_size

    print(f"{'=' * 80}")
    print("СИСТЕМА ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ")
    print(f"{'=' * 80}")
    print(f"Режим анализа: {args.mode}")
    print(f"Размер тестовой выборки: {Config.TEST_SIZE}")
    print(f"Максимальное количество рядов для M3/M4: {args.max_series}")
    print(f"Результаты будут сохранены в: {Config.RESULTS_DIR}")
    print(f"{'=' * 80}")

    try:
        if args.mode in ['synthetic', 'all']:
            print("\n" + "=" * 80)
            print("ЗАПУСК АНАЛИЗА СИНТЕТИЧЕСКИХ ДАННЫХ")
            print("=" * 80)
            synthetic_analysis = SyntheticDataAnalysis()
            synthetic_analysis.run_all_analyses()

        if args.mode in ['m3', 'm4', 'm3m4', 'all']:
            print("\n" + "=" * 80)
            print("ЗАПУСК АНАЛИЗА ДАННЫХ M3 И M4")
            print("=" * 80)
            m3m4_analysis = M3M4Analysis(max_series_per_dataset=args.max_series)
            
            if args.mode == 'm3':
                print("Запуск анализа только для M3")
                m3m4_analysis.run_analysis_m3_only()
            elif args.mode == 'm4':
                print("Запуск анализа только для M4")
                m3m4_analysis.run_analysis_m4_only()
            else:  # m3m4 or all
                m3m4_analysis.run_analysis()

        print(f"\n{'=' * 80}")
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print(f"Все результаты сохранены в директории: {Config.RESULTS_DIR}")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"ОШИБКА ПРИ ВЫПОЛНЕНИИ АНАЛИЗА: {str(e)}")
        print(f"{'=' * 80}")
        sys.exit(1)


if __name__ == "__main__":
    main()