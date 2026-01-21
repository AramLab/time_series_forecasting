"""
Чистая Python реализация CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)
для совместимости с ARM64 Mac
"""

import numpy as np
from scipy import signal


class SimpleCEEMDAN:
    """Упрощенная реализация CEEMDAN на чистом Python без зависимостей C++"""
    
    def __init__(self, trials=100, noise_width=0.05, n_imfs=None):
        """
        Инициализация CEEMDAN
        
        Args:
            trials: Количество испытаний с белым шумом
            noise_width: Амплитуда белого шума (относительно std сигнала)
            n_imfs: Максимальное количество IMF для извлечения
        """
        self.trials = trials
        self.noise_width = noise_width
        self.n_imfs = n_imfs
        self.IMFs = []
        self.residue = None
    
    def __call__(self, x):
        """
        Делает объект вызываемым как функция для совместимости с PyEMD API
        
        Args:
            x: Входной сигнал
            
        Returns:
            Список IMF
        """
        try:
            imfs, residue = self.ceemdan(x)
            self.IMFs = imfs
            self.residue = residue
            
            # Если нет IMF, возвращаем исходный сигнал
            if not imfs:
                return [np.asarray(x, dtype=float)]
            
            return imfs
        except Exception as e:
            # Fallback: возвращаем исходный сигнал
            return [np.asarray(x, dtype=float)]
        
    def emd(self, x, max_imf=None):
        """
        Простая реализация EMD (Empirical Mode Decomposition)
        
        Args:
            x: Входной сигнал
            max_imf: Максимальное количество IMF
            
        Returns:
            imfs: Список внутренних мод
            residue: Остаток
        """
        try:
            imfs = []
            x = np.asarray(x, dtype=float)
            
            # Проверка входных данных
            if len(x) < 4:
                return imfs, x
            
            if np.all(np.isnan(x)) or np.all(np.isinf(x)):
                return imfs, x
            
            if max_imf is None:
                max_imf = min(int(np.log2(len(x))) + 1, 10)  # Ограничиваем до 10 IMF
            
            for imf_num in range(max_imf):
                if len(x) < 4:  # Не можем продолжать при слишком малом сигнале
                    break
                
                # Извлечение одной IMF
                imf = self._extract_imf(x)
                
                if imf is None:
                    break
                
                # Проверяем, что IMF имеет достаточную энергию
                imf_energy = np.sum(np.abs(imf))
                if imf_energy < 1e-10:
                    break
                
                imfs.append(imf)
                x = x - imf
            
            return imfs, x
            
        except Exception as e:
            # Fallback: возвращаем то, что есть
            return imfs if 'imfs' in locals() else [], x
    
    def _extract_imf(self, x, max_iter=10):
        """
        Извлечение одной IMF через интерполяцию экстремумов
        
        Args:
            x: Входной сигнал
            max_iter: Максимум итераций
            
        Returns:
            imf: Извлеченная IMF или None если не может быть извлечена
        """
        x = np.asarray(x, dtype=float)
        if len(x) < 4:
            return None
            
        h = x.copy()
        
        for iteration in range(max_iter):
            # Найти локальные максимумы и минимумы
            d = np.diff(h)
            
            # Поиск экстремумов
            maxima_idx = []
            minima_idx = []
            
            for i in range(1, len(h) - 1):
                if h[i] > h[i-1] and h[i] > h[i+1]:
                    maxima_idx.append(i)
                elif h[i] < h[i-1] and h[i] < h[i+1]:
                    minima_idx.append(i)
            
            # Добавляем граничные точки если нужно
            if len(h) > 2:
                if h[0] > h[1]:
                    maxima_idx.insert(0, 0)
                elif h[0] < h[1]:
                    minima_idx.insert(0, 0)
                    
                if h[-1] > h[-2]:
                    maxima_idx.append(len(h) - 1)
                elif h[-1] < h[-2]:
                    minima_idx.append(len(h) - 1)
            
            # Проверяем количество экстремумов
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                # Если недостаточно экстремумов, возвращаем то, что есть
                return h if iteration == 0 else h
            
            # Интерполяция огибающих
            max_env = self._interpolate_envelope(np.array(maxima_idx), h)
            min_env = self._interpolate_envelope(np.array(minima_idx), h)
            
            if max_env is None or min_env is None:
                return h
            
            # Средняя огибающая
            mean_env = (max_env + min_env) / 2
            
            # Проверка условия сходимости
            prev_h = h.copy()
            h = h - mean_env
            
            # Проверка сходимости с обработкой исключений
            try:
                energy_ratio = np.sum(np.abs(h)) / (np.sum(np.abs(prev_h)) + 1e-10)
                if energy_ratio < 0.001 or np.isnan(energy_ratio):
                    break
            except:
                break
        
        return h
    
    def _interpolate_envelope(self, extrema_indices, signal):
        """
        Интерполирование огибающей через экстремумы
        
        Args:
            extrema_indices: Индексы экстремумов
            signal: Сигнал
            
        Returns:
            envelope: Интерполированная огибающая
        """
        if len(extrema_indices) < 2:
            return None
        
        try:
            extrema_indices = np.asarray(extrema_indices, dtype=int)
            signal = np.asarray(signal, dtype=float)
            
            # Убеждаемся, что индексы в пределах границ
            extrema_indices = extrema_indices[(extrema_indices >= 0) & (extrema_indices < len(signal))]
            
            if len(extrema_indices) < 2:
                return None
            
            x = np.arange(len(signal))
            y = signal[extrema_indices]
            
            # Линейная интерполяция для надежности
            try:
                envelope = np.interp(x, extrema_indices, y, left=y[0], right=y[-1])
                
                # Проверяем на NaN
                if np.any(np.isnan(envelope)):
                    # Fallback: копируем значения из сигнала
                    return signal.copy()
                    
                return envelope
            except Exception:
                # Если интерполяция не работает, возвращаем копию сигнала
                return signal.copy()
        except Exception as e:
            # Любая другая ошибка - возвращаем копию сигнала
            return signal.copy() if 'signal' in locals() else None
    
    def ceemdan(self, x):
        """
        Основной метод CEEMDAN
        
        Args:
            x: Входной сигнал
            
        Returns:
            imfs: Список IMF
            residue: Остаток
        """
        try:
            x = np.asarray(x, dtype=float)
            
            # Проверка входных данных
            if len(x) < 4:
                self.IMFs = []
                self.residue = x
                return [], x
            
            if np.all(np.isnan(x)) or np.all(np.isinf(x)):
                self.IMFs = []
                self.residue = x
                return [], x
            
            # Очистка от NaN/Inf
            x = np.where(np.isfinite(x), x, np.nanmean(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else 0)
            
            n = len(x)
            
            # Нормализация шума
            noise_std = np.std(x) * self.noise_width
            if noise_std == 0:
                noise_std = 1e-6
            
            # Инициализация аккумулятора для ensemble
            ensemble_imfs = None
            max_num_imfs = 0
            
            for trial in range(self.trials):
                try:
                    # Добавление белого шума
                    noise = np.random.randn(n) * noise_std
                    x_noisy = x + noise
                    
                    # EMD шумного сигнала
                    imfs, residue = self.emd(x_noisy, max_imf=self.n_imfs)
                    
                    # Отслеживаем максимальное количество IMF
                    max_num_imfs = max(max_num_imfs, len(imfs))
                    
                    if ensemble_imfs is None:
                        # Инициализируем с достаточным количеством слотов
                        # +1 для остатка
                        ensemble_imfs = [np.zeros(n) for _ in range(max_num_imfs + 1)]
                    
                    # Проверяем, не нужно ли расширить список
                    while len(imfs) > len(ensemble_imfs) - 1:
                        # Расширяем список
                        ensemble_imfs.insert(-1, np.zeros(n))
                    
                    # Аккумуляция IMF
                    for i, imf in enumerate(imfs):
                        if len(imf) == n:  # Проверяем размер
                            ensemble_imfs[i] += imf / self.trials
                    
                    # Добавляем остаток
                    if len(residue) == n:
                        ensemble_imfs[-1] += residue / self.trials
                        
                except Exception as trial_error:
                    # Пропускаем ошибочные trial и продолжаем
                    continue
            
            # Проверяем, был ли хотя бы один успешный trial
            if ensemble_imfs is None:
                self.IMFs = []
                self.residue = x
                return [], x
            
            # Возвращаем только заполненные IMF (исключая остаток)
            final_imfs = []
            for imf in ensemble_imfs[:-1]:
                if len(imf) == n and np.any(np.isfinite(imf)):
                    final_imfs.append(imf)
            
            # Убеждаемся, что остаток корректный
            final_residue = ensemble_imfs[-1] if len(ensemble_imfs[-1]) == n else x
            
            self.IMFs = final_imfs
            self.residue = final_residue
            
            return self.IMFs, self.residue
            
        except Exception as e:
            # Fallback: возвращаем исходный сигнал как остаток
            self.IMFs = []
            self.residue = x
            return [], x
    
    def decompose(self, x):
        """Альтернативное имя для метода ceemdan"""
        return self.ceemdan(x)


# Для совместимости с существующим кодом
CEEMDAN = SimpleCEEMDAN
