import numpy as np
import matplotlib.pyplot as plt
import os

# Перевірка встановлення модулів
try:
    import numpy
    import matplotlib
    print("Модулі numpy та matplotlib встановлені.")
except ImportError as e:
    print(f"Помилка: {e}. Встановіть модулі командою: pip install --user numpy matplotlib")
    exit(1)

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P
        return self.x

def analyze_result(result):
    name = result['name']
    before = result['noise_var_before']
    after = result['noise_var_after']
    params = result['params']
    
    if 'Збільшення Q' in name:
        return "Збільшення Q призводить до більшої довіри до вимірювань, що може погіршити згладжування."
    elif 'Зменшення Q' in name:
        return "Зменшення Q підвищує довіру до моделі, покращує згладжування."
    elif 'Збільшення R' in name:
        return "Збільшення R знижує довіру до вимірювань, оцінка залежить від моделі."
    elif 'Зменшення R' in name:
        return "Зменшення R підвищує довіру до вимірювань, покращує точність."
    elif 'Збільшення P' in name:
        return "Збільшення P уповільнює збіжність фільтра через високу початкову невпевненість."
    elif 'Зменшення P' in name:
        return "Зменшення P прискорює збіжність через високу впевненість у початковій оцінці."
    elif 'Зміна x' in name:
        return "Зміна початкового стану впливає на початкову оцінку, але фільтр адаптується."
    elif 'Зміна offset' in name:
        return "Зміна offset не впливає на якість фільтрації."
    elif 'Збільшення total_time' in name:
        return "Збільшення часу дозволяє фільтру краще стабілізуватися."
    else:
        return "Базовий випадок показує стандартну поведінку фільтра."

def run_kalman_filter(params, experiment_name):
    frequency = params.get('frequency', 1)
    amplitude = params.get('amplitude', 5)
    offset = params.get('offset', 10)
    sampling_interval = params.get('sampling_interval', 0.001)
    total_time = params.get('total_time', 1)
    noise_variance = params.get('noise_variance', 16)
    F = params.get('F', np.array([[1]]))
    H = params.get('H', np.array([[1]]))
    Q = params.get('Q', np.array([[1]]))
    R = params.get('R', np.array([[10]]))
    P = params.get('P', np.array([[1]]))
    x = params.get('x', np.array([[0]]))

    # Генерація сигналу
    time_steps = np.arange(0, total_time, sampling_interval)
    true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)
    noise_std_dev = np.sqrt(noise_variance)
    noisy_signal = true_signal + np.random.normal(0, noise_std_dev, len(true_signal))

    # Застосування фільтра Калмана
    kf = KalmanFilter(F, H, Q, R, P, x)
    kalman_estimates = []
    for measurement in noisy_signal:
        kf.predict()
        estimate = kf.update(measurement)
        kalman_estimates.append(estimate[0][0])

    # Розрахунок дисперсії
    noise_variance_before = np.var(noisy_signal - true_signal)
    noise_variance_after = np.var(np.array(kalman_estimates) - true_signal)

    # Побудова графіку
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, noisy_signal, label='Зашумлений сигнал', color='orange', linestyle='-', alpha=0.6)
    plt.plot(time_steps, true_signal, label='Чистий сигнал (синусоїда)', linestyle='--', color='blue')
    plt.plot(time_steps, kalman_estimates, label='Оцінка фільтра Калмана', color='green')
    plt.xlabel('Час (с)')
    plt.ylabel('Значення')
    plt.title(f'Фільтр Калмана: {experiment_name}, Q={Q[0][0]}, R={R[0][0]}, P={P[0][0]}, x={x[0][0]}, offset={offset}, total_time={total_time}')
    plt.legend()
    plt.grid()

    # Збереження графіку
    safe_name = experiment_name.replace(" ", "_").replace(":", "_")
    output_dir = r"C:\Users\MSI\Desktop\РПЗ КС\lab5\screenshots"
    output_path = os.path.join(output_dir, f"kalman_{safe_name}.png")
    try:
        plt.savefig(output_path)
        print(f"Графік збережено: {output_path}")
    except Exception as e:
        print(f"Помилка збереження графіку: {e}")
    
    # Відображення графіку
    plt.show()

    return noise_variance_before, noise_variance_after

def get_user_params():
    print("\nВведіть параметри для експерименту:")
    name = input("Назва експерименту: ")
    try:
        Q = float(input("Q (коваріація шуму процесу, наприклад, 1): "))
        R = float(input("R (коваріація шуму вимірювань, наприклад, 10): "))
        P = float(input("P (початкова коваріація, наприклад, 1): "))
        x = float(input("x (початковий стан, наприклад, 0): "))
        offset = float(input("offset (зсув сигналу, наприклад, 10): "))
        total_time = float(input("total_time (загальний час, наприклад, 1): "))
    except ValueError as e:
        print(f"Помилка: Введіть числові значення. {e}")
        return get_user_params()
    
    return {
        'name': name,
        'params': {
            'frequency': 1,
            'amplitude': 5,
            'offset': offset,
            'sampling_interval': 0.001,
            'total_time': total_time,
            'noise_variance': 16,
            'F': np.array([[1]]),
            'H': np.array([[1]]),
            'Q': np.array([[Q]]),
            'R': np.array([[R]]),
            'P': np.array([[P]]),
            'x': np.array([[x]])
        }
    }

if __name__ == "__main__":
    # Перевірка директорії
    output_dir = r"C:\Users\MSI\Desktop\РПЗ КС\lab5\screenshots"
    if not os.path.exists(output_dir):
        print(f"Помилка: Директорія {output_dir} не існує. Створіть її.")
        exit(1)

    results = []
    while True:
        exp = get_user_params()
        print(f"\nЕксперимент: {exp['name']}")
        try:
            noise_var_before, noise_var_after = run_kalman_filter(exp['params'], exp['name'])
            print(f"Дисперсія шуму до фільтрації: {noise_var_before:.2f}")
            print(f"Дисперсія шуму після фільтрації: {noise_var_after:.2f}")
            results.append({
                'name': exp['name'],
                'params': exp['params'],
                'noise_var_before': noise_var_before,
                'noise_var_after': noise_var_after
            })
        except Exception as e:
            print(f"Помилка виконання експерименту: {e}")

        continue_exp = input("\nБажаєте провести ще один експеримент? (y/n): ")
        if continue_exp.lower() != 'y':
            break

    # Висновки
    print("\nЗагальні висновки:")
    for result in results:
        print(f"\nЕксперимент: {result['name']}")
        print(f"Параметри: {result['params']}")
        print(f"Дисперсія до: {result['noise_var_before']:.2f}, після: {result['noise_var_after']:.2f}")
        print(f"Коментар: {analyze_result(result)}")