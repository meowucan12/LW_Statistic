import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def read_data(filename):
    df = pd.read_csv(filename, encoding='utf-8-sig')
    if len(df.columns) == 1 and any(s in df.columns[0] for s in (';', '\t')):
        df = pd.read_csv(filename, sep=None, engine='python', encoding='utf-8-sig')
    column = next((col for col in df if col.strip().lower() in ('logs', 'solubility')), None)
    if column is None:
        raise ValueError('В таблице нужен столбец LogS или Solubility (значения logS).')

    values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        raise ValueError('Недостаточно наблюдений (нужно хотя бы 20).')
    return values


def generate(values, n, rng):
    # Обратная эмпирическая функция распределения.
    sorted_values = np.sort(values)
    u = rng.random(n)
    indices = np.floor(u * len(sorted_values)).astype(int)
    return sorted_values[indices]


def metrics(values):
    return {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),
        'median': np.median(values),
        'q25': np.quantile(values, 0.25),
        'q75': np.quantile(values, 0.75),
    }


def run(filename, n=None, seed=42):
    values = read_data(filename)
    rng = np.random.default_rng(seed)

    # 80% для построения распределения, 20% для проверки.
    perm = rng.permutation(len(values))
    k = int(len(values) * 0.8)
    train = values[perm[:k]]
    test = values[perm[k:]]

    if n is None:
        n = len(test)
    if n < 1:
        raise ValueError('Число генерируемых значений должно быть положительным.')
    generated = generate(train, n, rng)

    output = Path('results')
    output.mkdir(exist_ok=True)
    pd.DataFrame({'LogS': generated}).to_csv(output / 'generated.csv', index=False)

    table = pd.DataFrame({'real': metrics(test), 'generated': metrics(generated)})
    table['difference'] = table['generated'] - table['real']
    table.to_csv(output / 'metrics.csv')

    ks = ks_2samp(test, generated)
    print(table.round(4))
    print(f'KS: D = {ks.statistic:.4f}; p = {ks.pvalue:.4f}')

    plt.figure(figsize=(8, 4))
    plt.hist(test, bins=30, density=True, alpha=0.5, label='Real')
    plt.hist(generated, bins=30, density=True, alpha=0.5, label='Generated')
    plt.xlabel('LogS')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / 'comparison.png', dpi=150)
    plt.close()

    return table, ks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='CSV-файл AqSolDB')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.csv, args.n, args.seed)
