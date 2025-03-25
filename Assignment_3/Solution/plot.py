import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-LCS']

def plot_bar_chart(metrics, df_scores, decoding_technique):
    df_scores.plot(kind='bar', figsize=(10,6))
    plt.title(f'Performance Scores for Different Models ({decoding_technique})')
    plt.xlabel(f'{decoding_technique}')
    plt.ylabel('Scores')
    plt.xticks(rotation=0)
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.savefig(f'./Results/1.1_Introduction_to_LLM_Decoding_Techniques/Performance_Metrics_for_{decoding_technique}.png')
    plt.close()

def plot_line_chart(metrics, df_scores, decoding_technique):
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(df_scores.index, df_scores[metric], marker='o', label=metric)
    plt.title(f'Performance Metrics for {decoding_technique}')
    plt.xlabel(f'{decoding_technique}')
    plt.ylabel('Scores')
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.savefig(f'./Results/1.1_Introduction_to_LLM_Decoding_Techniques/Line_Chart_for_Performance_Metrics_for_{decoding_technique}.png')
    plt.close()

def top_k_decoding():
    decoding_technique = 'Top_k_Sampling_Decoding_Techniques'
    scores = []

    for k in range(5, 11):
        file_name = f"./Results/1.1_Introduction_to_LLM_Decoding_Techniques/topk_{k}.txt"
        with open(file_name, 'r') as f:
            lines = f.readlines()[-4:]

        file_scores = []
        for line in lines:
            metric, score = line.strip().split(': ')
            file_scores.append(float(score))

        scores.append(file_scores)

    df_scores = pd.DataFrame(scores, columns=metrics, index=[f'Top-{k}' for k in range(5, 11)])

    plot_bar_chart(metrics, df_scores, decoding_technique)
    plot_line_chart(metrics, df_scores, decoding_technique)

def nucleus_sampling_decoding():
    decoding_technique = 'Nucleus_Sampling_Decoding_Techniques'
    scores = []

    for p in range(5, 10):
        file_name = f"./Results/1.1_Introduction_to_LLM_Decoding_Techniques/nucleus_sampling_0.{p}_outputs.txt"
        with open(file_name, 'r') as f:
            lines = f.readlines()[-4:]

        file_scores = []
        for line in lines:
            metric, score = line.strip().split(': ')
            file_scores.append(float(score))

        scores.append(file_scores)

    df_scores = pd.DataFrame(scores, columns=metrics, index=[f'0.{p}' for p in range(5, 10)])

    plot_bar_chart(metrics, df_scores, decoding_technique)
    plot_line_chart(metrics, df_scores, decoding_technique)

def random_sampling_with_temperature_scaling():
    decoding_technique = 'Random_Sampling_with_Temperature_Scaling'
    scores = []

    for tau in range(5, 10):
        file_name = f"./Results/1.1_Introduction_to_LLM_Decoding_Techniques/tau_0.{tau}_outputs.txt"
        with open(file_name, 'r') as f:
            lines = f.readlines()[-4:]

        file_scores = []
        for line in lines:
            metric, score = line.strip().split(': ')
            file_scores.append(float(score))

        scores.append(file_scores)

    df_scores = pd.DataFrame(scores, columns=metrics, index=[f'tau = 0.{p}' for p in range(5, 10)])

    plot_bar_chart(metrics, df_scores, decoding_technique)
    plot_line_chart(metrics, df_scores, decoding_technique)

top_k_decoding()
nucleus_sampling_decoding()
random_sampling_with_temperature_scaling()