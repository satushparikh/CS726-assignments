import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-LCS']
latex_table = ""

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

def update_latex_table(df_scores, decoding_technique):
    global latex_table

    if decoding_technique == 'Greedy_Decoding':
        latex_table += "Greedy Decoding "
        for i, _ in enumerate(df_scores.index):
            latex_table += f" & " + " & ".join([f"{score:.2f}" for score in df_scores.iloc[i]]) + " \\\\ \n"
        latex_table += "\hline \n"

    if decoding_technique == 'Random_Sampling_with_Temperature_Scaling':
        latex_table += "Random Sampling with Temperature Scaling & & & &\\\\ \n"
        for i, tau in enumerate(df_scores.index):
            latex_table += f"$\\tau$ = {tau[-3:]} & " + " & ".join([f"{score:.2f}" for score in df_scores.iloc[i]]) + " \\\\ \n"
        latex_table += "\hline \n"

    elif decoding_technique == 'Top_k_Sampling_Decoding_Techniques':
        latex_table += "Top-k sampling & & & &\\\\ \n"
        for i, k in enumerate(df_scores.index):
            latex_table += f"k = {k.split('-')[1]} & " + " & ".join([f"{score:.2f}" for score in df_scores.iloc[i]]) + " \\\\ \n"
        latex_table += "\hline \n"

    elif decoding_technique == 'Nucleus_Sampling_Decoding_Techniques':
        latex_table += "Nucleus Sampling Decoding Techniques & & & &\\\\ \n"
        for i, p in enumerate(df_scores.index):
            latex_table += f"p = {p} & " + " & ".join([f"{score:.2f}" for score in df_scores.iloc[i]]) + " \\\\ \n"
        latex_table += "\hline \n"

def greedy_decoding():
    decoding_technique = 'Greedy_Decoding'
    scores = []

    file_name = "./Results/1.1_Introduction_to_LLM_Decoding_Techniques/greedy_outputs.txt"
    with open(file_name, 'r') as f:
        lines = f.readlines()[-4:]

    file_scores = []
    for line in lines:
        metric, score = line.strip().split(': ')
        file_scores.append(float(score))

    scores.append(file_scores)
    df_scores = pd.DataFrame(scores, columns=metrics, index=['Greedy Decoding'])
    update_latex_table(df_scores, decoding_technique)

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
    update_latex_table(df_scores, decoding_technique)

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
    update_latex_table(df_scores, decoding_technique)

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
    update_latex_table(df_scores, decoding_technique)

if __name__ == '__main__':
    greedy_decoding()
    random_sampling_with_temperature_scaling()
    top_k_decoding()
    nucleus_sampling_decoding()

    with open('./Results/1.1_Introduction_to_LLM_Decoding_Techniques/Updated_Latex_Table.txt', 'w') as f:
        f.write("\\begin{center}\n\\begin{tabular}{|l||c c c c||} \n")
        f.write("\\hline\nDecoding Technique & BLEU & ROUGE-1 & ROUGE-2 & ROUGE-LCS \\\\ [0.5ex] \n")
        f.write("\\hline\\hline\n")
        f.write(latex_table)  # Write the accumulated table content
        f.write("\\end{tabular}\n\\end{center}\n")
