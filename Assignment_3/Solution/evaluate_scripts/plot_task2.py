import pandas as pd

metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-LCS', 'RTF']
latex_table = ""

def update_latex_table(df_scores, decoding_technique):
    global latex_table

    if decoding_technique == 'single-head':
        latex_table += "\\textbf{Single Medusa Head} "
        for i, _ in enumerate(df_scores.index):
            latex_table += f" & " + " & ".join([f"{score:.2f}" for score in df_scores.iloc[i]]) + " \\\\ \n"
        latex_table += "\hline \n"

    elif decoding_technique == 'multi-head':
        latex_table += "\\textbf{Multiple Medusa Heads} & & & & & \\\\ \n"
        for (W, S), row in df_scores.iterrows():
            latex_table += f"Beam-Width: {W}, Medusa-Heads: {S} & " + " & ".join([f"{score:.2f}" for score in row]) + " \\\\ \n"
        latex_table += "\hline \n"

def single_head_decoding():
    decoding_technique = 'single-head'
    scores = []

    file_name = f"./Results/1.3_Staring_into_Medusa_Heads/single_head_decoding.txt"
    with open(file_name, 'r') as f:
        lines = f.readlines()[-5:]

    file_scores = []
    for line in lines:
        metric, score = line.strip().split(':')
        file_scores.append(float(score))

    scores.append(file_scores)
    df_scores = pd.DataFrame(scores, columns=metrics, index=['Single Medusa Head'])
    update_latex_table(df_scores, decoding_technique)

def multiple_head_decoding(beam_widths, medusa_heads):
    decoding_technique = 'multi-head'
    scores = []
    
    # Loop over each combination of beam-width (W) and medusa-heads (S)
    for W in beam_widths:
        for S in medusa_heads:
            file_name = f"./Results/1.3_Staring_into_Medusa_Heads/multiple_{S}_head_{W}_beam_decoding.txt"
            
            with open(file_name, 'r') as f:
                lines = f.readlines()[-5:]

            file_scores = {}
            for line in lines:
                metric, score = line.strip().split(":")
                file_scores[metric] = float(score)

            file_scores['Beam Width'] = W
            file_scores['Medusa Heads'] = S
            scores.append(file_scores)

    df_scores = pd.DataFrame(scores)
    df_scores.set_index(['Beam Width', 'Medusa Heads'], inplace=True)

    update_latex_table(df_scores, decoding_technique)

# Example usage
beam_widths = [2, 5, 10]
medusa_heads = [2, 5]

if __name__ == '__main__':
    latex_table = ""
    single_head_decoding()
    multiple_head_decoding(beam_widths, medusa_heads)
    with open(f'./Results/1.3_Staring_into_Medusa_Heads/Latex_Table.txt', 'w') as f:
        f.write("\\begin{table}[h!]\n\\begin{center}\n\\begin{tabular}{|l||c c c c c||} \n")
        f.write("\\hline\nDecoding Technique & BLEU & ROUGE-1 & ROUGE-2 & ROUGE-LCS & RTF\\\\ [0.5ex] \n")
        f.write("\\hline\\hline\n")
        f.write(latex_table)  # Write the accumulated table content
        f.write("\\end{tabular}\n\\end{center}\n\\caption{Performance Metrics for Medusa}\n\\end{table}\n")