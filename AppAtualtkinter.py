import os
import sys
import re
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from shutil import copyfile
import time
from datetime import datetime
from openpyxl import load_workbook
import tkinter.messagebox as messagebox
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Common directory

common_parent_dir = r'C:\Users\pp01880\ML\Algo'

# Load the model
model_relativo_path = 'models/CNNbestalgotest59.h5'
model_path = os.path.join(common_parent_dir, model_relativo_path)
model = tf.keras.models.load_model(model_path)

# Define labels
labels = ['Colagem', 'DefeitoUS', 'Erro_leituraUS', 'OK', 'Profundidade', 'Recobrimento']

# Global variable for storing displayed warnings
displayed_warnings = set()

# Function to load the latest images
def carregar_ultimas_imagens():
    # Get the most recent 'NOK' folder
    nok_folders = [os.path.join(common_parent_dir, 'Teste', folder, 'NOK') for folder in os.listdir(os.path.join(common_parent_dir, 'Teste')) if os.path.isdir(os.path.join(common_parent_dir, 'Teste', folder, 'NOK'))]
    most_recent_nok_folder = max(nok_folders, key=os.path.getmtime) if nok_folders else None
    
    if most_recent_nok_folder:
        imagens = []
        nomes_imagens = []

        arquivos = os.listdir(most_recent_nok_folder)
        arquivos_imagem = [arquivo for arquivo in arquivos if arquivo.endswith('.jpeg')]
        arquivos_imagem.sort(key=lambda x: os.path.getmtime(os.path.join(most_recent_nok_folder, x)), reverse=True)
        ultimas_imagens = arquivos_imagem[:8]

        for imagem_nome in ultimas_imagens:
            caminho_imagem = os.path.join(most_recent_nok_folder, imagem_nome)
            imagem = Image.open(caminho_imagem)
            imagem = imagem.resize((95, 65))
            imagem = ImageTk.PhotoImage(imagem)
            imagens.append(imagem)

            # Adiciona o nome da imagem (sem a extensão) à lista de nomes de imagens
            nome_imagem = os.path.splitext(imagem_nome)[0]
            nomes_imagens.append(nome_imagem)

        return imagens, nomes_imagens, most_recent_nok_folder
    else:
        return None, None, None

current_folder = None


def process_images():
    global current_folder
    global displayed_warnings

    while True:
        # Listar todas as pastas no diretório
        all_folders = [folder for folder in os.listdir(os.path.join(common_parent_dir, 'Teste')) if
                       os.path.isdir(os.path.join(common_parent_dir, 'Teste', folder))]

        if not all_folders:
            time.sleep(60)
            continue

        folder_paths = [os.path.join(common_parent_dir, 'Teste', folder) for folder in all_folders]
        folder_mod_times = [os.path.getmtime(folder_path) for folder_path in folder_paths]

        most_recent_index = folder_mod_times.index(max(folder_mod_times))
        most_recent_folder = folder_paths[most_recent_index]

        if most_recent_folder != current_folder:
            current_folder = most_recent_folder
            test_images_folder = current_folder

            print(f'Nova pasta detectada: {test_images_folder}')

            nok_folder = os.path.join(test_images_folder, 'NOK')
            if not os.path.exists(nok_folder):
                os.makedirs(nok_folder)

        results = []

        processed_files = os.listdir(nok_folder) if os.path.exists(nok_folder) else []

        if os.path.exists(os.path.join(nok_folder, 'output.csv')):
            # Ler o arquivo CSV usando pandas
            registered_images_df = pd.read_csv(os.path.join(nok_folder, 'output.csv'))
            registered_images = registered_images_df['Image'].tolist()
        else:
            registered_images = []

        for image_file in os.listdir(test_images_folder):
            if image_file.lower().endswith('.jpeg'):
                if image_file in registered_images:
                    #print(f'A imagem {image_file} já está registrada. Ignorando processamento.')
                    continue

                img_path = os.path.join(test_images_folder, image_file)
                img = image.load_img(img_path, target_size=(500, 700))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array)

                if predictions[0][labels.index('OK')] <= 0.6:
                    #print(f'A imagem {image_file} é classificada como NOK.')
                    copyfile(img_path, os.path.join(nok_folder, image_file))
                else:
                    continue
                    #print(f'A imagem {image_file} é OK.')

                # Adicionar a hora no formato HH:MM:SS
                modification_time = os.path.getmtime(os.path.join(test_images_folder, image_file))
                modification_time = time.strftime('%H:%M:%S', time.localtime(modification_time))

                # Adicionar os dados à lista de resultados com a hora na última posição
                results.append([image_file] + predictions[0].tolist() + [modification_time])

                registered_images.append(image_file)

        df = pd.DataFrame(results, columns=['Image'] + labels + ['Horas'])

        # Caminho para o arquivo CSV
        csv_output_path = os.path.join(nok_folder, 'output.csv')

        if os.path.exists(csv_output_path):
            # Adiciona novos dados ao arquivo CSV existente
            df.to_csv(csv_output_path, mode='a', header=False, index=False)
            #print(f'Novos dados adicionados ao arquivo {csv_output_path}')
        else:
            # Cria um novo arquivo CSV
            df.to_csv(csv_output_path, index=False)
            #print(f'Arquivo CSV criado em {csv_output_path}')

        imagens, nomes_imagens, pasta_imagens = carregar_ultimas_imagens()
        if imagens and nomes_imagens and pasta_imagens:
            # Atualizar a interface com as últimas imagens carregadas
            registered_images_df = pd.read_csv(os.path.join(nok_folder, 'output.csv'))
            update_interface(imagens, nomes_imagens, pasta_imagens, registered_images_df)

            # Ler o arquivo CSV para verificar os últimos dados registrados
            registered_images_df = pd.read_csv(csv_output_path)

            # Verificar se existem pelo menos 5 linhas no DataFrame
            if len(registered_images_df) >= 5:
                # Selecionar as últimas 5 linhas
                last_five_rows = registered_images_df.iloc[-5:]

                # Contar o número de vezes que cada defeito ocorre nas últimas 5 linhas
                defect_counts = last_five_rows.drop(columns=['Image', 'Horas']).apply(lambda x: (x > 0.5).sum(), axis=0)

                if any(defect_counts >= 4):
                    if defect_counts['Colagem'] >= 4 and 'Colagem' not in displayed_warnings:
                        displayed_warnings.add('Colagem')
                        exibir_aviso_colagem()
                    if defect_counts['Recobrimento'] >= 4 and 'Recobrimento' not in displayed_warnings:
                        displayed_warnings.add('Recobrimento')
                        exibir_aviso_recobrimento()
                    if defect_counts['Erro_leituraUS'] >= 4 and 'Erro_leituraUS' not in displayed_warnings:
                        displayed_warnings.add('Erro_leituraUS')
                        exibir_aviso_erroleituraUS()
                    if defect_counts['Profundidade'] >= 4 and 'Profundidade' not in displayed_warnings:
                        displayed_warnings.add('Profundidade')
                        exibir_aviso_profundidade()

        time.sleep(15)

def exibir_aviso_recobrimento():
    messagebox.showwarning("Defeito: Fissura na zona de recobrimento", "O que fazer: \nVerificar limpeza da lente, direção do fio de solda e bico.")

def exibir_aviso_colagem():
    messagebox.showwarning("Defeito: Colagem", "O que fazer: \nVerificar limpeza da lente, banho da maq. de lavar e decapagem a laser.")

def exibir_aviso_erroleituraUS():
    messagebox.showwarning("Defeito: Erro na Leitura US", "O que fazer: \nLimpar a sonda com cuidado -> verificar peça padrão e masters verde e amarelo.")

def exibir_aviso_profundidade():
    messagebox.showwarning("Defeito: Falta de Profundidade", "O que fazer: \nVerificar limpeza da lente de proteção, verificar potência e lente do laser. \n (Realizado pela Manuetenção)")

def encontrar_defeito(nome_imagem, df):
    if not nome_imagem.endswith('.jpeg'):
        nome_imagem += '.jpeg'

    if not df.empty and 'Image' in df.columns:  # Check if DataFrame is not empty and contains 'Image' column
        if nome_imagem in df['Image'].values:
            linha = df.loc[df['Image'] == nome_imagem]
            for coluna in df.columns[1:]:
                if linha[coluna].iloc[0] >= 0.5:
                    print("Defeito encontrado na coluna:", coluna)
                    return coluna
        print("Nome da imagem não encontrado no Excel.")
    return 'Nenhum defeito encontrado'

def update_interface(imagens, nomes_imagens, nok_folder, df):
    global frame_mestre

    for subframe in frame_mestre.winfo_children():
        subframe.destroy()

    style = ttk.Style()
    style.configure("Custom.TFrame", background="#F8F8F8")  # Configure the style with desired background color

    for imagem, nome_imagem in zip(imagens[:8], nomes_imagens[:8]):
        subframe = ttk.Frame(frame_mestre, style="Custom.TFrame")  # Use the custom style here
        subframe.pack( pady=0, padx=0, anchor="w",fill='x', expand=True)

        label_imagem = tk.Label(subframe, bd=0.5, relief=tk.SOLID)
        label_imagem.image = imagem
        label_imagem.configure(image=imagem)
        label_imagem.pack(side="left", padx=5, pady=4)

        partes_nome = nome_imagem.split("_")
        
        try:
            peca = partes_nome[0]
            data_matrix = partes_nome[1]
            data = partes_nome[5], partes_nome[4], partes_nome[3]
            data =  "/".join(data)
            hora = partes_nome[6]
        except IndexError:
            peca = data_matrix = data = hora = "ERRO"

        label_peca = ttk.Label(subframe, text=f"Peça:{peca}     DataMatrix:{data_matrix}", font=("Lato", 8), foreground="#000000", background="#F8F8F8")
        label_peca.pack(side='top', padx=1, pady=0.5)

        label_data = ttk.Label(subframe, text=f"Data:{data}    Hora:{hora}", font=("Lato", 8), foreground="#000000", background="#F8F8F8")
        label_data.pack(side='top', padx=5, pady=0.5)

        defeito = encontrar_defeito(nome_imagem, df)
        if defeito:
            label_defeito = ttk.Label(subframe, text=defeito, font=("Lato", 12), foreground="#FF0000", background="#F8F8F8")
        else:
            label_defeito = ttk.Label(subframe, text='Nenhum defeito encontrado', font=("Lato", 12, "bold"), foreground="#FF0000", background="#F8F8F8")
        label_defeito.pack(side="bottom", pady=(2,0))
        
        subframe.config(borderwidth=2, relief="solid")

def Stats_window():
    defectsstats_window = tk.Toplevel()
    defectsstats_window.title("USAI: Defeitos do Dia")
    defectsstats_window.iconbitmap(r'C:\Users\pp01880\ML\Algo\USAI\USAI.ico')
    defectsstats_window.geometry("550x400")

    csv_file_path = os.path.join(current_folder, 'NOK', 'output.csv')
    print("Caminho do arquivo CSV:", csv_file_path)  
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path, header=0) 
    else:
        df = pd.DataFrame(columns=['Image'] + labels + ['Horas'])

    relevant_columns = ['Colagem', 'DefeitoUS', 'Erro_leituraUS', 'OK', 'Profundidade', 'Recobrimento']
    df[relevant_columns] = df[relevant_columns].astype(float)

    #print(df.dtypes)

    #print("Conteúdo do DataFrame:")
    #print(df)

    selected_columns = df.iloc[:, 1:7]

    counts = selected_columns.apply(lambda col: (col > 0.5).sum())

    if not counts.empty:
        print("Counts:")
        print(counts)
    else:
        print("Não há valores para contar.")

    counts_df = pd.DataFrame({'Labels': selected_columns.columns, 'Counts': counts})

    sns.set_theme(
        rc={'figure.dpi': 130},
        style="whitegrid",
        palette='pastel',
        font_scale=0.6
    )
    sns.set_context(rc={"font.family": "Lato"})

    plt.figure(figsize=(5, 2.8))
    bar_plot = sns.barplot(x='Labels', y='Counts', data=counts_df, palette="plasma", orient='v', hue='Labels', legend=False, width=0.7)

    plt.xlabel('Nº de Peças/dia', fontsize=8, labelpad=8)
    plt.ylabel('Defeitos', fontsize=8, labelpad=8)
    plt.tight_layout(pad=1.5)
    
    for patch, value in zip(bar_plot.patches, counts_df['Counts']):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_y() + patch.get_height() + 0.08
        bar_plot.annotate(f'{value}', (x, y), ha='center', va='center', fontsize=6, color='black')

    canvas = FigureCanvasTkAgg(bar_plot.get_figure(), master=defectsstats_window)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=0, pady=0, fill=tk.BOTH, expand=True)
    #canvas.get_tk_widget().config(borderwidth=0.5, relief=tk.SOLID, highlightbackground="#dbdbdb")

    frame_imagem = tk.Frame(defectsstats_window)
    frame_imagem.pack(pady=(1, 1))
    #imagem_png = Image.open(r'C:\Users\pp01880\ML\Algo\USAI\logo.png')  

    imagem_png = Image.open(r'C:\Users\pp01880\ML\Algo\USAI\logo.png')
    largura, altura = imagem_png.size
    imagem_png = imagem_png.resize((60, 33))  
    imagem_png = ImageTk.PhotoImage(imagem_png)

    label_imagem_png = tk.Label(frame_imagem, image=imagem_png)
    label_imagem_png.pack()

    label_imagem_png.image = imagem_png
    

def kill_terminal():
    root.destroy()

root = tk.Tk()
root.iconbitmap(r'C:\Users\pp01880\ML\Algo\USAI\USAI.ico')
root.title("USAI")
root.geometry("355x674")
root.resizable(False, False)

frame_mestre = ttk.Frame(root)
frame_mestre.pack()


imagem_png = Image.open(r'C:\Users\pp01880\ML\Algo\USAI\logo.png')
largura, altura = imagem_png.size
imagem_png = imagem_png.resize((60,33))  
imagem_png = ImageTk.PhotoImage(imagem_png)

frame_imagem = ttk.Frame(root)
frame_imagem.pack(side="bottom",pady=(2,2))

botao_encerrar = tk.Button(frame_imagem, text="Exit",
                  font=("Lato",10),
                  background="#F1948A", 
                  width=5, height=1,
                  bd=2, relief=tk.RIDGE,
                  cursor="hand2",
                  command=kill_terminal)
botao_encerrar.pack(side="left", padx=(0,5), pady=(2,2))

label_imagem_png = tk.Label(frame_imagem, image=imagem_png)
label_imagem_png.pack(side="left", pady=(2,2))

label_imagem_png.image = imagem_png

botao = tk.Button(frame_imagem, text="Stats",
                  font=("Lato",10),
                  background="#85C1E9", 
                  width=5, height=1,
                  bd=2, relief=tk.RIDGE,
                  cursor="hand2",
                  command=Stats_window)                 
botao.pack(side="right", padx=5, pady=(2,2))

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

processing_thread = threading.Thread(target=process_images)
processing_thread.start()

root.mainloop()
