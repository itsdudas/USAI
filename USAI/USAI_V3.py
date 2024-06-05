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
model_relativo_path = 'models/CNNbestalgotest71.h5'
model_path = os.path.join(common_parent_dir, model_relativo_path)
model = tf.keras.models.load_model(model_path)

# Define labels
labels = ['Colagem', 'Defeito US ', 'Erro Leitura US','Profundidade', 'Recobrimento']

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

        # Verificar se o arquivo de texto existe
        txt_file_path = os.path.join(test_images_folder, f'PI2K_{os.path.basename(test_images_folder)}.txt')
        if os.path.exists(txt_file_path):
            # Ler o arquivo de texto e associar os valores da taxa de soldagem às imagens
            data = pd.read_csv(txt_file_path, sep='\t')
            # Criar um dicionário para mapear o nome da imagem ao valor da taxa de soldagem
            image_to_weld_rate = dict(zip(data['Time'], data['Weld rate(%circ)']))
        else:
            # Se o arquivo de texto não existir, criar um dicionário vazio
            image_to_weld_rate = {}
            

        for image_file in os.listdir(test_images_folder):
            predictions = []
            # Verificar se o arquivo é uma imagem JPEG
            if image_file.lower().endswith('.jpeg'):
                if image_file in registered_images:
                    continue
                # Obter o valor da taxa de soldagem associado à imagem
                time_group = image_file.split('_')[6].split('.')[0]  # Extrair o grupo '22h10min42s' do nome do arquivo
                weld_rate_str = image_to_weld_rate.get(time_group, '0')  # Obter o valor da taxa de soldagem associado ao grupo de tempo como string
                # Substituir a vírgula pelo ponto
                weld_rate_str = weld_rate_str.replace(',', '.')
                weld_rate = float(weld_rate_str) 
                if weld_rate > 97:
                    print(f"{image_file}: Peça OK")
                    results.append([image_file] + [0] + [0]+ [0]+ [0] + [0]+ [1] + [time_group])
                    continue

                # If the image filename doesn't contain 'Time' + 'Part Name' + 'Ref' or the weld rate is not greater than 97, continue with normal classification
                img_path = os.path.join(test_images_folder, image_file)
                img = image.load_img(img_path, target_size=(600, 800))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array)

                if any(prediction <= 0.5 for prediction in predictions[0]):
                    # Se sim, copiar a imagem para a pasta NOK
                    copyfile(img_path, os.path.join(nok_folder, image_file))
                else:
                    continue
                    #print(f'A imagem {image_file} é OK.')



                # Adicionar os dados à lista de resultados com a hora na última posição
                results.append([image_file] + predictions[0].tolist() + [0] + [time_group])

                registered_images.append(image_file)

        # Criar o DataFrame
        df = pd.DataFrame(results, columns=['Image'] + labels + ['OK']+['Horas'])

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
                    if defect_counts['Erro Leitura US'] >= 4 and 'Erro Leitura US' not in displayed_warnings:
                        displayed_warnings.add('Erro Leitura US')
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
    messagebox.showwarning("Defeito: Erro na Leitura US", "O que fazer: \nLimpar a sonda com cuidado -> Verificar peça padrão e masters verde e amarelo.")

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
    style.configure("Custom.TFrame", background="#FFFFFF")  # Configure the style with desired background color

    contagem_reintroducoes, somaduplicacoes = contar_repeticoes(nomes_imagens, df)  # Desempacota a tupla
    print("Contagem de reintroduções:", contagem_reintroducoes)

    for imagem, nome_imagem in zip(imagens[:8], nomes_imagens[:8]):
        subframe = ttk.Frame(frame_mestre, style="Custom.TFrame")  # Use the custom style here
        subframe.pack( pady=0, padx=0, anchor="w",fill='both', expand=False)
        subframe.config(borderwidth=1, relief="solid")

        label_imagem = tk.Label(subframe, bd=0.25, relief=tk.SOLID)
        label_imagem.image = imagem
        label_imagem.configure(image=imagem)
        label_imagem.pack(side="left", padx=4, pady=4)

        partes_nome = nome_imagem.split("_")
        
        try:
            peca = partes_nome[0]
            data_matrix = partes_nome[1]
            data = partes_nome[5], partes_nome[4], partes_nome[3]
            data =  "/".join(data)
            hora = partes_nome[6]
        except IndexError:
            peca = data_matrix = data = hora = "ERRO"

        label_peca = ttk.Label(subframe, text=f"Peça:{peca}     DataMatrix:{data_matrix}    Data:{data}    Hora:{hora}", font=("Lato", 8), foreground="#000000", background="#FFFFFF")
        label_peca.pack(side='top', padx=1, pady=0.5)

        #label_data = ttk.Label(subframe, text=f"Data:{data}    Hora:{hora}", font=("Lato", 8), foreground="#000000", background="#FFFFFF")
        #label_data.pack(side='top', padx=5, pady=0.5)

        defeito = encontrar_defeito(nome_imagem, df)
        if defeito:
            label_defeito = ttk.Label(subframe, text=defeito, font=("Lato", 12, 'bold'), foreground="#F62C2C", background="#FFFFFF")
        else:
            label_defeito = ttk.Label(subframe, text='Nenhum defeito encontrado ', font=("Lato", 12, "bold"), foreground="#F62C2C", background="#FFFFFF")
        label_defeito.pack(side="top", pady=0.5, padx= 9)

        label_texto = ttk.Label(subframe,
                                text=f"Nº de Reintroduções: {contagem_reintroducoes.get(nome_imagem, 0)}",
                                font=("Lato", 8), foreground="#000000", background="#FFFFFF")
        label_texto.pack(side='bottom', padx=14, pady=0)

        if 'Recobrimento' in defeito:
            label_REC = ttk.Label(subframe, text='O que fazer: Verificar limpeza da lente, direção do fio de solda e bico.', font=("Lato", 8), foreground="#000000", background="#FFFFFF")
            label_REC.pack(side='bottom', padx=18, pady=0)

        if 'Defeito US' in defeito:
            label_DUS = ttk.Label(subframe, text="O que fazer: Limpar a sonda com cuidado. Verificar peça padrão e masters verde e amarelo.", font=("Lato", 8), foreground="#000000", background="#FFFFFF")
            label_DUS.pack(side='bottom', padx=18, pady=0)

        if 'Colagem' in defeito:
            label_COL = ttk.Label(subframe, text="O que fazer: Verificar limpeza da lente, banho da maq. de lavar e decapagem a laser.", font=("Lato", 8), foreground="#000000", background="#FFFFFF")
            label_COL.pack(side='bottom', padx=18, pady=0)

        if 'Profundidade' in defeito:
            label_PRO = ttk.Label(subframe, text="O que fazer: Verificar limpeza da lente de proteção, verificar potência e lente do laser. \n (Realizado pela Manuetenção)", font=("Lato", 8), foreground="#000000", background="#FFFFFF")
            label_PRO.pack(side='bottom', padx=18, pady=0)


def contar_repeticoes(nomes_imagens, df):
    contagem_por_data_matrix = {}
    contagem_incremental = {}

    imagens_ordenadas = sorted(nomes_imagens, key=lambda x: x.split("_")[5:7])  # Ordena por data e hora

    for nome_imagem in imagens_ordenadas:
        data_matrix = nome_imagem.split("_")[1]

        # Inicializa a contagem para o data_matrix se ainda não existir
        if data_matrix not in contagem_por_data_matrix:
            contagem_por_data_matrix[data_matrix] = 0

        # Incrementa a contagem para o data_matrix atual
        contagem_por_data_matrix[data_matrix] += 1

        # Armazena a contagem incremental para a imagem atual
        contagem_incremental[nome_imagem] = contagem_por_data_matrix[data_matrix] - 1 

    # Contar a frequência de cada código extraído do DataFrame
    codigos = df['Image'].astype(str).str.extract(r'_(.+?)_')[0].tolist()
    contagem_codigos = {}
    for codigo in codigos:
        if codigo in contagem_codigos:
            contagem_codigos[codigo] += 1
        else:
            contagem_codigos[codigo] = 1

    # Calcular o número de duplicações e filtrar os códigos com duplicações
    duplicacoes = {codigo: contagem - 1 for codigo, contagem in contagem_codigos.items() if contagem > 1}

    # Imprimir os códigos com duplicações e o número de duplicações
    if duplicacoes:
        print("Códigos com duplicações:")
        for codigo, num_duplicacoes in duplicacoes.items():
            print(f"Código: {codigo}, Número de duplicações: {num_duplicacoes}")
    else:
        print("Não foram encontradas duplicações.")

    
    somaduplicacoes= sum(duplicacoes.values())
    print(f'Número total de duplicações: {somaduplicacoes}')

    return contagem_incremental, somaduplicacoes


def Stats_window():
    defectsstats_window = tk.Toplevel()
    defectsstats_window.title("USAI: Defeitos do Dia")
    defectsstats_window.iconbitmap(r'C:\Users\pp01880\ML\Algo\USAI\pintas.ico')
    defectsstats_window.geometry("650x470")

    csv_file_path = os.path.join(current_folder, 'NOK', 'output.csv')
    print("Caminho do arquivo CSV:", csv_file_path)  
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path, header=0) 
    else:
        df = pd.DataFrame(columns=['Image'] + labels + ['Horas'])

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
    palette = ['#F62C2C', '#96FF54','#89B0AE', '#E77334', '#F2C57C', '#33658A']
    bar_plot = sns.barplot(x='Labels', y='Counts', data=counts_df, palette= palette, orient='v', hue='Labels', legend=False, width=0.75)

    plt.xlabel('Resultados', fontsize=8, labelpad=8)
    plt.ylabel('Nº de Peças/24h', fontsize=8, labelpad=8)
    plt.tight_layout(pad=1.5)
    
    for patch, value in zip(bar_plot.patches, counts_df['Counts']):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_y() + patch.get_height() + 0.08
        bar_plot.annotate(f'{value}', (x, y), ha='center', va='center', fontsize=6, color='black')

    canvas = FigureCanvasTkAgg(bar_plot.get_figure(), master=defectsstats_window)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=0, pady=0, fill=tk.BOTH, expand=True)

    info_frame = tk.Frame(defectsstats_window, relief=tk.SOLID, borderwidth=0, padx=10, pady=10, bg="#96FF54")
    info_frame.pack(fill=tk.X, padx=10, pady=(5, 5))  # Adicionar espaçamento

    # Atualizar o número total de duplicações
    _, total_duplicacoes = contar_repeticoes(df['Image'].tolist(), df)
    total_rows = len(df)
    percentage = (total_duplicacoes / len(df)) * 100 if len(df) > 0 else 0

    # Label para exibir o total de duplicações
    ttk.Label(info_frame,
               text=f"Nº Total de duplicações: {total_duplicacoes} ({percentage:.2f}%)",
               background='#96FF54',
               font=("Lato", 12)).pack(pady=(0,0))

    frame_imagem = tk.Frame(defectsstats_window)
    frame_imagem.pack(pady=(1, 1))

    imagem_png = Image.open(r'C:\Users\pp01880\ML\Algo\USAI\logosL.png')
    largura, altura = imagem_png.size
    imagem_png = imagem_png.resize((218, 38))  
    imagem_png = ImageTk.PhotoImage(imagem_png)

    label_imagem_png = tk.Label(frame_imagem, image=imagem_png)
    label_imagem_png.pack()

    label_imagem_png.image = imagem_png


def kill_terminal():
    root.destroy()

root = tk.Tk()
root.iconbitmap(r'C:\Users\pp01880\ML\Algo\USAI\pintas.ico')
root.title("USAI")
#root.geometry("560x710")
root.minsize(560, 710)
root.resizable(False, False)

frame_mestre = ttk.Frame(root)
frame_mestre.pack()


imagem_png = Image.open(r'C:\Users\pp01880\ML\Algo\USAI\logosL.png')
largura, altura = imagem_png.size
imagem_png = imagem_png.resize((218, 38))  
imagem_png = ImageTk.PhotoImage(imagem_png)

frame_imagem = ttk.Frame(root)
frame_imagem.pack(side="bottom",pady=(1,2))

# botao_encerrar = tk.Button(frame_imagem, text="Exit",
#                   font=("Lato",10),
#                   background="#FF7A7A", 
#                   width=5, height=1,
#                   bd=2, relief=tk.GROOVE,
#                   cursor="hand2",
#                   command=kill_terminal)
# botao_encerrar.pack(side="left", padx=(0,5), pady=(2,2))

label_imagem_png = tk.Label(frame_imagem, image=imagem_png)
label_imagem_png.pack(side="left", padx=(0,1),pady=(0,1))

label_imagem_png.image = imagem_png

botao = tk.Button(frame_imagem, text="Stats",
                  font=("Lato",10),
                  background="#6CACF0", 
                  width=5, height=1,
                  bd=2, relief=tk.GROOVE,
                  cursor="hand2",
                  command=Stats_window)                 
botao.pack(side="right", padx=5, pady=(2,2))

processing_thread = threading.Thread(target=process_images)
processing_thread.start()

root.protocol("WM_DELETE_WINDOW", kill_terminal)
root.mainloop()
