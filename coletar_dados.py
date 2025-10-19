import cv2 as cv
import mediapipe as mp
import csv
import time

# === Configuração do MediaPipe ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# === Arquivo CSV de saída ===
arquivo_csv = "data/dataset_libras.csv"

# === Criação do cabeçalho do CSV ===
cabecalho = ["letra"]
for i in range(21):
    cabecalho.append(f"x{i}")
    cabecalho.append(f"y{i}")

# === Função para capturar coordenadas ===
def capturar_letra(letra, duracao=10):
    cap = cv.VideoCapture(0)
    tempo_inicial = time.time()

    print(f"\n👉 Capturando dados para a letra '{letra}' por {duracao} segundos...")

    with open(arquivo_csv, mode="a", newline="") as f:
        escritor = csv.writer(f)

        while time.time() - tempo_inicial < duracao:
            sucesso, frame = cap.read()
            if not sucesso:
                break

            frame = cv.flip(frame, 1)
            imagem_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            resultado = hands.process(imagem_rgb)

            if resultado.multi_hand_landmarks:
                for mao in resultado.multi_hand_landmarks:
                    pontos_x = []
                    pontos_y = []

                    # Extrai as coordenadas
                    for ponto in mao.landmark:
                        pontos_x.append(ponto.x)
                        pontos_y.append(ponto.y)

                    # Define o ponto 0 como referência (punho)
                    x0, y0 = pontos_x[0], pontos_y[0]

                    # Normaliza as coordenadas em relação ao ponto 0
                    pontos_relativos = []
                    for i in range(21):
                        pontos_relativos.append(pontos_x[i] - x0)
                        pontos_relativos.append(pontos_y[i] - y0)

                    escritor.writerow([letra] + pontos_relativos)

                    # Desenha a mão na tela
                    mp_draw.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

            cv.imshow("Coleta de Dados - LIBRAS", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv.destroyAllWindows()
    print(f"✅ Coleta para a letra '{letra}' finalizada!")

# === Cria o cabeçalho se o arquivo estiver vazio ===
try:
    with open(arquivo_csv, "x", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(cabecalho)
except FileExistsError:
    pass

# === Captura de amostras ===
while True:
    letra = input("\nDigite a letra que deseja capturar (A-Z) ou 'sair': ").upper()
    if letra == "SAIR":
        break
    capturar_letra(letra)
