import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np

# === Carrega o modelo treinado ===
with open("models/modelo_libras.pkl", "rb") as f:
    modelo = pickle.load(f)

# === Inicializa o MediaPipe ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# === Função para obter os pontos normalizados ===
def extrair_pontos_normalizados(frame, resultado):
    if not resultado.multi_hand_landmarks:
        return None

    mao = resultado.multi_hand_landmarks[0]

    pontos_x = [p.x for p in mao.landmark]
    pontos_y = [p.y for p in mao.landmark]

    # Define o ponto 0 como referência
    x0, y0 = pontos_x[0], pontos_y[0]

    # Normaliza em relação ao ponto 0
    pontos_relativos = []
    for i in range(21):
        pontos_relativos.append(pontos_x[i] - x0)
        pontos_relativos.append(pontos_y[i] - y0)

    return np.array(pontos_relativos).reshape(1, -1)

# === Inicia a captura da webcam ===
cap = cv.VideoCapture(0)
ultima_letra = ""
contador = 0

while True:
    sucesso, frame = cap.read()
    if not sucesso:
        break

    frame = cv.flip(frame, 1)
    imagem_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    resultado = hands.process(imagem_rgb)

    if resultado.multi_hand_landmarks:
        for mao in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

        # Extrai pontos relativos
        pontos = extrair_pontos_normalizados(frame, resultado)

        if pontos is not None:
            # Faz a predição
            letra_prevista = modelo.predict(pontos)[0]

            # Evita piscar letras rapidamente
            if letra_prevista == ultima_letra:
                contador += 1
            else:
                contador = 0
                ultima_letra = letra_prevista

            if contador > 5:  # Mostra só se mantiver o gesto por alguns frames
                cv.putText(frame, f"Letra: {letra_prevista}", (50, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    else:
        contador = 0

    cv.imshow("Tradutor LIBRAS -> Português", frame)

    if cv.waitKey(1) & 0xFF == 27:  # Tecla ESC pra sair
        break

cap.release()
cv.destroyAllWindows()
