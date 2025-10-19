import mediapipe as mp
import cv2 as cv


class DetectorDeMaos:
    """
    Classe responsável pela detecção e rastreamento de mãos utilizando o MediaPipe.
    """

    def __init__(
        self,
        modo=False,
        max_maos=2,
        confianca_deteccao=0.5,
        confianca_rastreamento=0.5,
        cor_pontos=(0, 0, 255),
        cor_conexoes=(255, 255, 255)
    ):
        self.modo = modo
        self.max_maos = max_maos
        self.confianca_deteccao = confianca_deteccao
        self.confianca_rastreamento = confianca_rastreamento
        self.cor_pontos = cor_pontos
        self.cor_conexoes = cor_conexoes

        self._mediapipe_maos = mp.solutions.hands
        self._maos = self._mediapipe_maos.Hands(
            static_image_mode=self.modo,
            max_num_hands=self.max_maos,
            min_detection_confidence=self.confianca_deteccao,
            min_tracking_confidence=self.confianca_rastreamento
        )

        self._desenho = mp.solutions.drawing_utils
        self._config_pontos = self._desenho.DrawingSpec(color=self.cor_pontos)
        self._config_conexoes = self._desenho.DrawingSpec(color=self.cor_conexoes)

    def detectar_maos(self, frame, desenhar=True):
        """
        Detecta mãos em um frame e, opcionalmente, desenha os pontos.
        """
        imagem_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.resultados = self._maos.process(imagem_rgb)

        if self.resultados.multi_hand_landmarks:
            for mao in self.resultados.multi_hand_landmarks:
                if desenhar:
                    self._desenho.draw_landmarks(
                        frame,
                        mao,
                        self._mediapipe_maos.HAND_CONNECTIONS,
                        self._config_pontos,
                        self._config_conexoes
                    )
        return frame

    def obter_pontos(self, imagem, indice_mao=0, desenhar=True, cor=(255, 0, 255), raio=7):
        """
        Retorna uma lista com as coordenadas (x, y) dos pontos da mão.
        """
        lista_pontos = []

        if self.resultados.multi_hand_landmarks:
            mao = self.resultados.multi_hand_landmarks[indice_mao]

            for id_ponto, ponto in enumerate(mao.landmark):
                altura, largura, _ = imagem.shape
                cx, cy = int(ponto.x * largura), int(ponto.y * altura)
                lista_pontos.append([id_ponto, cx, cy])

                if desenhar and id_ponto in [4, 8, 12, 16, 20]:  # pontas dos dedos
                    cv.circle(imagem, (cx, cy), raio, cor, cv.FILLED)

        return lista_pontos


def main():
    webcam = cv.VideoCapture(0)
    detector = DetectorDeMaos(max_maos=1)

    while True:
        sucesso, frame = webcam.read()
        if not sucesso:
            break

        frame = cv.flip(frame, 1)
        frame = detector.detectar_maos(frame)
        pontos = detector.obter_pontos(frame)

        cv.imshow("Tradutor LIBRAS", frame)

        if cv.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

    webcam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
