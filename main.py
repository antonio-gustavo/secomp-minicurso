import cv2
import time
from deteccao_facial import DetectorFacial
from deteccao_emocional import DetectorEmocional

class DetectorEmocao:
    def __init__(self):
        print("Carregando detectores...")
        self.detector_facial = DetectorFacial()
        self.detector_emocional = DetectorEmocional()
        
        # Estado atual
        self.emocao_atual = "neutral"
        self.confianca = 0.0
        self.ultimo_update = 0
        print("Pronto!")
    
    def iniciar(self):
        """Inicia a webcam"""
        print("Iniciando webcam...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Webcam iniciada! Pressione 'q' para sair")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Espelha imagem
            frame = cv2.flip(frame, 1)
            
            # Detecta rostos
            rostos = self.detector_facial.detectar_rostos(frame)
            
            # Para cada rosto
            for (x, y, w, h) in rostos:
                # Analisa emoção a cada 1 segundo
                if time.time() - self.ultimo_update > 1.0:
                    rosto = frame[y:y+h, x:x+w]
                    self.emocao_atual, self.confianca = self.detector_emocional.analisar_emocao(rosto)
                    self.ultimo_update = time.time()
                
                # Desenha resultado
                cor = self.detector_emocional.obter_cor(self.emocao_atual)
                # cor = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 3)
                
                texto = f"{self.emocao_atual.title()}: {self.confianca:.0%}"
                cv2.putText(frame, texto, (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
            
            # Mostra frame
            cv2.imshow('Detector de Emocao', frame)
            
            # Sair com 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Finaliza
        cap.release()
        cv2.destroyAllWindows()
        print("Encerrado!")

# Executa
if __name__ == "__main__":
    detector = DetectorEmocao()
    detector.iniciar()