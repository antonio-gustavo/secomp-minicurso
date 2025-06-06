import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import time

class DetectorEmocao:
    def __init__(self):
        print("🤖 Carregando IA de emoções...")
        
        # Carrega modelo de IA
        self.detector_ia = pipeline(
            "image-classification",
            model="trpakov/vit-face-expression",
            device=-1
        )
        
        # Detector de rosto
        self.detector_rosto = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Estado atual
        self.emocao_atual = "neutral"
        self.confianca = 0.0
        self.ultimo_update = 0
        
        # Cores das emoções
        self.cores = {
            'joy': (0, 255, 0),        # Verde
            'happiness': (0, 255, 0),  # Verde
            'sadness': (255, 0, 0),    # Azul
            'anger': (0, 0, 255),      # Vermelho
            'fear': (128, 0, 128),     # Roxo
            'surprise': (0, 255, 255), # Ciano
            'disgust': (0, 128, 255),  # Laranja
            'neutral': (128, 128, 128) # Cinza
        }
        
        print("✅ Pronto!")
    
    def analisar_emocao(self, rosto_img):
        """Analisa emoção do rosto"""
        # Converte para PIL e redimensiona
        rosto_pil = Image.fromarray(cv2.cvtColor(rosto_img, cv2.COLOR_BGR2RGB))
        rosto_pil = rosto_pil.resize((224, 224))
        
        # Analisa com IA
        resultado = self.detector_ia(rosto_pil)
        
        # Pega a melhor predição
        emocao = resultado[0]['label'].lower()
        confianca = resultado[0]['score']
        
        # Mapeia nomes das emoções
        mapeamento = {
            'angry': 'anger',
            'happy': 'joy',
            'sad': 'sadness'
        }
        
        emocao_final = mapeamento.get(emocao, emocao)
        return emocao_final, confianca
    
    def iniciar(self):
        """Inicia a webcam"""
        print("📷 Iniciando webcam...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Webcam iniciada! Pressione 'q' para sair")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Espelha imagem
            frame = cv2.flip(frame, 1)
            
            # Detecta rostos
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostos = self.detector_rosto.detectMultiScale(gray, 1.1, 4, minSize=(120, 120))
            
            # Para cada rosto
            for (x, y, w, h) in rostos:
                # Analisa emoção a cada 1 segundo
                if time.time() - self.ultimo_update > 1.0:
                    rosto = frame[y:y+h, x:x+w]
                    self.emocao_atual, self.confianca = self.analisar_emocao(rosto)
                    self.ultimo_update = time.time()
                
                # Desenha resultado
                cor = self.cores.get(self.emocao_atual, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 3)
                
                # Texto da emoção
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
        print("👋 Encerrado!")

# Executa
if __name__ == "__main__":
    detector = DetectorEmocao()
    detector.iniciar()