import cv2
from transformers import pipeline
from PIL import Image

class DetectorEmocional:
    def __init__(self):
        print("Carregando IA de emoções...")
        
        # Carrega modelo de IA
        self.detector_ia = pipeline(
            "image-classification",
            model="trpakov/vit-face-expression",
            device=-1
        )
        
        # Cores das emoções
        self.cores = {
            'happy': (0, 255, 0),      # Verde
            'sad': (255, 0, 0),        # Azul
            'angry': (0, 0, 255),      # Vermelho
            'fear': (128, 0, 128),     # Roxo
            'surprise': (0, 255, 255), # Ciano
            'disgust': (0, 128, 255),  # Laranja
            'neutral': (128, 128, 128) # Cinza
        }
        
        print("IA carregada!")
    
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
        

        return emocao, confianca
    
    def obter_cor(self, emocao):
        """Retorna cor da emoção"""
        return self.cores.get(emocao, (255, 255, 255))