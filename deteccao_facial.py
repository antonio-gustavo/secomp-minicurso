import cv2

class DetectorFacial:
    def __init__(self):
        print("Carregando detector facial...")
        
        # Detector de rosto
        self.detector_rosto = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("Detector facial carregado!")
    
    def detectar_rostos(self, frame):
        """Detecta rostos no frame"""
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detecta rostos
        rostos = self.detector_rosto.detectMultiScale(
            gray, 1.1, 4, minSize=(120, 120)
        )
        
        return rostos