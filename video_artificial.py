from ultralytics import YOLO
import cv2

def main():
    # Cargar modelo YOLOv8 nano (rápido y ligero)
    print("Cargando modelo YOLOv8n...")
    modelo = YOLO('yolov8n.pt')
    
    # Iniciar cámara (0 = cámara principal)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return
    
    print("Cámara iniciada. Presiona ESC para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame de la cámara.")
            break
        
        # Detección en tiempo real 
        results = modelo(frame)
        annotated_frame = results[0].plot()
        
        # Mostrar resultados
        cv2.imshow('YOLOv8 - Detección en vivo', annotated_frame)
        
        # Presiona ESC (tecla 27) para salir
        if cv2.waitKey(1) & 0xFF == 27:
            print("Detención manual por el usuario.")
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada correctamente.")
    

if __name__ == "__main__":
    main()