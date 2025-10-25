from ultralytics import YOLO
import cv2

#Cargar modelo preentrenamiento 
modelo = YOLO('yolov8n.pt')

#Leer una imagen 
image_path = 'adele.jpg'
img = cv2.imread(image_path)

#Ejecutrar deteccion
results = modelo(img)

#Mostrar resultados 
for r in results:
    #Dibujar los resultados en la imgagen
    annotated_img = r.plot()
    cv2.imshow('Deteccion YOLOv8', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    
#Guardar resultados
results[0].save(filename='resultado_yolo.jpg')
print("Deteccion cimmpletada. Imagen guardada como 'resultado_yolo.jpg'")
