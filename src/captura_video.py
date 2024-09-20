import cv2
import time
import math as m
import mediapipe as mp
import argparse
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    try:
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        return int(180 / m.pi * theta)
    except ValueError:
        return 0

def sendWarning():
    print("¡Alerta! Manteniendo mala postura durante demasiado tiempo.")

def draw_transparent_box(image, text, position):
    overlay = image.copy()
    cv2.rectangle(overlay, position, (position[0] + 400, position[1] + 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    for i, line in enumerate(text.split('\n')):
        cv2.putText(image, line, (position[0] + 10, position[1] + 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def initialize_plot():
    plt.ion()  # Activar modo interactivo
    fig, ax = plt.subplots()
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Duración (s)')
    ax.set_title('Postura a lo largo del tiempo')
    ax.set_xlim(0, 300)  # Ajusta el rango según sea necesario
    ax.set_ylim(0, 300)
    return fig, ax

def update_line_plot(ax, good_time_list, bad_time_list, time_list):
    ax.clear()
    ax.plot(time_list, good_time_list, color='lightgreen', label='Buena Postura', linewidth=2)
    ax.plot(time_list, bad_time_list, color='lightcoral', label='Mala Postura', linewidth=2)
    ax.legend(loc='upper right')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Duración (s)')
    ax.set_title('Postura a lo largo del tiempo')
    plt.pause(0.001)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Posture Monitor with MediaPipe')
    parser.add_argument('--video', type=str, default=0, help='Path to the input video file. If not provided, the webcam will be used.')
    parser.add_argument('--offset-threshold', type=int, default=100, help='Threshold value for shoulder alignment.')
    parser.add_argument('--neck-angle-threshold', type=int, default=25, help='Threshold value for neck inclination angle.')
    parser.add_argument('--torso-angle-threshold', type=int, default=10, help='Threshold value for torso inclination angle.')
    parser.add_argument('--time-threshold', type=int, default=180, help='Time threshold for triggering a posture alert.')
    return parser.parse_args()

def main(video_path=None, offset_threshold=100, neck_angle_threshold=25, torso_angle_threshold=10, time_threshold=180):
    good_frames = 0
    bad_frames = 0
    total_good_time = 0
    total_bad_time = 0
    bad_posture_start_time = 0

    time_list = []
    good_time_list = []
    bad_time_list = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = {
        'green': (127, 255, 0),
        'red': (50, 50, 255),
        'white': (255, 255, 255),
        'light_green': (127, 233, 100)
    }

    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Inicializar la gráfica
    fig, ax = initialize_plot()

    while True:
        success, image = cap.read()
        if not success:
            print("No se pudo capturar el video.")
            break

        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if keypoints.pose_landmarks:
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            # Coordenadas de los puntos clave
            landmarks = {
                'left_shoulder': (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)),
                'right_shoulder': (int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)),
                'left_ear': (int(lm.landmark[lmPose.LEFT_EAR].x * w), int(lm.landmark[lmPose.LEFT_EAR].y * h)),
                'left_hip': (int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h))
            }

            # Dibujar puntos clave
            for point in landmarks.values():
                cv2.circle(image, point, 10, colors['white'], -1)

            # Conexiones entre puntos clave
            connections = [
                (lmPose.LEFT_SHOULDER, lmPose.RIGHT_SHOULDER),
                (lmPose.LEFT_SHOULDER, lmPose.LEFT_HIP),
                (lmPose.RIGHT_SHOULDER, lmPose.RIGHT_HIP),
                (lmPose.LEFT_SHOULDER, lmPose.LEFT_ELBOW),
                (lmPose.RIGHT_SHOULDER, lmPose.RIGHT_ELBOW),
                (lmPose.LEFT_ELBOW, lmPose.LEFT_WRIST),
                (lmPose.RIGHT_ELBOW, lmPose.RIGHT_WRIST),
                (lmPose.LEFT_HIP, lmPose.LEFT_KNEE),
                (lmPose.RIGHT_HIP, lmPose.RIGHT_KNEE),
                (lmPose.LEFT_KNEE, lmPose.LEFT_ANKLE),
                (lmPose.RIGHT_KNEE, lmPose.RIGHT_ANKLE),
                (lmPose.LEFT_HIP, lmPose.RIGHT_HIP),
            ]

            for connection in connections:
                start = (int(lm.landmark[connection[0]].x * w), int(lm.landmark[connection[0]].y * h))
                end = (int(lm.landmark[connection[1]].x * w), int(lm.landmark[connection[1]].y * h))
                cv2.line(image, start, end, colors['white'], 5)

            # Calcular distancias y ángulos
            offset = findDistance(*landmarks['left_shoulder'], *landmarks['right_shoulder'])
            neck_inclination = findAngle(*landmarks['left_shoulder'], *landmarks['left_ear'])
            torso_inclination = findAngle(*landmarks['left_hip'], *landmarks['left_shoulder'])

            # Mostrar las métricas de postura
            metrics_text = f'Offset: {int(offset)}\nÁngulo Cuello: {neck_inclination}\nÁngulo Torso: {torso_inclination}'
            draw_transparent_box(image, metrics_text, (10, 10))

            # Evaluar la postura
            if offset < offset_threshold and neck_inclination < neck_angle_threshold and torso_inclination < torso_angle_threshold:
                bad_frames = 0
                good_frames += 1
                total_good_time += (1 / fps)
                cv2.putText(image, 'Buena postura', (10, 210), font, 0.6, colors['light_green'], 2)
                bad_posture_start_time = 0  # Reiniciar el contador de mala postura
            else:
                good_frames = 0
                bad_frames += 1
                total_bad_time += (1 / fps)
                cv2.putText(image, 'Mala postura', (10, 210), font, 0.6, colors['red'], 2)

                if bad_posture_start_time == 0:
                    bad_posture_start_time = time.time()
                elif time.time() - bad_posture_start_time >= 10:
                    message = 'Llevas 10 seg con una mala postura'
                    textsize = cv2.getTextSize(message, font, 1, 2)[0]
                    textX = (image.shape[1] - textsize[0]) // 2
                    textY = (image.shape[0] + textsize[1]) // 2
                    cv2.putText(image, message, (textX, textY), font, 1, (0, 0, 255), 2)

            # Actualizar la lista de tiempos
            current_time = total_good_time + total_bad_time
            time_list.append(current_time)
            good_time_list.append(total_good_time)
            bad_time_list.append(total_bad_time)

            # Actualizar la gráfica
            update_line_plot(ax, good_time_list, bad_time_list, time_list)

            # Mostrar el tiempo total
            cv2.putText(image, f'Tiempo Buena postura: {round(total_good_time, 1)}s', (10, h - 40), font, 0.9, colors['green'], 2)
            cv2.putText(image, f'Tiempo Mala postura: {round(total_bad_time, 1)}s', (10, h - 20), font, 0.9, colors['red'], 2)

            # Consejos para mejorar la postura
            tips = "Consejos para mejorar la postura:\n1. Mantén los pies apoyados en el suelo.\n2. Alinea tus hombros con tus caderas.\n3. Toma descansos para estirarte."
            draw_transparent_box(image, tips, (10, 300))

            # Alerta si se necesita
            if total_bad_time > time_threshold:
                sendWarning()

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Desactivar modo interactivo
    plt.show()  # Mostrar la gráfica final

if __name__ == "__main__":
    args = parse_arguments()

    main(args.video, args.offset_threshold, args.neck_angle_threshold, args.torso_angle_threshold, args.time_threshold)
