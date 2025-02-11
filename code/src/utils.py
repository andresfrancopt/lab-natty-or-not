import subprocess
import cv2

def load_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

def process_frame(frame, counter):
    return counter.count(frame)

def initialize_video_capture(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    return cap

def release_resources(cap, video_writer):
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def convert_video_format(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-crf', '18',
        '-preset', 'slow',
        output_path
    ]
    subprocess.run(command, check=True)

def draw_custom_counter(im0, in_count, out_count):
    # Define the position and size of the counter box
    box_x, box_y, box_w, box_h = 3540, 20, 230, 120

    # Draw the background rectangle
    cv2.rectangle(im0, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)

    # Define the text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    font_color = (255, 255, 255)
    thickness = 3

    # Draw the in count text
    in_text = f"In: {in_count}"
    cv2.putText(im0, in_text, (box_x + 20, box_y + 40), font, font_scale, font_color, thickness)

    # Draw the out count text
    out_text = f"Out: {out_count}"
    cv2.putText(im0, out_text, (box_x + 20, box_y + 100), font, font_scale, font_color, thickness)

    return im0