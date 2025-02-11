import cv2
from ultralytics import solutions
from utils import convert_video_format
from utils import process_frame
from utils import release_resources
from utils import convert_video_format
from utils import draw_custom_counter


def main():
    input_video_path = "../input_videos/birds_3840_2160_25fps.mp4"
    output_video_path = "../output_videos/object_counting_output.avi"
    final_output_video_path = "../output_videos/object_counting_output.mp4"

    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define region points
    # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # For rectangle region counting
    region_points = [(2300, 20), (2300, 2100)]  # For line counting

    # Video writer
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Init ObjectCounter
    counter = solutions.ObjectCounter(
        show=True,
        region=region_points,
        model="yolo11n.pt",
        classes=[14],  # If you want to count specific classes i.e person and car with COCO pretrained model.
        show_in=True,  # Display in counts
        show_out=True,  # Display out counts    
        )

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = process_frame(im0, counter)

        # Get the in and out counts
        in_count = counter.in_count
        out_count = counter.out_count

        # Draw the custom counter box
        im0 = draw_custom_counter(im0, in_count, out_count)

        video_writer.write(im0)

    release_resources(cap, video_writer)

    # Convert output video from AVI to MP4
    convert_video_format(output_video_path, final_output_video_path)

if __name__ == "__main__":
    main()