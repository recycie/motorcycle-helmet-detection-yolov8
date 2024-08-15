import cv2

def save_frames(video_path, output_folder, frame_rate=5):
    frame_number = 0

    for vid in video_path:
        # Open the video file
        video_capture = cv2.VideoCapture(vid)

        # Get the frames per second (fps) of the video
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        # Calculate the interval to capture frames based on the desired frame rate
        interval = int(round(fps / frame_rate))


        while True:
            # Read the next frame from the video
            ret, frame = video_capture.read()

            if not ret:
                # Break the loop if the video has ended
                break

            if frame_number % interval == 0:
                # Save the frame as an image
                output_path = f"{output_folder}/frame_{frame_number}.jpg"
                print(output_path)
                cv2.imwrite(output_path, frame)

            # Increment the frame number
            frame_number += 1

        # Release the video capture object
        video_capture.release()

# Example usage
video_path = ["data/bru06.mp4", "data/bru07.mp4", "data/bru08.mp4"]
output_folder = "output/bru06"

save_frames(video_path, output_folder, frame_rate=5)
