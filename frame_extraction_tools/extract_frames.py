import cv2
import os

def batch_extract_frames_recursive():
    input_folder = "videos"
    output_root = "extracted_images"
    frame_interval_seconds = 4

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    video_paths = []
    
    # os.walk acts like a drill, going through every single sub-folder automatically
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # The .lower() fix ensures we catch .MOV, .mov, .MP4, etc.
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_paths.append(os.path.join(root, file))

    if not video_paths:
        print(f"No videos found inside '{input_folder}' or any of its sub-folders.")
        return

    print(f"Found {len(video_paths)} videos deeply nested. Starting extraction...")

    for video_path in video_paths:
        # 1. Figure out exactly which sub-folders this video is buried in
        # Example: if path is "videos/sdd_videos/bookstore/video0/video.mov"
        # folder_structure becomes "bookstore/video0"
        folder_structure = os.path.relpath(os.path.dirname(video_path), input_folder)
        
        video_name = os.path.basename(video_path).split('.')[0]
        
        # 2. Create a unique prefix to prevent overwriting duplicate "video.mov" files
        if folder_structure == ".":
            unique_prefix = video_name
        else:
            # Replace folder slashes with underscores (e.g., "bookstore_video0_video")
            safe_folder_name = folder_structure.replace(os.sep, "_").replace("/", "_")
            unique_prefix = f"{safe_folder_name}_{video_name}"

        # 3. Create the unique output folder
        video_output_folder = os.path.join(output_root, unique_prefix)
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)

        # 4. Extract the frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0 or fps is None:
            fps = 30.0 
            
        frame_skip = int(fps * frame_interval_seconds)
        
        count = 0
        saved_count = 0
        
        print(f"Processing: {unique_prefix} (Extracting 1 frame every {frame_interval_seconds}s)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_skip == 0:
                # Name the file using the unique prefix
                filename = os.path.join(video_output_folder, f"{unique_prefix}_frame_{saved_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
                
            count += 1
            
        cap.release()
        print(f"  -> Saved {saved_count} frames to {video_output_folder}")

    print("\nExtraction Complete! You can now drag these folders into Roboflow.")

if __name__ == '__main__':
    batch_extract_frames_recursive()