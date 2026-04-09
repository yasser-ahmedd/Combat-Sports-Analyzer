import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import mediapipe as mp  # For pose estimation (using MediaPipe as an example)

# Define your class mappings
class_map = {'punch': 0, 'kick': 1, 'downtime': 2}

# Set up MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

class PunchingBagDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_frames=16):
        self.data_dir = data_dir
        self.transform = transform
        self.max_frames = max_frames
        self.data = []
        
        # Collect all video files and labels based on folders
        for class_name, class_idx in class_map.items():
            class_folder = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_folder):
                if file_name.endswith('.mov'): #When expaning Dataset you will most likely need to check for more than .mov
                    self.data.append((os.path.join(class_folder, file_name), class_idx))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        
        # Load the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        pose_keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize or preprocess frame
            # if self.transform:
            #     frame = self.transform(frame)

            frame_resized = cv2.resize(frame, (64,64))
            frame_tensor = torch.tensor(frame_resized).permute(2,0,1) #(C,H,W)

            frames.append(frame_tensor)
            
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform pose estimation
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract keypoints (x, y coordinates of each landmark)
                keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
                pose_keypoints.append(torch.tensor(keypoints))
            else:
                # If no pose detected, append a placeholder or skip frame
                pose_keypoints.append(torch.zeros((33,2)))
        
        cap.release()


        # Pad or truncate to ensure consistent number of frames
        # num_frames = len(pose_keypoints)
        # if num_frames < self.max_frames:
        #     # Pad with empty (zero) keypoints
        #     pad_frames = [torch.zeros((33, 2)) for _ in range(self.max_frames - num_frames)]
        #     pose_keypoints.extend(pad_frames)
        # elif num_frames > self.max_frames:
        #     # Truncate the keypoints
        #     pose_keypoints = pose_keypoints[:self.max_frames]

        # Ensure consistent frame and keypoint sequence lengths
        frames = frames[:self.max_frames] + [torch.zeros((3, 64, 64))] * max(0, self.max_frames - len(frames))
        pose_keypoints = pose_keypoints[:self.max_frames] + [torch.zeros((33, 2))] * max(0, self.max_frames - len(pose_keypoints))

        # Stack into tensors
        frames_tensor = torch.stack(frames)  # Shape: (max_frames, 3, 64, 64)
        pose_tensor = torch.stack(pose_keypoints)  # Shape: (max_frames, 33, 2)


        
        return frames_tensor, pose_tensor, label
    

#     def display_random_video_with_pose(self):
#         video_path, label = random.choice(self.data)
#         cap = cv2.VideoCapture(video_path)
        
#         print(f"Displaying video: {video_path} (Label: {label})")

#         mp_drawing = mp.solutions.drawing_utils
#         mp_drawing_styles = mp.solutions.drawing_styles

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(frame_rgb)
            
#             if results.pose_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame,
#                     results.pose_landmarks,
#                     mp_pose.POSE_CONNECTIONS,
#                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                 )
            
#             cv2.imshow("Pose Estimation", frame)
            
#             # Exit the display by pressing 'q'
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()

# # Usage
# data_directory = './dataset'
# dataset = PunchingBagDataset(data_directory)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
# for i in range(10):
#     dataset.display_random_video_with_pose()
