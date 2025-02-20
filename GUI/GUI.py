import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import ImageTk, Image
import threading
import sys
from pathlib import Path
import importlib
from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import supervision as sv
from more_itertools import chunked
import numpy as np

# Get the absolute path of the FOOTBALLAI_ENDPROJECT root directory
ROOT_DIR = Path(__file__).resolve().parent  # FOOTBALLAI_ENDPROJECT
sys.path.append(str(ROOT_DIR))

import config.paths
importlib.reload(config.paths)
from config.paths import paths

class GUI:
    def __init__(self):
        self.main_window = ctk.CTk()
        self.main_window.title('FootballAI_player_recognition')

        self.load_models()

        # Width and height of the window
        self.screen_width = self.main_window.winfo_screenwidth()
        self.screen_height = self.main_window.winfo_screenheight()

        # Load background image
        self.bg_image = cv2.imread(paths['HOME'] / 'GUI' / 'background.jpg')
        self.bg_image = cv2.cvtColor(self.bg_image, cv2.COLOR_BGR2RGB)
        self.bg_image = cv2.resize(self.bg_image, (self.screen_width, self.screen_height))
        self.bg_image = ImageTk.PhotoImage(Image.fromarray(self.bg_image))

        self.background = ctk.CTkLabel(self.main_window, image=self.bg_image, text='')
        self.background.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Scale the window
        window_width = int(self.screen_width * 0.75)
        window_height = int(self.screen_height * 0.9)
        self.main_window.geometry(f'{window_width}x{window_height}')

        # Create frames for video areas
        self.create_video_frames()

        # Buttons
        self.create_buttons()

        # Separate frame counters for each video
        self.frame_counter_left = 0
        self.frame_counter_right = 0

        self.main_window.mainloop()

    def load_models(self):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        MODEL_CHECKPOINT = "theButcher22/deta-swin-large"

        # Player detection model
        login(token=hf_token)
        self.player_det_model = AutoModelForObjectDetection.from_pretrained(MODEL_CHECKPOINT)
        self.player_det_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

        # Team dividing model
        siglip_model_preatrained = 'google/siglip-base-patch16-224'
        self.siglip_model = SiglipVisionModel.from_pretrained(siglip_model_preatrained)
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_preatrained)

        # Projection and clustering models
        self.umap = umap.UMAP(n_components=3)
        self.KMeans = KMeans(n_clusters=2)

    def train_models(self, video_path):
        SOURCE_VIDEO_PATH = video_path

        PLAYER_ID = 3
        STRIDE = 30

        frame_generator = sv.get_video_frames_generator(
            source_path=SOURCE_VIDEO_PATH, stride=STRIDE)

        crops = []
        for frame in tqdm(frame_generator, desc='collecting crops'):
            torch.cuda.empty_cache()  # Opróżnianie pamięci GPU co iterację
                
            val_inputs = self.player_det_processor(images=frame, return_tensors='pt')

            with torch.no_grad():
                val_outputs = self.player_det_model(pixel_values=val_inputs['pixel_values'])

            pred_boxes = val_outputs.pred_boxes[0].detach().cpu().numpy()  # (300, 4)
            image_height, image_width = frame.shape[:2]

            logits = val_outputs.logits[0].detach().cpu().numpy()
            scores = torch.softmax(torch.tensor(logits), dim=-1).numpy()

            confidence_scores = scores.max(axis=-1)
            class_ids = scores.argmax(axis=-1)

            confidence_threshold = 0.4
            high_conf_indices = confidence_scores > confidence_threshold

            filtered_boxes = pred_boxes[high_conf_indices]
            filtered_scores = confidence_scores[high_conf_indices]
            filtered_class_ids = class_ids[high_conf_indices]

            if filtered_boxes.shape[0] > 0:
                filtered_boxes[:, 0] -= (filtered_boxes[:, 2] / 2)
                filtered_boxes[:, 1] -= (filtered_boxes[:, 3] / 2)
                filtered_boxes[:, 2] += filtered_boxes[:, 0]
                filtered_boxes[:, 3] += filtered_boxes[:, 1]

                filtered_boxes[:, [0, 2]] *= image_width
                filtered_boxes[:, [1, 3]] *= image_height

                boxes_tensor = torch.tensor(filtered_boxes)
                scores_tensor = torch.tensor(filtered_scores)
                iou_threshold = 0.3
                nms_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)

                filtered_boxes = filtered_boxes[nms_indices.numpy()]
                filtered_scores = filtered_scores[nms_indices.numpy()]
                filtered_class_ids = filtered_class_ids[nms_indices.numpy()]

                detections = sv.Detections(
                    xyxy=filtered_boxes,
                    confidence=filtered_scores,
                    class_id=filtered_class_ids
                )

            detections = detections[detections.class_id == PLAYER_ID]
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
            crops += players_crops

        BATCH_SIZE = 32

        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = chunked(crops, BATCH_SIZE)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='embedding extraction'):
                inputs = self.siglip_processor(images=batch, return_tensors="pt")
                outputs = self.siglip_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        data = np.concatenate(data)

        projections = self.umap.fit_transform(data)
        self.KMeans.fit(projections)

    def process_video(self, video_path):
        id2label = {1: "ball", 2: "goalkeeper", 3: "player", 4: "referee"}
        label2id = {v: k for k, v in id2label.items()}

        SOURCE_VIDEO_PATH = video_path
        TARGET_VIDEO_PATH = f"{video_path}_processed.mp4"

        video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
        video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)

        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )

        tracker = sv.ByteTrack()
        tracker.reset()

        frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.player_det_model.to(device)

        with video_sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                torch.cuda.empty_cache()  # Opróżnianie pamięci GPU co iterację
                
                val_inputs = self.player_det_processor(images=frame, return_tensors='pt')
                val_inputs['pixel_values'] = val_inputs['pixel_values'].to(device)

                with torch.no_grad():
                    val_outputs = self.player_det_model(pixel_values=val_inputs['pixel_values'])

                pred_boxes = val_outputs.pred_boxes[0].detach().cpu().numpy()  # (300, 4)
                image_height, image_width = frame.shape[:2]

                logits = val_outputs.logits[0].detach().cpu().numpy()
                scores = torch.softmax(torch.tensor(logits), dim=-1).numpy()

                confidence_scores = scores.max(axis=-1)
                class_ids = scores.argmax(axis=-1)

                confidence_threshold = 0.4
                high_conf_indices = confidence_scores > confidence_threshold

                if high_conf_indices.sum() == 0:
                    video_sink.write_frame(frame)  # Jeśli nie wykryto obiektów, zapisz oryginalną klatkę
                    continue

                filtered_boxes = pred_boxes[high_conf_indices]
                filtered_scores = confidence_scores[high_conf_indices]
                filtered_class_ids = class_ids[high_conf_indices]

                if filtered_boxes.shape[0] > 0:
                    filtered_boxes[:, 0] -= (filtered_boxes[:, 2] / 2)
                    filtered_boxes[:, 1] -= (filtered_boxes[:, 3] / 2)
                    filtered_boxes[:, 2] += filtered_boxes[:, 0]
                    filtered_boxes[:, 3] += filtered_boxes[:, 1]

                    filtered_boxes[:, [0, 2]] *= image_width
                    filtered_boxes[:, [1, 3]] *= image_height

                    boxes_tensor = torch.tensor(filtered_boxes)
                    scores_tensor = torch.tensor(filtered_scores)
                    iou_threshold = 0.3
                    nms_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)

                    filtered_boxes = filtered_boxes[nms_indices.numpy()]
                    filtered_scores = filtered_scores[nms_indices.numpy()]
                    filtered_class_ids = filtered_class_ids[nms_indices.numpy()]

                    detections = sv.Detections(
                        xyxy=filtered_boxes,
                        confidence=filtered_scores,
                        class_id=filtered_class_ids
                    )

                    ball_detections = detections[np.array(detections.class_id) == label2id['ball']]
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

                    all_detections = detections[np.array(detections.class_id) != label2id['ball']]
                    all_detections = tracker.update_with_detections(detections=all_detections)

                    goalkeepers_detections = all_detections[all_detections.class_id == label2id['goalkeeper']]
                    players_detections = all_detections[all_detections.class_id == label2id['player']]
                    referees_detections = all_detections[all_detections.class_id == label2id['referee']]

                    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
                    BATCH_SIZE = 32

                    crops = [sv.cv2_to_pillow(crop) for crop in players_crops]
                    batches = chunked(crops, BATCH_SIZE)
                    data = []
                    with torch.no_grad():
                        for batch in batches:
                            inputs = self.siglip_processor(images=batch, return_tensors="pt")
                            outputs = self.siglip_model(**inputs)
                            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                            data.append(embeddings)
                    data = np.concatenate(data)
                    projections = self.umap.transform(data)
                    players_detections.class_id = self.KMeans.predict(projections)

                    goalkeepers_detections.class_id = self.resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
                    referees_detections.class_id -= 2
                    
                    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

                    labels = [
                        f"#{tracker_id}"
                        for tracker_id
                        in all_detections.tracker_id
                    ]

                    all_detections.class_id = all_detections.class_id.astype(int)

                    annotated_frame = frame.copy()
                    annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
                    annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)

                    video_sink.write_frame(annotated_frame)
        
        return TARGET_VIDEO_PATH

    def resolve_goalkeepers_team_id(self, players, goalkeepers):
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
        team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
        goalkeepers_team_id = []
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

        return np.array(goalkeepers_team_id)
    
    def create_video_frames(self):
        """Create left and right frames for original and processed video."""
        self.left_frame = ctk.CTkFrame(self.main_window, width=300, height=500, corner_radius=10)
        self.left_frame.place(relx=0.05, rely=0.15, relwidth=0.4, relheight=0.75)

        self.right_frame = ctk.CTkFrame(self.main_window, width=300, height=500, corner_radius=10)
        self.right_frame.place(relx=0.55, rely=0.15, relwidth=0.4, relheight=0.75)

        self.left_video_label = ctk.CTkLabel(self.left_frame, text='', width=300, height=300)
        self.left_video_label.pack(expand=True, fill='both')

        self.right_video_label = ctk.CTkLabel(self.right_frame, text='', width=300, height=300)
        self.right_video_label.pack(expand=True, fill='both')

    def create_buttons(self):
        """Create buttons for the interface."""
        self.load_video_img = cv2.imread(paths['HOME'] / 'GUI' / 'load_video.png')
        self.load_video_img = cv2.cvtColor(self.load_video_img, cv2.COLOR_BGR2RGB)
        self.load_video_img = cv2.resize(self.load_video_img, (100, 100))
        self.load_video_img = ImageTk.PhotoImage(Image.fromarray(self.load_video_img))

        self.buttonx = tk.Button(
            self.main_window,
            image=self.load_video_img,
            text='',
            highlightthickness=0,
            command=self.load_video
        )
        self.buttonx.place(relx=0.47, rely=0.04)  # Positioning with relative coordinates

    def load_video(self):
        """Load and play the original video."""
        file_path = tk.filedialog.askopenfilename(title='Select the Video')
        if file_path:
            self.train_models(file_path)
            processed_video_path = self.process_video(file_path)

            cap1 = cv2.VideoCapture(file_path)
            cap2 = cv2.VideoCapture(processed_video_path)
            threading.Thread(target=self.update_frame, args=(cap1, self.left_video_label, 'left')).start()
            threading.Thread(target=self.update_frame, args=(cap2, self.right_video_label, 'right')).start()

    def load_processed_video(self):
        """Load and play the processed video."""
        file_path = '0bfacc_0.mp4'
        if file_path:
            cap = cv2.VideoCapture(file_path)
            threading.Thread(target=self.update_frame, args=(cap, self.right_video_label, True, 'right')).start()

    def update_frame(self, cap, label, side):
        """Class method to update video frame in the given label, displaying every second frame."""
        ret, frame = cap.read()
        if ret:
            if side == 'left':
                self.frame_counter_left += 1
                if self.frame_counter_left % 2 != 0:  # Skip every second frame for the left video
                    label.after(15, lambda: self.update_frame(cap, label, side))
                    return
            elif side == 'right':
                self.frame_counter_right += 1
                if self.frame_counter_right % 2 != 0:  # Skip every second frame for the right video
                    label.after(15, lambda: self.update_frame(cap, label, side))
                    return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_AREA)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            label.configure(image=img)
            label.image = img
            label.after(10, lambda: self.update_frame(cap, label, side))
        else:
            cap.release()
            if side == 'left':
                self.frame_counter_left = 0
            elif side == 'right':
                self.frame_counter_right = 0
