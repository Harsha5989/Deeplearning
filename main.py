from Utils import read_video,save_video 
import cv2
from Trackers import Tracker
from Team_assigner import TeamAssigener
from Player_ball_assigner import PlayerBallAssigner
import torch
def main():
  video_path=read_video('C://Users//dimaag//Documents//Python Class//Projects//Football analysis//input_video//08fd33_4.mp4')



  def video_to_tensor(video_path, frame_size=None, normalize=True):
      """
      Converts a video into a PyTorch tensor.

      Args:
          video_path (str): Path to the video file.
          frame_size (tuple): (width, height) to resize frames. If None, no resizing is applied.
          normalize (bool): If True, normalize pixel values to [0, 1].

      Returns:
          torch.Tensor: Video tensor of shape (frames, channels, height, width).
      """
      cap = cv2.VideoCapture(video_path)
      frames = []

      while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
              break
          
          # Resize frame if needed
          if frame_size:
              frame = cv2.resize(frame, frame_size)
          
          # Convert BGR to RGB
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          
          # Convert frame to tensor
          frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
          frames.append(frame_tensor)
      
      cap.release()

      # Stack frames into a 4D tensor
      video_tensor = torch.stack(frames)  # (frames, channels, height, width)

      # Normalize to [0, 1]
      if normalize:
          video_tensor = video_tensor.float() / 255.0

      return video_tensor

  # Example usage
  video_frames = video_to_tensor(video_path, frame_size=(360, 640), normalize=True)
  print("Video tensor shape:", video_frames.shape)



















































































  tracker = Tracker('C://Users//dimaag//Documents//Python Class//Projects//Football analysis//Models//best.pt')
  tracks= tracker.get_object_track(video_frames,read_from_stub=True,stub_path='stubs/track_stub.pkl')
  # print(tracks['players'])


  tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

  # for track_id,player in tracks['players'][0].items():
  #   bbox = player['bbox']
  #   frame =video_frames[0]

  #   cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

  #   cv2.imwrite(f'Output_videos/cropped_image.jpg',cropped_image)
    
  #   break
  team_assigner = TeamAssigener()
  team_assigner.assign_team_color(video_frames[0],tracks['players'][0])

  for frame_num,player_track in enumerate(tracks['players']):
    for player_id,track in player_track.items():
      team = team_assigner.get_player_team(video_frames[frame_num],track['bbox'],player_id)

      tracks['players'][frame_num][player_id]['team'] = team
      tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

  player_assiner=PlayerBallAssigner()
  for frame_num,ball_track in enumerate(tracks['players']):
    ball_bbox = tracks['ball'][frame_num][1]['bbox']
    assigned_player=player_assiner.assign_ball_to_player(player_track,ball_bbox)

    if assigned_player !=-1:
      tracks['players'][frame_num][assigned_player]['has_ball']=True
      
        


  
  #Draw output_video 
  #Draw_object tracks
  output_video_frames = tracker.draw_annotations(video_frames,tracks)


  save_video(output_video_frames,'Output_videos/output.mp4')

if __name__=="__main__":
  main() 




