from Utils import read_video,save_video 
import cv2
from Trackers import Tracker
from Team_assigner import TeamAssigener
from Player_ball_assigner import PlayerBallAssigner
import numpy as np
from Camera_movement_estimator import CameraMovementEstimator

def main():
  video_frames=read_video('C://Users//dimaag//Documents//Python Class//Projects//Football analysis//input_video//08fd33_4.mp4')

  tracker = Tracker('C://Users//dimaag//Documents//Python Class//Projects//Football analysis//Models//best.pt')
  tracks= tracker.get_object_track(video_frames,read_from_stub=True,stub_path='stubs/track_stub.pkl')
  # print(tracks['players'])

  tracker.add_position_to_tracks(tracks)

  camera_movement_estimator = CameraMovementEstimator(video_frames[0])
  camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement_stub.pkl')

  camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

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
  team_ball_control =[]


  for frame_num,player_track in enumerate(tracks['players']):
    ball_bbox = tracks['ball'][frame_num][1]['bbox']
    assigned_player=player_assiner.assign_ball_to_player(player_track,ball_bbox)
  
    if assigned_player !=-1:
      tracks['players'][frame_num][assigned_player]['has_ball'] = True
      team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
    else:
      team_ball_control.append(team_ball_control[-1])
  team_ball_control = np.array(team_ball_control)

      
        
  #Draw output_video 
  #Draw_object tracks
  output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control)

  output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

  save_video(output_video_frames,'Output_videos/output.mp4')

if __name__=="__main__":
  main() 

