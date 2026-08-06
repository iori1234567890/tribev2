from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

df = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events=df)
print(preds.shape)  # (n_timesteps, n_vertices)