from roboflow import Roboflow

rf = Roboflow(api_key="Smj9OWWRBoljb8blaiqQ")
project = rf.workspace("cctv-detection-njw1h").project("fire-dataset-for-yolov8-deupo")
version = project.version(1)
dataset = version.download("yolov8")

print("Dataset downloaded!")
print(f"Location: {dataset.location}")
