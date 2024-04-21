
cur_detector = "ivdetect"
slice_level: bool = (cur_detector == "deepwukong")

data_path = {
    "reveal": "function/explain_reveal.json",
    "devign": "function/explain_devign.json",
    "ivdetect": "function/explain_ivdetect.json",
    "deepwukong": "slice/explain_deepwukong.json"
}