import parselmouth
import os
from parselmouth.praat import call
# import tgt

def count_annotations(text_grid, annotation):
    nr_occurences = 0
    for _ in text_grid.tiers[0].get_annotations_with_text(annotation):
        nr_occurences += 1
    return nr_occurences


def extract_info(file_name):
    sound = parselmouth.Sound(file_name)
    duration = call(sound, "Get total duration")
    silences = call(sound, "To TextGrid (silences)", 100, 0.0, -25, 0.1, 0.1, "silent", "sounding")


    pitch_tier = call(sound, "To Pitch", 0.0, 75, 600)
    pitch_mean = call(pitch_tier, "Get mean", 0.0, 0.0, "Hertz")
    pitch_max = call(pitch_tier, "Get maximum", 0, 0, "Hertz", "Parabolic")
    pitch_min = call(pitch_tier, "Get minimum", 0, 0, "Hertz", "Parabolic")
    pitch_slope_mean = pitch_tier.get_mean_absolute_slope()
    grid = silences.to_tgt()

    nr_silences = count_annotations(grid, "silent")
    nr_sounds = count_annotations(grid, "sounding")

    # print("Length of sound:", duration, "sec")
    # print("Nr. of silences:", nr_silences)  
    # print("Nr. of sounding segments:", nr_sounds
    # print("Max pitch:",pitch_max, "Hz") 
    # print("Min pitch:",pitch_min, "Hz") 
    # print("Pitch mean:",pitch_mean, "Hz"
    # print("Pitch mean absolute slope:",pitch_slope_mean,"Hz"

    return (duration, nr_silences, nr_sounds, pitch_mean, pitch_max, pitch_min, pitch_slope_mean)


directory = os.path.dirname(os.path.realpath(__file__))
data = []
for entry in os.scandir(directory):
    if entry.path.endswith(".wav") and entry.is_file():
        print(f"Configuring: {entry.name}")
        data.append(extract_info(entry.path))

[print(tup) for tup in data]
