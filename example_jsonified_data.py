from json_schema_test import parse_into_workout_schema
from transcription import transcribe_audio
import json

# transcribed_audio = transcribe_audio()
# print(str(transcribed_audio.text))
# parsed_workout = parse_into_workout_schema(str(transcribed_audio.text))
parsed_workout = parse_into_workout_schema("I bench pressed 100 pounds for 3 sets of 10 reps and then I squatted 50 pounds for 5 sets of 5 reps.")
print(parsed_workout)


with open("workout.json", "w") as f:
    if parsed_workout:
        f.write(parsed_workout)

saved_workout = open("workout.json", "r")
loaded_json = json.load(saved_workout)
print(loaded_json)
