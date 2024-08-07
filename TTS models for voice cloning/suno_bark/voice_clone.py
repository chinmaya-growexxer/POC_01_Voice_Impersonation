text = "Growexx offers Data Science and AI related services. There are more than 15 services we offer. If you want to know about any specific service let me know."

from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
import scipy

config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="bark/", eval=True)

# cloning a speaker.
# It assumes that you have a speaker file in `bark_voices/speaker_n/speaker.wav` or `bark_voices/speaker_n/speaker.npz`
output_dict = model.synthesize(text, config, speaker_id="V", voice_dirs="bark_voices/",language="en")

# write the file to disk .wav file
sample_rate = 24000
scipy.io.wavfile.write("vikas_cloned.wav", rate=sample_rate, data=output_dict["wav"])
