import librosa
import soundfile 
import scipy
import numpy as np
from utils import plotWaveforms
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path
from typing import TypedDict, Optional, Tuple


# ╭───────────────────────────────────────────────────────────────────────────╮
# │                      Load and store audio functions                       │
# ╰───────────────────────────────────────────────────────────────────────────╯


def store_audio(path: str | Path, 
                audio_data: np.ndarray, 
                sample_rate: int = 44100) -> None:
    soundfile.write(str(path), audio_data, sample_rate)





def load_audio(file_path: str | Path, sampling_rate: int | None = None, 
               mono: bool = False) -> tuple[np.ndarray, float | int]:
    """ Loads an audio file as a floating point time series.  
        sampling_rate=None ensures librosa keeps the file's original 
        sample rate """
    audio_data, actual_sr = librosa.load(str(file_path), 
                                         sr=sampling_rate, mono=mono)
    return audio_data, actual_sr





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                               Data Classes                                │
# ╰───────────────────────────────────────────────────────────────────────────╯


@dataclass
class DataInfo():
    rate:                   int
    validation_section1:    tuple[int, int]
    validation_section2:    tuple[int, int]
    blip_section:           tuple[int, int]
    blip_locations:         tuple[int, int]
    background_interval:    tuple[int, int]





@dataclass
class CaptureState:
    # Delay/alignment info
    delay:                         int   | None = None
    safety_factor:                 int   | None = None
    lookback:                      int   | None = None
    lookahead:                     int   | None = None
    delta_samples:                 int   | None = None
    trimmed_samples:               int   | None = None
    trimmed_seconds:               float | None = None
    input_gain:                    float | None = None
    output_gain:                   float | None = None
    input_val_ESR:                 float | None = None
    output_val_ESR:                float | None = None

    # Audio file checks
    input_samplingrate_unchanged:  bool  | None = None
    input_channels_preserved:      bool  | None = None
    output_samplingrate_unchanged: bool  | None = None
    output_channels_preserved:     bool  | None = None
    similar_lengths:               bool  | None = None
    similar_input_val_sections:    bool  | None = None
    similar_output_val_sections:   bool  | None = None

    # Match quality
    able_to_match:                 bool  | None = None
    match_after_lookback:          bool  | None = None
    match_before_lookahead:        bool  | None = None






# ╭───────────────────────────────────────────────────────────────────────────╮
# │         Main class containing audio and their state information           │
# ╰───────────────────────────────────────────────────────────────────────────╯


class AudioFile():
    def __init__(self, path: str | Path, 
                 sampling_rate: int | None = None, 
                 mono: bool = False):
        self.path = Path(path)
        info = soundfile.info(str(self.path))

        self.resampling = False
        if sampling_rate is not None and info.samplerate != sampling_rate:
            self.resampling = True
            # (info.samplerate, sampling_rate)

        self.mono_conversion = True if info.channels != 1 and mono is True \
                               else False

        self.waveform, self.sampling_rate = librosa.load(
            str(self.path), sr=sampling_rate, mono=mono)
        
        if self.waveform.ndim == 1:
            self.waveform = self._handle_dims(self.waveform)
        

    def _handle_dims(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            return y[np.newaxis, :]
        if y.ndim == 2:
            return y
        raise ValueError("Audio must be 1D or 2D (channels, samples)")





class CapturePair():

    def __init__(self, 
                 data_info:   DataInfo, 
                 path_output: str | Path,
                 path_input:  str | Path | None = None,
                 input_file:  AudioFile  | None = None,
                 input_mono:  bool              = False, 
                 output_mono: bool              = False):

        self.path_output = path_output
        self.data_info = data_info
        
        self.input_file = AudioFile(path_input, data_info.rate, mono=input_mono) \
                      if input_file is None else input_file
        self.output_file = AudioFile(path_output, data_info.rate, 
                                mono=output_mono)
        
        self.state = CaptureState(
            input_samplingrate_unchanged = not self.input_file.resampling,
            input_channels_preserved     = not self.input_file.mono_conversion,
            output_samplingrate_unchanged= not self.output_file.resampling,
            output_channels_preserved    = not self.output_file.mono_conversion
        )

        self._checkAudioLengths()


    def _checkAudioLengths(self, 
                           min_delta_seconds = -1, 
                           max_delta_seconds = 10):
        if self.input_file.sampling_rate != self.data_info.rate:
            raise ValueError("Input sampling rate must match data_info rate ",
                  f"{self.input_file.sampling_rate} vs {self.data_info.rate}")

        _, len_input  = self.input_file.waveform.shape
        _, len_output = self.output_file.waveform.shape

        delta_samples = len_output - len_input
        delta_seconds = delta_samples/self.input_file.sampling_rate
        
        passed = min_delta_seconds < delta_seconds < max_delta_seconds 
        self.state.similar_lengths = passed
        self.state.delta_samples   = delta_samples


    def update_state_delay(self, delay:   int | None, 
                           lookback:      int, 
                           lookahead:     int, 
                           safety_factor: int) -> None:
        self.state.delay         = delay
        self.state.safety_factor = safety_factor
        self.state.lookback      = lookback
        self.state.lookahead     = lookahead

        self.state.able_to_match = delay is not None

        delay = np.nan if delay is None else delay
        self.state.match_after_lookback   = delay > -lookback 
        self.state.match_before_lookahead = delay < lookahead


    def update_state_gain(self, input_gain: float, output_gain: float):
        self.state.input_gain  = input_gain
        self.state.output_gain = output_gain


    def update_state_crop(self, trimmed: int, sampling_rate: int) -> None:
        self.state.trimmed_samples = trimmed
        self.state.trimmed_seconds = trimmed / sampling_rate

    
    def update_val_check(self,
                         input_check:    bool  | None = None, 
                         input_val_esr:  float | None = None,  
                         output_check:   bool  | None = None,
                         output_val_esr: float | None = None):
        if input_check is not None:
            self.state.similar_input_val_sections = input_check
            self.state.input_val_ESR = float(input_val_esr)
        if output_check is not None:
            self.state.similar_output_val_sections = output_check
            self.state.output_val_ESR = float(output_val_esr)
        

    def plot_alignment(self, show: bool = True):
        delay = self.state.delay
        numeric_delay = np.nan if delay is None else delay

        blip_location = self.data_info.blip_locations[0]
        fig, ax = plotWaveforms(self.output_file.waveform, 
                                start_at=blip_location - self.state.lookback, 
                                end_at=blip_location + self.state.lookahead,
                                vlines=[(blip_location + numeric_delay, 
                                         f'match at {numeric_delay} samples'),
                                        (blip_location, 'ideal')])

        ax.set_title("Summary of synchronization")
        if show:
            plt.show()
        return fig, ax


    def print_state(self) -> None:
        spacer_before = ['input_samplingrate_unchanged', 'able_to_match']
        print("-~" * 26)
        print(f"State of {self.path_output}:")
        
        for key, value in asdict(self.state).items():
            if key in spacer_before: print()
            display_val = 'Not checked' if value is None else value

            is_float = isinstance(display_val, float)
            formatted = f"{display_val:.7e}" if is_float else str(display_val)

            print(f"  - {key:<30} {formatted}")
        print("-~" * 26)





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                            Preprocessing Class                            │
# ╰───────────────────────────────────────────────────────────────────────────╯


class Preprocessing():

    def __init__(self,
                 lookback:              int   = 1_000,                          # How much we will search in the past
                 lookahead:             int   = 10_000,                         # and in the future (changed from NAM code, it was confusing for me)
                 absolute_threshold:    float = 0.0003,
                 relative_threshold:    float = 0.001,
                 validation_threshold:  float = 0.01,
                 in_gain_db:            float = 0.0,
                 out_gain_db:           float = 0.0
                 ):

        self.lookback  = lookback
        self.lookahead = lookahead
        self.absolute_threshold   = absolute_threshold
        self.relative_threshold   = relative_threshold
        self.validation_threshold = validation_threshold
        self.in_gain  = 10**(in_gain_db/20)
        self.out_gain = 10**(out_gain_db/20)
        
        self._checkArgs()


    def _checkArgs(self):
        if not (0 < self.lookback < self.lookahead): 
            raise ValueError("Args must be 0 < lookback < lookahead")
        if not(0<self.absolute_threshold<1 and 0<self.relative_threshold<1):
            raise ValueError("Thresholds must be between 0 and 1")
    

    def compute_cross_correlation(self, y_input: np.ndarray, y_output: np.ndarray):
        self.assert_ndarray(y_input, y_output )
        (_, len_in), (_, len_out)  = y_input.shape,  y_output.shape
        y_input, y_output = y_input.mean(axis=0), y_output.mean(axis=0)

        corr = scipy.signal.correlate(y_input, y_output, mode='full')
        lags = scipy.signal.correlation_lags(len_in, len_out, mode='full')

        lag = lags[np.argmax(corr)]
        return lag


    def compute_delay(self, 
                      y:    np.ndarray, 
                      info: DataInfo, 
                      safety_factor: int = 1) -> int | None:
        self.assert_ndarray(y)
        y = y.mean(axis=0)                                                      # Now the array is 1D

        y_background = y[info.background_interval[0]:info.background_interval[1]]
        background_level = np.abs(y_background).max()
        
        threshold = max(background_level + self.absolute_threshold,
                        (1.0 + self.relative_threshold) * background_level)     # maybe we can use some percentile

        y_scans = [y[location-self.lookback:location+self.lookahead] 
                   for location in info.blip_locations]
        y_scan_average = np.stack(y_scans).mean(axis=0)
        above_threshold, = np.where(np.abs(y_scan_average) > threshold)

        if len(above_threshold) == 0: return None
        
        delay = above_threshold[0] - self.lookback - safety_factor
        return delay
    

    def align(self, y_input: np.ndarray, y_output: np.ndarray, 
              delay: int | None) -> tuple[np.ndarray, np.ndarray]:
        self.assert_ndarray(y_input, y_output)

        if delay is None: 
            warnings.warn("Not able to match, returning output not delayed")
            delay = 0

        if delay > 0:                                                           # Output is delayed, drop 
            y_output = y_output [:, delay:]                                     # the start of the output
        elif delay < 0:                                                         # Input is delayed (rare)
            y_input  = y_input[:, -delay:]                                      # drop the start of the input

        return y_input, y_output


    def crop(self, y_input: np.ndarray, y_output: np.ndarray
         ) -> tuple[tuple[np.ndarray, np.ndarray], int]:
        self.assert_ndarray(y_input, y_output)

        _, len_input = y_input.shape
        _, len_output = y_output.shape
        min_len = min(len_input, len_output)
        max_len = max(len_input, len_output)

        y_input = y_input[:, :min_len]
        y_output = y_output[:, :min_len]
        
        trimmed = max_len - min_len        
        return (y_input, y_output), trimmed
    

    def apply_gain(self, y_input: np.ndarray, y_output: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
        self.assert_ndarray(y_input, y_output)
        y_input  = self.in_gain  * y_input
        y_output = self.out_gain * y_output

        return y_input, y_output
    

    def get_validation_segments(self, y: np.ndarray, info: DataInfo, 
                                offset: int, equal_length: bool = False
                                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the two validation segments, accounting for the truncation offsets.
        When equal_length==True ensures that both segments are trimmed equally. 
        """
        self.assert_ndarray(y)
        
        v1_start = info.validation_section1[0] - offset                         # Shifted coordinates based on how many 
        v1_end   = info.validation_section1[1] - offset                         # samples were dropped during align()
        v2_start = info.validation_section2[0] - offset
        v2_end   = info.validation_section2[1] - offset        
        
        cut_from_start = -v1_start if v1_start < 0 else 0                       # If we will cut the first validation section, we   
        v2_start = v2_start + cut_from_start if equal_length else v2_start      # apply the exact same cut to the second section

        n = y.shape[-1]
        v1_start = min(max(0, v1_start), n)                                     # Safe bounds check
        v1_end   = min(max(0, v1_end), n)    
        v2_start = min(max(0, v2_start), n)
        v2_end   = min(max(0, v2_end), n)

        y_val1 = y[:, v1_start:v1_end]
        y_val2 = y[:, v2_start:v2_end]
        
        min_len = min(y_val1.shape[-1], y_val2.shape[-1])
        min_len = min_len if equal_length else None                            # Truncate to ensure equal length
        return y_val1[:, :min_len], y_val2[:, :min_len]
    

    def check_val_allignment(self, y: np.ndarray, info: DataInfo, 
                             offset: int):
        y_val1, y_val2 = self.get_validation_segments(y, info, offset, True)
        esr = self._esr(y_val1, y_val2)
        below_threshold = bool(esr < self.validation_threshold)

        return below_threshold, esr


    def _esr(self, y1: np.ndarray, y2: np.ndarray) -> float:
        error_squared = np.sum((y1 - y2)**2)
        energy        = np.sum(y2**2)
        esr = error_squared/energy
        return esr


    def split_train_val(self, y: np.ndarray, info: DataInfo, offset: int):
        """ Returns the training and the longest validation segment. """
        v1_end   = max(0, info.validation_section1[1] - offset)
        v2_start = min(y.shape[-1], max(0, info.validation_section2[0]-offset))
        
        y_train = y[:, v1_end : v2_start]                                       # Train is everything between V1 and V2

        y_val1, y_val2 = self.get_validation_segments(y, info, offset)          # For the actual validation set 
        y_val = y_val1 if y_val1.shape[-1] > y_val2.shape[-1] else y_val2       # we use the longer segment

        return y_train, y_val


    @staticmethod
    def get_normalization_factor(y: np.ndarray) -> np.ndarray:
        return 1/(np.max(np.abs(y)) + 1e-9)


    def assert_ndarray(self, *args: np.ndarray) -> None:
        for arg in args:
            if not isinstance(arg, np.ndarray) or arg.ndim != 2: 
                raise ValueError("Value should be a ndarray with 2 dims")


    def __call__(self, 
                 capture_pair: 'CapturePair', 
                 safety_factor: int = 1
                 ) -> tuple[np.ndarray, np.ndarray]:
        y_input  = capture_pair.input_file.waveform
        y_output = capture_pair.output_file.waveform
        info = capture_pair.data_info

        delay = self.compute_delay(y_output, info, safety_factor)
        capture_pair.update_state_delay(delay, self.lookback, self.lookahead, 
                                        safety_factor)
        
        y_input, y_output = self.apply_gain(y_input, y_output)
        capture_pair.update_state_gain(self.in_gain, self.out_gain)

        y_input, y_output = self.align(y_input, y_output, delay)

        (y_input, y_output), trimmed = self.crop(y_input, y_output)
        capture_pair.update_state_crop(trimmed, 
                                       capture_pair.input_file.sampling_rate)

        delay_val = 0 if delay is None else delay
        in_offset = -delay_val if delay_val < 0 else 0
        out_offset = delay_val if delay_val > 0 else 0

        input_check  = self.check_val_allignment(y_input, info, in_offset)
        output_check = self.check_val_allignment(y_output, info, out_offset)
        capture_pair.update_val_check(*input_check, *output_check)

        y_train, y_val = self.split_train_val(y_input, info, in_offset)
        print(f"y_train: {y_train.shape}")
        print(f"y_train: {y_val.shape}")

        y_train, y_val = self.split_train_val(y_output, info, out_offset)
        print(f"y_train: {y_train.shape}")
        print(f"y_train: {y_val.shape}")
        
        return y_input, y_output











"""
h0 = 1
h1 = A_1 + B_1 x_1
h2 = A_2 h1                + B_2 x_2
   = A_2 (A_1 + B_1 x_1)   + B_2 x_2
   = A_2 A_1 + A_2 B_1 x_1 + B_2 x_2
h3 = A_3 h2                                       + B_3 x_3
   = A_3 (A_2 A_1 + A_2 B_1 x_1 + B_2 x_2)        + B_3 x_3
   = A_3 A_2 A_1 + A_3 A_2 B_1 x_1 + A_3 B_2 x_2  + B_3 x_3
"""

