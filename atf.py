import numpy as np
import librosa
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import threading

class AudioFunction:
    """
    A class that converts an audio file into a mathematical function
    that can be sampled at any time point, with playback capabilities.
    """
    
    def __init__(self, audio_file_path, channel=0):
        """
        Initialize the AudioFunction with an audio file.
        
        Args:
            audio_file_path (str): Path to the audio file
            channel (int): Channel to use for multi-channel audio (0 for left, 1 for right, etc.)
        """
        # Load the audio file
        self.audio_data, self.sample_rate = librosa.load(audio_file_path, sr=None, mono=False)
        
        # Handle multi-channel audio
        if self.audio_data.ndim > 1:
            if channel >= self.audio_data.shape[0]:
                raise ValueError(f"Channel {channel} not available. Audio has {self.audio_data.shape[0]} channels.")
            self.audio_data = self.audio_data[channel]
        
        # Create time array
        self.duration = len(self.audio_data) / self.sample_rate
        self.time_points = np.linspace(0, self.duration, len(self.audio_data))
        
        # Create interpolation function
        self.interpolator = interp1d(
            self.time_points, 
            self.audio_data, 
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Playback control variables
        self.is_playing = False
        self.playback_start_time = None
        self.playback_thread = None
    
    def __call__(self, t):
        """
        Get the audio value at time t (in seconds).
        
        Args:
            t (float or array-like): Time(s) in seconds
            
        Returns:
            float or array: Audio amplitude value(s) at the specified time(s)
        """
        return self.interpolator(t)
    
    def get_value(self, t):
        """
        Alternative method to get audio value at time t.
        """
        return self.__call__(t)
    
    def play_and_read_value(self, duration, start_time=0, read_time=None):
        """
        Play audio for a specified duration and then print the y-axis value at a specific time.
        
        Args:
            duration (float): How long to play the audio (in seconds)
            start_time (float): Where to start playing in the audio (in seconds)
            read_time (float): Time point to read the value from (if None, uses start_time + duration)
        """
        if read_time is None:
            read_time = start_time + duration
        
        print(f"Playing audio from {start_time:.2f}s for {duration:.2f}s...")
        print(f"Will read value at t={read_time:.2f}s after playback completes.")
        
        # Extract the audio segment to play
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)
        
        # Ensure we don't go beyond audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)
        
        if start_sample >= len(self.audio_data):
            print("Start time is beyond audio duration!")
            return
        
        audio_segment = self.audio_data[start_sample:end_sample]
        
        # Play the audio
        try:
            sd.play(audio_segment, samplerate=self.sample_rate)
            sd.wait()  # Wait until playback is finished
            print("Playback completed.")
        except Exception as e:
            print(f"Error during playback: {e}")
        
        # Read and print the value at the specified time
        value = self.get_value(read_time)
        print(f"Y-axis value at t={read_time:.2f}s: {value:.6f}")
        
        return value
    
    def play_with_live_reading(self, duration, start_time=0, read_interval=0.1):
        """
        Play audio while continuously reading and printing values.
        
        Args:
            duration (float): How long to play the audio (in seconds)
            start_time (float): Where to start playing in the audio (in seconds)
            read_interval (float): How often to read values (in seconds)
        """
        print(f"Playing audio from {start_time:.2f}s for {duration:.2f}s with live value reading...")
        
        # Extract the audio segment to play
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)
        
        if start_sample >= len(self.audio_data):
            print("Start time is beyond audio duration!")
            return
        
        audio_segment = self.audio_data[start_sample:end_sample]
        
        # Start playback
        try:
            sd.play(audio_segment, samplerate=self.sample_rate)
            
            # Live reading while playing
            playback_start = time.time()
            current_time = start_time
            
            print(f"{'Time (s)':<10} {'Amplitude':<15}")
            print("-" * 25)
            
            while current_time < start_time + duration:
                value = self.get_value(current_time)
                print(f"{current_time:<10.2f} {value:<15.6f}")
                
                time.sleep(read_interval)
                current_time = start_time + (time.time() - playback_start)
            
            sd.wait()  # Wait for playback to complete
            print("Playback completed with live reading.")
            
        except Exception as e:
            print(f"Error during playback: {e}")
    
    def play_and_read_multiple(self, duration, start_time=0, read_times=None):
        """
        Play audio for a duration and then read values at multiple specified times.
        
        Args:
            duration (float): How long to play the audio (in seconds)
            start_time (float): Where to start playing in the audio (in seconds)
            read_times (list): List of time points to read values from
        """
        if read_times is None:
            read_times = [start_time + duration]
        
        print(f"Playing audio from {start_time:.2f}s for {duration:.2f}s...")
        
        # Extract and play audio segment
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)
        
        if start_sample >= len(self.audio_data):
            print("Start time is beyond audio duration!")
            return
        
        audio_segment = self.audio_data[start_sample:end_sample]
        
        try:
            sd.play(audio_segment, samplerate=self.sample_rate)
            sd.wait()
            print("Playback completed.")
        except Exception as e:
            print(f"Error during playback: {e}")
        
        # Read values at specified times
        print(f"\nReading values at specified times:")
        print(f"{'Time (s)':<10} {'Amplitude':<15}")
        print("-" * 25)
        
        values = []
        for t in read_times:
            value = self.get_value(t)
            values.append(value)
            print(f"{t:<10.2f} {value:<15.6f}")
        
        return values
    
    def stop_playback(self):
        """
        Stop any ongoing playback.
        """
        sd.stop()
        self.is_playing = False
        print("Playback stopped.")
    
    def get_info(self):
        """
        Get information about the audio function.
        """
        return {
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'num_samples': len(self.audio_data),
            'max_amplitude': np.max(np.abs(self.audio_data)),
            'rms_amplitude': np.sqrt(np.mean(self.audio_data**2))
        }
    
    def plot_with_markers(self, read_times, start_time=0, end_time=None, num_points=1000):
        """
        Plot the audio function with markers at specified read times.
        
        Args:
            read_times (list): Time points to mark on the plot
            start_time (float): Start time for plotting
            end_time (float): End time for plotting
            num_points (int): Number of points to plot
        """
        if end_time is None:
            end_time = self.duration
            
        t = np.linspace(start_time, end_time, num_points)
        y = self(t)
        
        plt.figure(figsize=(12, 6))
        plt.plot(t, y, 'b-', alpha=0.7, label='Audio Function')
        
        # Add markers for read times
        for read_time in read_times:
            if start_time <= read_time <= end_time:
                value = self.get_value(read_time)
                plt.plot(read_time, value, 'ro', markersize=8, label=f't={read_time:.2f}s')
                plt.annotate(f'({read_time:.2f}, {value:.3f})', 
                           xy=(read_time, value), 
                           xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title('Audio Function with Read Points')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

def demo_playback_and_reading():
    """
    Demonstration of the playback and reading functionality.
    """
    # audio_file = "C:/Users/akibr/Downloads/samp.wav"  # Change this to your audio file
    audio_file = "C:/Users/akibr/Downloads/rickroll.mp3"

    try:
        # Create the audio function
        audio_func = AudioFunction(audio_file)
        
        # Print audio information
        info = audio_func.get_info()
        print("Audio Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
        
        # Example 1: Play for 3 seconds and read value at the end
        print("=== Example 1: Play and read single value ===")
        audio_func.play_and_read_value(duration=3.0, start_time=0, read_time=3.0)
        print()
        
        # Example 2: Play for 2 seconds and read values at multiple times
        print("=== Example 2: Play and read multiple values ===")
        read_times = [1.0, 2.0, 2.5, 3.0]
        audio_func.play_and_read_multiple(duration=2.0, start_time=0, read_times=read_times)
        print()
        
        # Example 3: Play with live reading (commented out to avoid overwhelming output)
        print("=== Example 3: Uncomment below for live reading ===")
        print("# audio_func.play_with_live_reading(duration=3.0, start_time=0, read_interval=0.2)")
        
        return audio_func
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have an audio file and update the file path.")
        print("Also ensure you have sounddevice installed: pip install sounddevice")
        return None

# Simple wrapper function for easy use
def create_playback_function(audio_file_path):
    """
    Create an audio function with playback capabilities.
    
    Returns a tuple: (audio_function, play_and_read_function)
    """
    audio_func = AudioFunction(audio_file_path)
    
    def play_and_read(duration, read_time=None, start_time=0):
        return audio_func.play_and_read_value(duration, start_time, read_time)
    
    return audio_func, play_and_read

# Installation requirements:
"""
pip install librosa scipy matplotlib numpy sounddevice

sounddevice is used for audio playback functionality.
"""

if __name__ == "__main__":
    print("Audio Function with Playback and Value Reading")
    print("=" * 50)
    
    print("Installation requirements:")
    print("pip install librosa scipy matplotlib numpy sounddevice")
    print()
    
    print("Usage examples:")
    print("1. audio_func = AudioFunction('my_audio.wav')")
    print("2. audio_func.play_and_read_value(duration=3.0, read_time=3.0)")
    print("3. audio_func.play_and_read_multiple(duration=2.0, read_times=[1,2,3])")
    print("4. audio_func.play_with_live_reading(duration=5.0, read_interval=0.5)")
    print()
    
    # Uncomment to run demo
    demo_playback_and_reading()