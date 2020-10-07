"""
Utils for visualization of audio parameters/waveforms. Use to get
intuition about how DDSP behaves.
"""

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['standard_plot', 'alt_plot']

# utils for plotting
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def standard_plot(audio_parameters):
    # standard plot: no overlays, etc.
    
    plt.figure(figsize=(7,4))
    plt.plot(audio_parameters['f0_hz'], label='f0')
    plt.title('f0 Hz')
    plt.legend()
    plt.show()
    plt.figure(figsize=(7,4))
    plt.plot(audio_parameters['loudness_db'])
    plt.show()
    plt.figure(figsize=(7,4))
    plt.plot(audio_parameters['f0_confidence'])
    plt.show()
    

def get_percent(arr, percentile):
    return np.quantile(arr, [percentile])

def alt_plot(audio_parameters, shade_param=None, cutoff_percentile=0.5, plot_freqs=False, waveform=None):
    # plot with extra stuff (overlays, etc.)
    # audio_parameters: the audio parameters obtained from before
    # shade_param: the parameter potentially used to shade the graph vertically
    # cutoff_percentile: percentile to cutoff the shading for shade_param
    # plot_freqs: plot musical note frequencies as horizontal lines on the plot
    # waveform: if audio provided, plot it
    
    plt.figure(figsize=(14,8))
    plt.plot(audio_parameters['f0_hz'], label='f0')
    if not(shade_param is None):
        # if -1, use smooth shading, otherwise use percentile
        if cutoff_percentile == -1:
            max_, min_ = audio_parameters[shade_param].max(), audio_parameters[shade_param].min()
            for i, param in enumerate(audio_parameters[shade_param]):
                plt.axvspan(i, i + 1, alpha=(param - min_) / (2 * (max_ - min_)))
        else:
            cutoff = get_percent(audio_parameters[shade_param], cutoff_percentile)
            for region in contiguous_regions(audio_parameters[shade_param] > cutoff):
                plt.axvspan(region[0], region[1], alpha=0.3)

    title = 'f0 Hz'
    if not(shade_param is None):
        title += ', shaded by {} ({})'.format(shade_param, 'smoothly' if cutoff_percentile==-1 else cutoff_percentile)
    plt.title(title)

    # extra waveform ontop
    if not(waveform is None):
        # skipping over 63 elements since waveform is sampled 64x
        # compared to the audio parameters
        plt.plot(100 * waveform.flatten()[::64] + get_percent(audio_parameters['f0_hz'], 0.1),
                 label='waveform')
    
    # extra frequency horizontal lines
    if plot_freqs:
        frequencies = {'_A': 220,
                       '_B': 246,
                       '_C': 261,
                       '_D': 293,
                       '_E': 329,
                       '_F': 349,
                       '_G': 392,
                       'A': 440,
                       'B': 493,
                       'C': 523,
                       'D': 587,
                       'D#': 622,
                       'E': 659,
                       'F': 698,
                       'F#': 740,
                       'G': 784,
                       'A_': 880}
        for note, f in frequencies.items():
            plt.plot(f * np.ones(1300), label='{}{}'.format(note, f))
    
    plt.legend()
    plt.show()
    plt.figure(figsize=(14,8))
    plt.plot(audio_parameters['loudness_db'])
    plt.show()
    plt.figure(figsize=(14,8))
    plt.plot(audio_parameters['f0_confidence'])
    plt.show()
