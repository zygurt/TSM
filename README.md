# TSM
Matlab Implementations of Time Scale Modification
Documentation is gradually being moved to https://zygurt.github.io/TSM/

### FDTSM - Frequency Dependent Time Scale Modification
Time Scale Modification algorithms scale all frequencies by the same amount.  FDTSM allows for frequency ranges to be scaled arbitarily.  The frequency range is split into regions (up to N/2+1 regions) with time scaling applied to each region.

**FDTSM files:**
  - FDTSM.m is the Frequency Dependent Time Scale Modification function.
  - FDTSM_script.m gives the minimum needed for using the FDTSM function.
  - FDTSM_10_Band_GUI.fig and FDTSM_10_Band_GUI.m are the files for the GUI.
  - FDTSM_GUI_example.m is a script which calls the GUI and then does the appropriate TSM.

  Output audio files are saved in the AudioOut folder as filename_TSM_ratios_FDTSM.wav

### Functions - Useful MATLAB Functions
A collection of useful functions for signal processing in MATLAB

**Functions**
  - Checking_script.m is a script for checking functions and scripts in this repository.
  - LinSpectrogram.m plots a dual column latex paper suitable spectrogram with a linear frequency axis.
  - LogSpectrogram.m plots a dual column latex paper suitable spectrogram with a logarithmic frequency axis.
  - crosscorr_t.m computes the normalised time domain cross correlation between 2 vectors
  - find_peaks.m finds the peaks in a vector with the qualifying factors proposed by Laroche and Dolson.
  - find_peaks_log.m finds the peaks in a vector with the qualifying factors proposed by Karrer, Lee and Borchers.
  - maxcrosscorrlag.m computes the lag for maximum cross-correlation between two vectors.
  - previous_peak.m finds the location of the related peak in a previous vector. Proposed by Laroche and Dolson.
  - previous_peak_heuristic.m finds the location of the related peak in the previous vector while considering the distance between the current and previous peak location. Proposed by Karrer, Lee and Borchers.
  - st_balance.m computes frame and file stereo balance of a stereo signal.
  - st_phase_coherence.m computes frame and file stereo phase coherence of a stereo signal.
  - st_width.m computes frame and file stereo width of a stereo signal.
  - ZFR.m computes speech epochs (glottal closure instants) proposed by Murt and Yegnanarayana.

### N_Channel - N Channel Phase Vocoder Based Implementations  
A collection of Phase Vocoder based Time Scale Modification Implementations.  Easch channel is processed individually, with no regard for relationships between channels.

**N_Channel files:**
  - PL_PV.m contains Identity Phase Locking and Scaled Phase Locking Phase Vocoders proposed by Laroche and Dolson.
  - PV.m contains a base Phase Vocoder proposed by Portnoff
  - Phavorit.m contains the PhaVoRIT: A Phase Vocoder for Real-Time Interactive Time-Stretching (Karrer, Lee and Borchers) without silent passage phase reset.

### Stereo - Stereo Phase Vocoder Implementations
A collection of published and unpublished implementations of stereo TSM using the phase vocoder.

**Stereo files:**
  - PV.m is an n-channel capable Phase Vocoder.
  - PV_Altoe.m is a stereo capable Phase Vocoder using the method proposed by Altoe.
  - PV_Bonada.m is a stereo capable Phase Vocoder using the method proposed by Bonada.  This is the current state-of-the-art stereo method.
  - PV_MS_File.m is a stereo capable Phase Vocoder using sum and difference transformation to increase the quality of stereo TSM.  The entire file is processed prior to, and after, TSM.
  - PV_MS_Frame.m is a stereo capable Phase Vocoder using sum and difference transformation to increase the quality of stereo TSM.  The appropriate difference equation is used for each frame in this method.

### Time_Domain - Time Domain Time Scale Modification Implementations
A collection of published Time Domain TSM implementations.

**Time Domain files:**
  - SOLA.m is a single channel implementation of the Synchronised Overlap Add (SOLA) method of time-scale modification. Proposed by Roucos and Wilgus.
  - SOLA_DAFX.m is a modified version of the SOLA implementation in Zolzer et al.
  - ESOLA.m single channel implementation of Epoch Synchronous Overlap Add (ESOLA). Proposed by Rudresh et al.
  - WSOLA.m is an N channel implementation of the Waveform Similarity Overlap Add (WSOLA) method of time-scale modification.  Proposed by Verhelst and Roelands.
  - WSOLA_Driedger.m is a modified version of the WSOLA implementation by Driedger (2014).
  
## References
- Flanagan and Golden, Phase Vocoder, 1966.
- Laroche and Dolson, Improved Phase Vocoder Time-Scale Modification of Audio, 1999.
- Portnoff, Implementation of the Digital Phase Vocoder Using the Fast Fourier Transform, 1985.
- Karrer, Lee and Borchers, PhaVoRIT: A Phase Vocoder for Real-Time Interactive Time-Stretching, 2006.
- Murt and Yegnanarayana, Epoch Extraction from Speech Signals, 2008.
- Altoe, A Transient-Preserving Audio Time-Stretching Algorithm and a Real-Time Realization for a Commercial Music Product, 2012.
- Bonada, Audio Time-Scale Modification in the Context of Professional Audio Post-production, 2002.
- Roucos and Wilgus, High Quality Time-Scale Modification for Speech, 1985.
- Zolzer et al., DAFx - Digital Audio Effects, John Willey & Sons, 2002.
- Rudresh et al., Epoch-Synchronous Overlap-Add (ESOLA) for Time- and Pitch-Scale Modification of Speech Signals, 2018.
- Verhelst and Roelands, An Overlap-Add Technique Based on Waveform Similarity (WSOLA) for High Quality Time-Scale Modification of Speech, 1993
- Driedger and Mueller, TSM Toolbox: MATLAB Implementations of Time-Scale Modification Algorithms, 2014.
