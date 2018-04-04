# TSM
Matlab Implementations of Time Scale Modification

### FDTSM - Frequency Dependent Time Scale Modification
Time Scale Modification algorithms scale all frequencies by the same amount.  FDTSM allows for frequency ranges to be scaled arbitarily.  The frequency range is split into regions (up to N/2+1 regions) with time scaling applied to each region.

**FDTSM files:**
  - FDTSM.m is the Frequency Dependent Time Scale Modification function.
  - FDTSM_script.m gives the minimum needed for using the FDTSM function.
  - FDTSM_10_Band_GUI.fig and FDTSM_10_Band_GUI.m are the files for the GUI.
  - FDTSM_GUI_example.m is a script which calls the GUI and then does the appropriate TSM.

  Output audio files are saved in the AudioOut folder as filename_TSM_ratios_FDTSM.wav

### Stereo - Stereo Phase Vocoder Implementations
A collection of published and unpublished implementations of stereo TSM using the phase vocoder.

**Stereo files:**
  - PV.m is an n-channel capable Phase Vocoder.
  - PV_Altoe.m is a stereo capable Phase Vocoder using the method proposed by Altoe.
  - PV_Bonada.m is a stereo capable Phase Vocoder using the method proposed by Bonada.  This is the current state-of-the-art stereo method.
  - PV_MS_File.m is a stereo capable Phase Vocoder using sum and difference transformation to increase the quality of stereo TSM.  The entire file is processed prior to, and after, TSM.
  - PV_MS_Frame.m is a stereo capable Phase Vocoder using sum and difference transformation to increase the quality of stereo TSM.  The appropriate difference equation is used for each frame in this method.

### Functions - Useful MATLAB Functions
A collection of useful functions for signal processing in MATLAB

**Functions**
  - LinSpectrogram.m plots a dual column latex paper suitable spectrogram with a linear frequency axis.
  - LogSpectrogram.m plots a dual column latex paper suitable spectrogram with a logarithmic frequency axis.
  - st_balance.m computes frame and file stereo balance of a stereo signal.
  - st_phase_coherence.m computes frame and file stereo phase coherence of a stereo signal.
  - st_width.m computes frame and file stereo width of a stereo signal.
