# TSM
Matlab Implementations of Time Scale Modification

### FDTSM - Frequency Dependent Time Scale Modification
Time Scale Modification algorithms scale all frequencies by the same amount.  FDTSM allows for frequency ranges to be scaled arbitarily.  The frequency range is split into regions (up to N/2 regions) with time scaling applied to each region.

**FDTSM files:**
  - FDTSM.m is the Frequency Dependent Time Scale Modification function.
  - FDTSM_script.m gives the minimum needed for using the FDTSM function.
  - FDTSM_10_Band_GUI.fig and FDTSM_10_Band_GUI.m are the files for the GUI.
  - FDTSM_GUI_example.m is a script which calls the GUI and then does the appropriate TSM.
  
  Output audio files are saved in the AudioOut folder as filename_TSM_ratios_FDTSM.wav
