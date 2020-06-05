# Multi-Channel Time-Scale Modification

## N_Channel - N Channel Phase Vocoder Based Implementations
A collection of Phase Vocoder based Time Scale Modification Implementations.  Easch channel is processed individually, with no regard for relationships between channels.

**N_Channel files:**
  - PL_PV.m contains Identity Phase Locking and Scaled Phase Locking Phase Vocoders proposed by Laroche and Dolson.
  - PV.m contains a base Phase Vocoder proposed by Portnoff
  - Phavorit.m contains the PhaVoRIT: A Phase Vocoder for Real-Time Interactive Time-Stretching (Karrer, Lee and Borchers) without silent passage phase reset.

## Stereo - Stereo Phase Vocoder Implementations
A collection of published and unpublished implementations of stereo TSM using the phase vocoder.

**Stereo files:**
  - PV.m is an n-channel capable Phase Vocoder.
  - PV_Altoe.m is a stereo capable Phase Vocoder using the method proposed by Altoe.
  - PV_Bonada.m is a stereo capable Phase Vocoder using the method proposed by Bonada.  This is the current state-of-the-art stereo method.
  - PV_MS_File.m is a stereo capable Phase Vocoder using sum and difference transformation to increase the quality of stereo TSM.  The entire file is processed prior to, and after, TSM.
  - PV_MS_Frame.m is a stereo capable Phase Vocoder using sum and difference transformation to increase the quality of stereo TSM.  The appropriate difference equation is used for each frame in this method.
