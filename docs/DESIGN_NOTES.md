# Design Notes

Engineering notes for the CW Doppler respiratory radar built for EEE 4550/5557
Radar Systems (Florida International University, Spring 2026, Group 04). The
README covers how to run the code and who did what; this document covers how
the signal chain works, why each parameter has the value it has, and what was
found when the delivered code was put under test after the course ended.

Every number below comes from the course report, the slide deck, the project
notes kept during development, or the code in `respradar/`. Where the sources
disagree, section 9 says which one the code follows.

## 1. Principle of operation

The radar transmits an unmodulated 5.95 GHz carrier and measures the phase of
the echo from the subject's chest. A change in range of `d` metres changes the
two-way path by `2d`, so the echo phase moves by

    phi = 4 * pi * d / lambda

At 5.95 GHz the wavelength is `lambda = c / f = 50.4 mm`, which gives

    1 rad of phase  =  lambda / (4 pi)  =  4.01 mm of chest displacement
    1 mm of chest displacement  =  0.249 rad

Quiet breathing moves the chest wall by roughly 3 to 12 mm; 8 mm peak-to-peak
is about 2 rad of phase swing and deep breathing is several full turns. That is
why the chain cannot simply read `atan2(Q, I)`: the wrapped phase crosses
plus or minus pi every breath and must be unwrapped as a stream.

The other consequence of the short wavelength is that breathing is enormous in
radar terms. The respiration band (0.1 to 0.5 Hz, 6 to 30 breaths per minute)
contains radians of modulation, while the cardiac wall moves 0.2 to 0.5 mm,
only 0.05 to 0.12 rad. The system targets respiration; heart rate was scoped out
(section 10).

### Why the transmit tone sits at a 100 kHz offset

The Pluto transmits a 100 kHz complex cosine, so the echo arrives at +100 kHz
in the receive baseband rather than at DC. Everything that the AD9363 adds at
DC (LO leakage, converter offsets, the DC-correction loops themselves) then
falls outside the signal and is removed by the channeliser's frequency shift
and low-pass filter instead of corrupting the phase measurement. The static
component that remains after the shift (the stationary echo from walls and the
subject's torso, plus direct TX-to-RX coupling on the chip) is handled
separately by the DC blocker in section 3.

## 2. Hardware

| Item | Value | Note |
|---|---|---|
| SDR | ADALM-Pluto (AD9363) | 12-bit converters, TX and RX share one LO, so TX/RX phase coherence is guaranteed by hardware |
| Carrier | 5.95 GHz | needs the Pluto firmware tuning-range extension to ~6 GHz |
| Sample rate | 1 MS/s complex | streamed over IP (USB-Ethernet gadget) |
| RF bandwidth | 20 MHz | AD9363 analog filter setting |
| TX | attenuation 0 dB | the Pluto's maximum output |
| RX | manual gain 50 dB | watch the raw IQ constellation for saturation |
| Antennas | two Vivaldi, 1 to 7 GHz | built by a student in an FIU lab; measured S11 of -10.76 dB at 5.95 GHz |
| LO phase noise | about -90 dBc/Hz at 1 kHz offset | sufficient for breathing, the limiting factor for heart rate |

### 2.1 Reaching 5.95 GHz on a stock Pluto

The AD9363 is specified from 325 MHz to 3.8 GHz, so a Pluto will not tune to
5.95 GHz as shipped. The AD9363 and AD9364 are the same silicon with different
guaranteed ranges, and Analog Devices documents a firmware setting that makes
the Pluto drive the chip as an AD9364 (70 MHz to 6 GHz, 56 MHz bandwidth,
one RX and one TX channel). The report cites this through [RTL-SDR's write-up
of the hack](https://www.rtl-sdr.com/adalm-pluto-sdr-hack-tune-70-mhz-to-6-ghz-and-gqrx-install/),
which in turn points at the ADI wiki page
[Customizing the Pluto configuration](https://wiki.analog.com/university/tools/pluto/users/customizing).
To reproduce it, log in to the Pluto (`ssh root@192.168.2.1`, password
`analog`, or the USB serial console) and set the U-Boot environment:

    # firmware v0.32 and later
    fw_setenv compatible ad9364
    fw_setenv mode 1r1t
    reboot

    # firmware before v0.32
    fw_setenv attr_name compatible
    fw_setenv attr_val ad9364
    reboot

`fw_printenv` shows the stored values after the reboot, and the device then
accepts LO frequencies up to 6 GHz from GNU Radio's PlutoSDR blocks. The
firmware version is on the Pluto's info page (`http://192.168.2.1`).

What this does and does not change: the converters stay 12-bit at 1 MS/s, so
there is no loss of ADC resolution. What is given up is the guarantee: ADI
only promises the extended range on the early units that shipped with a real
AD9364, and on an AD9363 unit the RF front end runs outside its specification
above 3.8 GHz. TX output is lower there, which is one reason the project runs
TX attenuation at 0 dB. The measured S11 and the live IQ constellation were
the check that the link actually worked at 5.95 GHz.

Two other platforms were tried first and rejected. A pair of HackRF One boards
has independent LOs, and their relative phase drifted by more than the
breathing signal over a minute; their 8-bit converters also leave little
margin. A BladeRF 2.0 micro has the right converter and a shared LO, but under
WSL2 its USB 3.0 stream underflowed about once a second at 1 MS/s, breaking
phase continuity. The Pluto's IP transport avoids the USB layer and sustained
1 MS/s without drops.

The first antenna pair (log-periodics with mismatched SMA adapters) returned
15 to 20 dB less signal than expected; the raw IQ constellation barely moved
off its static point. The matched Vivaldi pair fixed it, and the constellation
became a clean arc as the subject breathed. The slide deck, written before the
antenna change, still refers to 5.8 GHz horns; the report and the hardware
photo show the Vivaldi pair that was actually used.

## 3. Signal chain

```
Pluto RX, 1 MS/s complex, echo at +100 kHz
  └─ freq_xlating_fir_filter_ccc   shift -100 kHz, Hamming LPF 100 Hz, decimate 2000 -> 500 S/s
       ├─ breathing path
       │    dc_blocker_cc(3200, long form)   remove the static echo in the IQ plane
       │    complex_to_arg                   atan2(Q, I), wrapped phase
       │    phase unwrap                     streaming, state kept across work() calls
       │    fir_filter_fff(2001 taps)        firwin([0.1, 0.5], fs=500): see 3.3
       │    ├─ FFT BPM estimator
       │    ├─ zero-crossing BPM estimator
       │    └─ autocorrelation BPM estimator
       └─ motion path (no DC blocker)
            complex_to_arg -> phase unwrap -> |dphi/dn| -> motion detector
```

### 3.1 Channeliser

`firdes.low_pass(1.0, 1e6, 100, 50, HAMMING)` produces a 48,181-tap filter.
Because the block only computes outputs at the decimated rate, that is about
24 M complex multiply-accumulates per second: cheap. The 2000:1 coherent sum
gives `10 log10(2000) = 33 dB` of processing gain, which is what makes a
milliwatt of TX power enough at 0.5 to 2 m. The group delay is 24 ms.

### 3.2 DC blocker

The stationary echo is the largest component of the received signal. If it is
left in, `atan2` returns the angle of (static + moving) rather than the angle
of the moving part, and the breathing waveform is distorted in an
amplitude-dependent way. Removing it in the IQ plane, before the arctangent, is
the clean solution. Measured through GNU Radio's `dc_blocker_cc(3200, long_form=True)` at
500 S/s, the gain is -31.5 dB at 0.01 Hz, -19.8 dB at 0.02 Hz, -6.0 dB at
0.05 Hz, -0.4 dB at 0.1 Hz and flat above 0.15 Hz: a high-pass with its -3 dB
point near 0.06 Hz, below the lowest breathing rate of interest. The long form
is four cascaded moving averages, which costs `2 (N - 1) = 6,398` samples of
delay: the breathing waveform on screen lags the chest by about 13 s from this
block alone, 15 s with the FIR below it.

The motion path deliberately bypasses this block (section 6).

### 3.3 The 2001-tap "bandpass"

The project notes, the report, and the code all describe this filter as a
0.1 to 0.5 Hz bandpass. Its measured response tells a different story:

| Frequency | 0.02 Hz | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gain (dB) | -1.9 | -1.8 | -1.4 | -0.4 | 0.0 | -0.9 | -3.5 | -8.0 | -24.9 | -47.4 |

A 2001-tap Hamming-windowed FIR at 500 S/s has a transition width of about
`3.3 * fs / N = 0.8 Hz`, twice the width of the whole band, so it cannot form
an edge at 0.1 Hz at all. In practice the filter is a low-pass with its -3 dB
point near 0.46 Hz. It does its real job (rejecting heartbeat, harmonics, and
wideband phase noise above 0.5 Hz), while the high-pass side of the
"bandpass" is actually provided by the DC blocker upstream. The combination is
a bandpass; the FIR alone is not. A genuine 0.1 Hz edge would need an impulse
response of 15 s or more, which would add that much latency to every
estimator. The filter was left as delivered and documented here instead.

The one place this matters is after a large body movement: a step in phase
passes the filter almost unattenuated (-1.9 dB at 0.02 Hz) and the breathing
path needs the DC blocker to recover, see section 6.

## 4. BPM estimators

All three read the same 500 S/s filtered phase and output BPM as a stream
(0.0 until the first valid estimate). Running three structurally different
estimators was a deliberate choice: each fails differently, and agreement
between two of them is a stronger statement than any one of them alone.

### 4.1 FFT peak (`respradar/blocks/fft_bpm.py`)

Every 15 s the last 15 s (7,500 samples) are linearly detrended, Hann
windowed, zero-padded to 32,768 points and transformed. The largest bin in
0.1 to 0.5 Hz is refined with a parabola through the log power of the bin and
its two neighbours:

    delta = 0.5 * (a - c) / (a - 2 b + c),   a, b, c = log P[k-1], log P[k], log P[k+1]
    f_peak = (k + delta) * fs / N_fft

The bin spacing is 0.0153 Hz (0.92 BPM); the interpolation brings a clean tone
to within about 0.1 BPM. An estimate is accepted only when the record's
peak-to-peak amplitude is at least `min_pp = 0.005 rad` and the peak is at
least `snr_gate = 6` times the median of the in-band bins that lie outside the
window's main lobe (plus or minus 2/T = 0.133 Hz for Hann). Section 5 explains
why that last clause exists.

The linear detrend costs accuracy at the bottom of the band, where a 15 s
record holds only two cycles: 8 BPM reads 7.70 with the detrend and 8.00
without. It stays because of what it buys under drift: with a 0.5 rad/s phase
ramp added to 15 BPM breathing, the detrended estimate is still 15.00 while
mean removal alone gives 4.45 and no correction gives 4.20.

### 4.2 Zero crossing (`respradar/blocks/zc_bpm.py`)

A software Schmitt trigger. The block keeps the last 15 s of signal, takes its
minimum and maximum, and sets two thresholds at `centre +/- 0.3 * pp / 2`. A
crossing is counted when the signal rises above the upper threshold, and the
trigger re-arms only after it falls below the lower one, so noise riding on
the waveform cannot produce double counts. With `n` crossings in the window,

    BPM = (n - 1) * 60 / (t_last - t_first)

The thresholds rescale themselves to whatever amplitude the subject produces
(distance, posture, and gain all change it), and `min_pp = 0.005 rad` keeps
the trigger off when there is nothing to trigger on. Output updates every
work() call, so this is the fastest-reacting of the three and also the most
jittery.

### 4.3 Autocorrelation (`respradar/blocks/ac_bpm.py`)

Every 60 s the last 60 s (30,000 samples) are mean-removed, zero-padded to
twice their length (so the result is a linear, not circular, correlation) and
autocorrelated through the FFT:

    R[tau] = irfft( |rfft(x_padded)|^2 ),   normalised so that R[0] = 1

The first peak of `R` in the lag range 2 s to 20 s (30 down to 3 BPM) gives the
period, refined with the same three-point parabola. A peak below
`corr_threshold = 0.3` is ignored. Because the search ignores lag zero and the
mean is removed, DC offsets and slow drift do not affect it, and because the
lag range is set by the window rather than by the filter, it can read rates
below the FIR's nominal 6 BPM floor.

### 4.4 Comparison

| | FFT | Zero crossing | Autocorrelation |
|---|---|---|---|
| Window | 15 s | 15 s rolling | 60 s |
| Update | every 15 s | every work() call | every 60 s |
| Resolution | 0.92 BPM per bin, ~0.1 BPM after interpolation | set by crossing timing | sub-sample after interpolation |
| Rate range | 6 to 30 BPM (band) | 6 to 30 BPM (band) | 3 to 30 BPM (lag search) |
| Validity gate | min_pp and spectral ratio | min_pp, two crossings | normalised correlation 0.3 |
| Strength | unambiguous for a clean periodic signal | fastest, amplitude-independent | best steady-state accuracy, drift-immune |
| Failure mode | first window after a movement contains filter ringing; biased ~4% low at 8 BPM | harmonics or clipping can add or drop crossings | 60 s latency; first window includes settling transients |
| Measured on clean tones (10 to 27 BPM) | within 0.3 BPM | exact | exact |

The report's screenshot of a stable session shows the practical consequence
of the AC window: FFT and ZC agree at 27 to 28 BPM while AC still reads 3.5,
a long-lag peak from its first, partly settled window. The consensus layer
reported the FFT+ZC pair and ignored it.

## 5. The FFT gate: what was wrong and what changed

The delivered estimator windowed the record with a 4-term Blackman-Harris and
required the peak to exceed six times the median of all bins in 0.1 to 0.5 Hz.
Synthetic clean breathing through the delivered block gave:

| Rate (BPM) | 12 | 15 | 18 | 20 | 22 | 25 | 27 |
|---|---|---|---|---|---|---|---|
| Peak / median, delivered | 6.5 | 11.2 | 12.0 | 4.1 | 14.4 | 9.0 | 44.7 |
| Accepted (gate 6) | yes | yes | yes | no | yes | yes | yes |

A perfectly clean 20 BPM tone produced no output at all. The cause is
resolution: a Blackman-Harris main lobe is 8/T wide null to null, 0.53 Hz on a
15 s record, which covers the entire 0.4 Hz band. The "median of in-band
bins" was therefore the skirt of the signal's own main lobe, not a noise
estimate, and whether the ratio cleared 6 depended on where the lobe happened
to fall relative to the band edges. Enlarging the FFT to 65,536 points made it
worse (15 to 22 BPM all landed at 5.4 to 5.7) because more bins sampled the
same lobe more densely.

The fix has three parts, all in `fft_bpm.py`:

* Hann window instead of Blackman-Harris. Its main lobe is 4/T = 0.27 Hz on
  15 s, and with the FIR already removing everything outside the band there is
  no strong out-of-band component for the -92 dB sidelobes to suppress.
* The noise reference is the median of in-band bins more than 2/T from the
  peak, outside the main lobe. With that reference the ratio for clean tones is
  30 at 10 BPM and above 10,000 at 27 BPM; a realistic waveform (second
  harmonic at 30 percent, amplitude wobble, noise) gives 30 to 40 at 10 BPM and
  100 to 3,000 above 15 BPM.
* The FFT length is never smaller than the record. The report specifies a 20 s
  window (10,000 samples) with an 8,192-point transform; `numpy.fft.rfft(x,
  n=8192)` silently truncates such a record to its first 8,192 samples
  (16.4 s). The block now pads to at least the next power of two above the
  window length (32,768 by default).

What the gate still cannot do: with a 15 s record and a 0.4 Hz band there are
only about six independent spectral samples, so filtered noise alone clears
the ratio of 6 in roughly 40 percent of windows (median ratio 4.6, 90th
percentile 15). A spectral gate on this little data is a sanity check, not a
presence detector. The consensus vote absorbs most of the consequences, but a
real "nobody in the beam" decision belongs upstream, on the magnitude of the
DC-blocked IQ signal: with no moving target that magnitude is receiver noise,
with a breathing subject it is the chest echo. This is listed in section 10.

## 6. Motion path

Gross body movement (sitting down, turning, someone walking through the beam)
produces phase excursions that swamp the breathing signal and ring the filters
for seconds afterwards. The V4 design detects it on a separate branch and
tells the operator, rather than trying to make the estimators immune.

The branch takes the channeliser output before the DC blocker, extracts and
unwraps the phase, and computes `|dphi/dn|` in rad per sample. The detector
averages that over 50 samples (100 ms), compares the average with 0.1 rad per
sample, and holds its output at 1 for 500 samples (1 s) after the last sample
above threshold. At 500 S/s and 4.01 mm per radian,

    0.1 rad/sample  =  0.1 * 500 * 4.01 mm/s  =  about 200 mm/s radial velocity

Breathing sits far below this: a 5 mm excursion at 0.25 Hz peaks at 7.9 mm/s,
or 0.002 rad per sample, and even a 12 mm breath at 0.33 Hz is 0.006. A 30 cm
shift of the torso in one second peaks at roughly 470 mm/s (0.23 rad per
sample) and trips the detector within the 100 ms averaging window. The hold
stops the LED flickering during the pauses inside one movement.

The DC blocker is bypassed on this branch because it would remove exactly the
slow, large component the detector is looking for, and because its 13 s of
delay would make the LED late. On the breathing branch the same 30 cm shift is
a 75 rad step. Measured on the simulator, the step emerges from the breathing
path's 15 s pipeline delay as a 74 rad excursion lasting about 6 s, after
which the waveform is back to its 4 rad breathing swing. The estimators take
longer: the zero-crossing block's thresholds are set from a 15 s window that
now contains the excursion, so it reports nothing until the excursion has left
that window (about 15 s more), the FFT needs a clean 15 s record, and the
autocorrelation a clean 60 s one. In the meantime the outputs are meaningless.
The dashboard shows the motion LED next to the gauges for that reason:
readings taken while it is lit, and for the next half minute, should be
discarded, and the session log records the start and end of every event so
they can be excluded afterwards.

## 7. Consensus and dashboard

### 7.1 Vote (`respradar/consensus.py`)

The dashboard averages the last 50 samples (0.1 s) of each estimator's output
and votes 20 times a second:

1. An estimator reporting 0.5 BPM or less has not locked and is left out.
   Fewer than two live estimators gives `waiting`.
2. The pair with the smallest absolute difference is taken. If the difference
   exceeds 3 BPM the result is `diverging` and no number is shown.
3. Otherwise the consensus is the mean of that pair. If the third estimator is
   also within 3 BPM of it the confidence is `3/3`; if not, the agreeing pair
   is named (`FFT+ZC`, `FFT+AC`, or `ZC+AC`).

The rule set never has to diagnose which estimator is wrong; it reports a
number only when two independent methods corroborate it.

### 7.2 Data path (`respradar/taps.py`, `respradar/flowgraph.py`)

Every signal shown on screen passes through a `TapBuffer`: a fixed-capacity
ring buffer with a lock and a monotonically increasing write count. GNU Radio
work() threads push into them through `probe_sink` blocks; the Qt timer reads
`latest(n)` from each at 20 Hz. There is no inter-process communication and no
queue: a slow GUI drops frames instead of back-pressuring the flowgraph.

| Tap | Rate | Capacity | Shown as |
|---|---|---|---|
| raw_iq | 1 MS/s complex | 50,000 | last 500 samples (0.5 ms, 50 IF cycles) as a constellation |
| breathing | 500 S/s | 30,000 | last 10,000 samples (20 s) |
| motion_raw, motion_detect | 500 S/s | 15,000 | last 5,000 samples (10 s); LED from the last 250 |
| bpm_fft, bpm_zc, bpm_ac | 500 S/s | 4,000 | mean of the last 50 samples |
| phase_breath, phase_motion | 500 S/s | 20,000 | not plotted, kept for debugging |

The flowgraph class takes `source="pluto"` or `source="sim"`; everything after
the front end is identical, so the simulator exercises the real channeliser,
filters, estimators and GUI.

### 7.3 Session log (`respradar/session_logger.py`)

Pressing record opens `radar_session_<timestamp>.csv` with the columns

    t_sec, event, fft_bpm, zc_bpm, ac_bpm, consensus_bpm, consensus_confidence, motion_state, notes

`t_sec` is monotonic-clock time since the session started, immune to wall-clock
corrections. Events are `SESSION_START` (notes carry the file name and wall
time), `BPM_SAMPLE` once per second, `MOTION_START` and `MOTION_END` on the
detector's edges, and `SESSION_END`. Every row is flushed as it is written.

## 8. Simulator (`respradar/sim_source.py`)

The simulator replaces the Pluto with a GNU Radio source that emits, at
1 MS/s, the IF tone multiplied by a two-component echo:

    s(n) = [ A_static + A_body * exp(j * 0.249 * d(t)) ] * exp(j 2 pi 100 kHz t) + noise
    d(t) = A_b sin(2 pi f_b t) + A_drift sin(2 pi f_drift t) + movements(t)

`A_static` is the stationary reflection, `A_body` the chest echo, `d(t)` the
two-way range change in millimetres. Movements are raised-cosine steps of a
configurable travel (300 mm default) and duration (1 s) at a configurable
period, so the body stays displaced after each one as a real subject would.
Noise is complex Gaussian per sample, and the 33 dB of decimation gain applies
to it just as it does on hardware.

It is good for three things: running the dashboard without hardware, pushing
a known rate through the complete chain to check the estimators end to end
(the test suite does this at 15 BPM and checks that a movement trips the
detector only during the movement), and reproducing settling behaviour after
a step. It does not model multipath, direct TX-to-RX coupling as a separate
term, I/Q imbalance, converter saturation, the real asymmetric shape of a
breath, or the LO's 1/f phase noise, which on hardware is the dominant noise
term inside the respiration band.

## 9. Report versus code

The course report was written before the final debugging pass; the code is
the reference. Differences, stated as post-submission corrections:

| Topic | Report or slides | Code |
|---|---|---|
| FFT window | 20 s | 15 s; the window was shortened while debugging and left there. With the delivered 8,192-point FFT a 20 s record was being truncated (section 5) |
| FFT window function | Blackman-Harris | Hann (section 5) |
| Zero-crossing formula | `(N - 1) * 60 / (2 t_window)`, two crossings per cycle | one Schmitt-trigger event per cycle, divided by the time between first and last event, not the window length |
| Antennas | slides: 5.8 GHz horns | Vivaldi 1 to 7 GHz, as in the report and photo |
| Bandpass | 0.1 to 0.5 Hz bandpass | 0.46 Hz low-pass in effect; the DC blocker is the high-pass (section 3.3) |
| AC reading in the report's stable-breathing figure | 3.5 BPM against 27 to 28 for the others | first 60 s window captured settling; the vote correctly excluded it |

## 10. Lessons learned and future work

From the course:

* Phase coherence between TX and RX is the first requirement. Two radios with
  independent LOs drift by more than the signal; one transceiver with a
  shared LO removes the problem by construction.
* Antenna match matters more than antenna gain. A bad SMA transition cost
  more signal than any amplifier could have returned.
* The transport layer is part of the radar. USB pass-through under WSL2 broke
  phase continuity once a second; the IP link did not.
* Early runs locked on a stable reading near 18 BPM with nobody in front of
  the antenna. Adding the DC blocker and the filter reduced it, and it is
  consistent with filtered noise peaking at the filter's own centre, about
  0.3 Hz (section 3.3). It is the same effect that limits the FFT gate.
* One estimator is never enough. Each method's failure mode was found by
  watching it disagree with the others.

Scoped out of the course deliverable: heart rate (0.05 to 0.12 rad of
modulation against the Pluto's close-in phase noise), a 3 to 12 Hz tremor
branch, an FMCW range branch for fall detection and range-gated heart rate,
and fusion of the two radars.

Worth doing next, in rough order of value:

1. Presence detection on the magnitude of the DC-blocked IQ, gating all three
   estimators, instead of relying on the FFT's spectral ratio.
2. Overlapped updates for the FFT (every 5 s on a 15 s window) and a shorter
   autocorrelation window with overlap, so the vote has fresh inputs during
   the first minute and after every movement.
3. A real high-pass edge, either a longer FIR with its latency accepted or a
   second-stage decimation to 50 S/s where a 0.1 Hz edge is cheap.
4. A velocity readout in mm/s beside the motion trace, and two thresholds
   (slight versus large movement) as the notes proposed.
5. A CSV post-processor that plots each estimator over a session and marks
   the motion events.
6. Validation against a respiration belt or capnograph across postures and
   at the 2 m edge of the stated range, which the report identifies as the
   step that would turn the project into a measurement instrument.
