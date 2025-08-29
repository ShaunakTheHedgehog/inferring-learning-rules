"""Tools to generate and identify spacers.

Spacers are sequences of up and down pulses with a specific, identifiable pattern.
They are generated with a chirp coding to reduce cross-correlaation sidelobes.
They are used to mark the beginning of a behaviour sequence within a session.

Example
-------
>>> spacer = Spacer()
>>> spacer.add_spacer_states(sma, t, next_state='first_state')
>>> for i in range(ntrials):
... sma.add_state(
...     state_name='first_state',
...     state_timer=tup,
...     state_change_conditions={'Tup': f'spacer_low_{i:02d}'},
...     output_actions=[('BNC1', 255)],  # To FPGA
... )

"""

import numpy as np
from scipy import signal


class Spacer:
    def __init__(self, dt_start=0.02, dt_end=0.4, n_pulses=8, tup=0.05):
        """Computes spacer up times using a chirp up and down pattern.

        Parameters
        ----------
        dt_start : float
            First spacer up time.
        dt_end : float
            Last spacer up time.
        n_pulses : int
            Number of spacer up times, one-sided (i.e. 8 means 16 - 1 spacers times)
        tup: float
            Duration of the spacer up time.
        """
        self.dt_start = dt_start
        self.dt_end = dt_end
        self.n_pulses = n_pulses
        self.tup = tup
        assert np.all(np.diff(self.times) > self.tup), 'Spacers are overlapping'

    def __repr__(self):
        return f'Spacer(dt_start={self.dt_start}, dt_end={self.dt_end}, n_pulses={self.n_pulses}, tup={self.tup})'

    @property
    def times(self):
        """Computes spacer up times using a chirp up and down pattern.

        Each time corresponds to an up time of the BNC1 signal.

        Returns
        -------
        numpy.array
            Numpy arrays of spacer times.
        """
        # upsweep
        t = np.linspace(self.dt_start, self.dt_end, self.n_pulses) + self.tup
        # downsweep
        t = np.r_[t, np.flipud(t[1:])]
        t = np.cumsum(t)
        return t

    def generate_template(self, fs=1000):
        """
        Generates a spacer voltage template to cross-correlate with a voltage trace from a DAQ to
        detect a voltage trace.

        Parameters
        ----------
        fs : int
            DAQ sampling frequency.

        Returns
        -------
        numpy.array
            The template spacer signal.
        """
        t = self.times
        ns = int((t[-1] + self.tup * 10) * fs)
        sig = np.zeros(
            ns,
        )
        sig[(t * fs).astype(np.int32)] = 1
        sig[((t + self.tup) * fs).astype(np.int32)] = -1
        sig = np.cumsum(sig)
        return sig

    def add_spacer_states(self, sma=None, next_state='exit'):
        """
        Add spacer states to a state machine.

        Parameters
        ----------
        sma : pybpodapi.state_machine.StateMachine
            A Bpod state machine instance.
        next_state : str
            The name of the state to follow the spacer state.
        """
        assert next_state is not None
        t = self.times
        dt = np.diff(t, append=t[-1] + self.tup * 2)
        for i, time in enumerate(t):
            if sma is None:
                print(i, time, dt[i])
                continue
            next_loop = f'spacer_high_{i + 1:02d}' if i < len(t) - 1 else next_state
            sma.add_state(
                state_name=f'spacer_high_{i:02d}',
                state_timer=self.tup,
                state_change_conditions={'Tup': f'spacer_low_{i:02d}'},
                output_actions=[('BNC1', 255)],  # To FPGA
            )
            sma.add_state(
                state_name=f'spacer_low_{i:02d}',
                state_timer=dt[i] - self.tup,
                state_change_conditions={'Tup': next_loop},
                output_actions=[],
            )

    def find_spacers_from_fronts(self, fronts, fs=1000):
        """
        Given the timestamps and polarities of a digital signal, returns the timestamps of each
        signal.  This method first finds the locations where there are n consecutive pulses of the
        correct width then convolves this part of the signal with the template signal.

        This method may be relaxed in order to make it robust to noise in the signal.

        Parameters
        ----------
        fronts : dict[str, numpy.array]
            Dictionary with keys ('times', 'polarities') containing the timestamps and polarities
            of the signal fronts, respectively.
        fs : int
            The sampling frequency of the DAQ signal.

        Returns
        -------
        numpy.array
            The times of the protocol spacer signals.
        """
        n_pulses = (self.n_pulses * 2) - 1
        is_pulse = np.isclose(np.diff(fronts['times']), self.tup, rtol=1e-2)
        is_pulse = np.insert(is_pulse, 0, False)
        (ind,) = np.where(is_pulse)

        # Find consecutive pulses that are the correct length close together
        max_d = 1.0  # look for fronts less than 1 second apart
        consecutive = np.logical_and(np.diff(ind) == 2, np.diff(fronts['times'][ind]) < max_d)
        consecutive = np.pad(consecutive, 1, 'constant', constant_values=False)
        (edges,) = np.where(~consecutive)
        spacer_times = []
        for i in np.arange(edges.size - 1):
            if edges[i + 1] - edges[i] == n_pulses:  # This could be relaxed to allow for noise
                idx = np.arange(ind[edges[i]], ind[edges[i + 1] - 1] + 1)  # +1 to include final down
                t = fronts['times'][idx]
                ts = np.arange(t[0], t[-1], 1 / fs)  # Evenly resample at given frequency
                # Reconstruct trace where 1 = high, 0 = low
                signal = np.zeros_like(ts)
                ii = np.searchsorted(ts, t, side='left')
                signal[ii[ii < len(signal)]] = fronts['polarities'][idx[ii < len(signal)]]
                signal = np.cumsum(signal) + 1  # {-1, 0} -> {0, 1}
                try:
                    (spacer,) = self.find_spacers(signal, fs=fs)
                    spacer_times.append(spacer + t[0])
                except IndexError:
                    continue

        return np.array(spacer_times)

    def find_spacers(self, signal, threshold=0.9, fs=1000):
        """
        Find spacers in a voltage time series. Assumes that the signal is a digital signal between
        0 and 1.

        Parameters
        ----------
        signal : numpy.ndarray
            The signal in which to find the spacer.
        threshold : float
            The cross-correlation detection threshold.
        fs : int
            The sampling frequency of the DAQ signal.

        Returns
        -------
        numpy.ndarray
            An array containing the times of each spacer signal relative to the first sample.
        """
        template = self.generate_template(fs=fs)
        xcor = np.correlate(signal, template, mode='full') / np.sum(template)
        idetect = np.where(xcor > threshold)[0]
        iidetect = np.cumsum(np.diff(idetect, prepend=0) > 1)
        nspacers = iidetect[-1]
        tspacer = np.zeros(nspacers)
        for i in range(nspacers):
            ispacer = idetect[iidetect == i + 1]
            imax = np.argmax(xcor[ispacer])
            tspacer[i] = (ispacer[imax] - template.size + 1) / fs
        return tspacer

    def find_spacers_from_timestamps(self, timestamps: np.ndarray, atol: np.float64 = 1e-4) -> np.ndarray:
        """
        finds spacers in a series of timestamps. Returns the indices of the first spacer front

        Parameters
        ----------
        timestamps : np.ndarray
            an array of the timestamps to check
        atol : np.float64, optional
            absolute tolerance for the squared sum of the residuals

        Returns
        -------
        np.ndarray
            an array of the indices of the first front of a spacer
        """

        res = []
        for i in range(timestamps.shape[0] - self.n_pulses * 2):
            tcheck = timestamps[i : i + self.n_pulses * 2]
            tcheck = tcheck - tcheck[0] + self.times[0]
            res.append(np.sum((tcheck[:-1] - self.times) ** 2))  # squared sum of resituals
        res = np.array(res)

        return np.where(np.isclose(res, 0, atol=atol))[0]

    def find_spacers_from_positive_fronts(self, timestamps, fs=1000, prominence=4):
        """
        finds spacers in timestamps that consist of only the positive fronts.

        Parameters
        ----------
        timestamps : np.ndarray
            the positive fronts
        fs : int, optional
            sampling frequency used for resampling, by default 1000
        prominence : int, optional
            prominence for peak detection after convolution, passed to signal.find_peaks, by default 4
        """

        def digitize(tstamps, fs):
            # local helper to create a boolean vector from timestamped data sampled at fs
            t = np.arange(tstamps[0], tstamps[-1], 1 / fs)
            y = np.zeros_like(t)
            y[np.digitize(tstamps, t) - 1] = 1
            return y, t

        y, t = digitize(timestamps, fs)
        y_spacer, _ = digitize(self.times, fs)
        # convolve to find spacers
        y_c = np.convolve(y, y_spacer, mode='same')
        peak_inds, _ = signal.find_peaks(y_c, prominence=prominence)
        # adjust start time by spacer width
        w = (self.times[-1] - self.times[0]) / 2
        spacer_times_ = t[peak_inds] - w - self.times[0]
        # convert spacer onset times to to index into timestamps
        # spacer_ix = np.array([np.argmin((timestamps - t_s)**2) for t_s in spacer_times])
        spacer_ix = []
        spacer_times = []
        for spacer_time in np.sort(spacer_times_):
            dt = (timestamps - spacer_time) ** 2
            if np.min(dt) > 1 / fs:  # when first timestamp of spacer is missing
                spacer_ix.append(np.nan)  # the returned index is nan
                spacer_times.append(spacer_time)  # the returned time is the inferred time
            else:
                ix = np.argmin(dt)
                spacer_ix.append(ix)  # otherwise the returned index is the index of the fist spacer front
                spacer_times.append(timestamps[ix])  # and it's corresponding timestamp

        return np.array(spacer_ix), np.array(spacer_times)
