# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2026 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

import logging
import importlib
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hvsrpy
from hvsrpy.spectral_amplitude import (
    SpectralResult,
    compute_fourier_amplitude_spectra,
    compute_power_spectral_density,
    smooth_fourier_amplitude_spectra,
    smooth_spectra,
    plot_spectrum_results,
    plot_spectrum_summary,
    plot_spectrum_component,
    plot_spectra,
)
from testing_tools import unittest, TestCase

logger = logging.getLogger("hvsrpy")
logger.setLevel(level=logging.CRITICAL)


class TestSpectralAmplitude(TestCase):

    @classmethod
    def setUpClass(cls):
        dt = 0.01
        t = np.arange(0, 10, dt)
        f0 = 2.0
        ns = hvsrpy.TimeSeries(np.sin(2*np.pi*f0*t), dt)
        ew = hvsrpy.TimeSeries(0.5*np.sin(2*np.pi*f0*t), dt)
        vt = hvsrpy.TimeSeries(0.25*np.sin(2*np.pi*f0*t), dt)
        record = hvsrpy.SeismicRecording3C(ns, ew, vt)
        cls.records = [record, record]
        cls.settings = hvsrpy.HvsrTraditionalProcessingSettings(
            fft_settings=dict(n=2048)
        )

    def test_module_clean_import(self):
        module = importlib.import_module("hvsrpy.spectral_amplitude")
        self.assertTrue(hasattr(module, "compute_fourier_amplitude_spectra"))
        self.assertTrue(hasattr(module, "compute_power_spectral_density"))
        self.assertTrue(hasattr(module, "SpectralResult"))
        self.assertTrue(hasattr(module, "smooth_fourier_amplitude_spectra"))
        self.assertTrue(hasattr(module, "plot_spectrum_component"))
        self.assertTrue(hasattr(module, "plot_spectrum_results"))
        self.assertTrue(hasattr(module, "plot_spectrum_summary"))
        self.assertTrue(hasattr(module, "plot_spectra"))
        self.assertFalse(hasattr(module, "plot_fourier_amplitude_spectra"))
        self.assertFalse(hasattr(module, "plot_power_spectral_density"))

    def test_plotting_module_imports(self):
        module = importlib.import_module("hvsrpy.spectral_plotting")
        self.assertTrue(hasattr(module, "plot_spectrum_component"))
        self.assertTrue(hasattr(module, "plot_spectrum_results"))
        self.assertTrue(hasattr(module, "plot_spectrum_summary"))
        self.assertTrue(hasattr(module, "plot_spectra"))
        self.assertFalse(hasattr(module, "plot_fourier_amplitude_spectra"))
        self.assertFalse(hasattr(module, "plot_power_spectral_density"))

    def test_spectral_result_dataclass_normalizes_shapes(self):
        result = SpectralResult(
            frequency=np.array([1.0, 2.0, 3.0]),
            ns=np.array([1.0, 2.0, 3.0]),
            ew=np.array([2.0, 3.0, 4.0]),
            vt=np.array([3.0, 4.0, 5.0]),
            spectrum_type="fas",
            is_smoothed=False,
        )

        self.assertEqual(result.ns.shape, (1, 3))
        self.assertEqual(result.ew.shape, (1, 3))
        self.assertFalse(result.has_horizontal)
        self.assertEqual(result.n_records, 1)
        self.assertEqual(result.n_frequencies, 3)
        self.assertTrue("frequency" in result)
        self.assertEqual(result["spectrum_type"], "fas")

    def test_compute_fourier_amplitude_spectra(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        self.assertIsInstance(spectra, SpectralResult)
        self.assertTrue("frequency" in spectra)
        self.assertEqual(spectra.spectrum_type, "fas")
        self.assertFalse(spectra.is_smoothed)
        self.assertEqual(spectra["ns"].shape[0], len(self.records))
        self.assertEqual(spectra["ew"].shape, spectra["ns"].shape)
        self.assertEqual(spectra["vt"].shape, spectra["ns"].shape)
        self.assertEqual(spectra["horizontal"].shape, spectra["ns"].shape)

        # frequency index 0 is DC; peak should be near 2 Hz.
        peak_idx = np.argmax(spectra["ns"][0, 1:]) + 1
        self.assertAlmostEqual(spectra["frequency"][peak_idx], 2.0, places=1)
        self.assertArrayAlmostEqual(
            spectra["horizontal"][0],
            np.sqrt(spectra["ns"][0] * spectra["ew"][0]),
        )

    def test_fft_length_considers_all_three_components(self):
        dt = 0.01
        short = hvsrpy.TimeSeries(np.ones(128), dt)
        long_ew = hvsrpy.TimeSeries(np.ones(40000), dt)
        long_vt = hvsrpy.TimeSeries(np.ones(256), dt)
        record = SimpleNamespace(ns=short, ew=long_ew, vt=long_vt)

        spectra = compute_fourier_amplitude_spectra([record], self.settings)

        expected_n = 65536
        expected_frequency = np.fft.rfftfreq(expected_n, dt)
        self.assertEqual(spectra["frequency"].shape, expected_frequency.shape)
        self.assertEqual(spectra["ns"].shape, (1, expected_frequency.size))
        self.assertEqual(spectra["ew"].shape, (1, expected_frequency.size))
        self.assertEqual(spectra["vt"].shape, (1, expected_frequency.size))

    def test_single_azimuth_horizontal_fas(self):
        settings = hvsrpy.HvsrTraditionalSingleAzimuthProcessingSettings(
            fft_settings=dict(n=2048),
            azimuth_in_degrees=30.0,
        )

        spectra = compute_fourier_amplitude_spectra(
            self.records,
            settings,
            include_horizontal=True,
        )

        theta = np.radians(settings.azimuth_in_degrees)
        fft_settings = dict(n=2*(spectra["horizontal"].shape[1]-1))
        expected_horizontal = np.abs(
            np.fft.rfft(
                self.records[0].ns.amplitude*np.cos(theta)
                + self.records[0].ew.amplitude*np.sin(theta),
                **fft_settings
            )
        )
        self.assertArrayAlmostEqual(spectra["horizontal"][0], expected_horizontal)

    def test_smooth_fourier_amplitude_spectra_from_settings(self):
        spectra = compute_fourier_amplitude_spectra(self.records, self.settings)
        self.settings.smoothing = dict(
            operator="konno_and_ohmachi",
            bandwidth=40,
            center_frequencies_in_hz=np.geomspace(0.2, 20, 32),
        )
        smoothed = smooth_fourier_amplitude_spectra(spectra, settings=self.settings)

        self.assertIsInstance(smoothed, SpectralResult)
        self.assertEqual(smoothed.spectrum_type, "fas")
        self.assertTrue(smoothed.is_smoothed)
        self.assertArrayAlmostEqual(
            smoothed["frequency"],
            self.settings.smoothing["center_frequencies_in_hz"],
        )
        self.assertEqual(smoothed["ns"].shape, (len(self.records), 32))
        self.assertEqual(smoothed["ew"].shape, (len(self.records), 32))
        self.assertEqual(smoothed["vt"].shape, (len(self.records), 32))

    def test_smooth_fourier_amplitude_spectra_nyquist_guard(self):
        spectra = compute_fourier_amplitude_spectra(self.records, self.settings)
        self.settings.smoothing = dict(
            operator="konno_and_ohmachi",
            bandwidth=40,
            center_frequencies_in_hz=np.array([1000.0]),
        )
        with self.assertRaises(ValueError):
            smooth_fourier_amplitude_spectra(
                spectra,
                settings=self.settings,
            )

    def test_dt_validation_catches_mismatch_within_record(self):
        ns = hvsrpy.TimeSeries(np.ones(32), 0.01)
        ew = hvsrpy.TimeSeries(np.ones(32), 0.02)
        vt = hvsrpy.TimeSeries(np.ones(32), 0.01)
        record = SimpleNamespace(ns=ns, ew=ew, vt=vt)

        with self.assertRaisesRegex(ValueError, "inconsistent component sampling intervals"):
            compute_fourier_amplitude_spectra([record], self.settings)

    def test_dt_validation_catches_mismatch_across_records(self):
        first = self.records[0]
        dt = 0.02
        t = np.arange(0, 10, dt)
        second = hvsrpy.SeismicRecording3C(
            hvsrpy.TimeSeries(np.sin(2*np.pi*2*t), dt),
            hvsrpy.TimeSeries(np.sin(2*np.pi*2*t), dt),
            hvsrpy.TimeSeries(np.sin(2*np.pi*2*t), dt),
        )

        with self.assertRaisesRegex(ValueError, "share a common dt_in_seconds"):
            compute_fourier_amplitude_spectra([first, second], self.settings)

    def test_invalid_record_structure_produces_clear_error(self):
        invalid_record = SimpleNamespace(
            ns=SimpleNamespace(amplitude=np.ones(16), dt_in_seconds=0.01),
            ew=SimpleNamespace(amplitude=np.ones(16), dt_in_seconds=0.01, n_samples=16),
            vt=SimpleNamespace(amplitude=np.ones(16), dt_in_seconds=0.01, n_samples=16),
        )

        with self.assertRaisesRegex(ValueError, "missing required attribute\\(s\\): n_samples"):
            compute_fourier_amplitude_spectra([invalid_record], self.settings)

    def test_smooth_fourier_amplitude_spectra_invalid_operator(self):
        spectra = compute_fourier_amplitude_spectra(self.records, self.settings)
        self.settings.smoothing = dict(
            operator="not_a_real_operator",
            bandwidth=40,
            center_frequencies_in_hz=np.geomspace(0.2, 20, 32),
        )

        with self.assertRaisesRegex(ValueError, "not recognized"):
            smooth_fourier_amplitude_spectra(spectra, settings=self.settings)

    def test_invalid_single_azimuth_input_raises_clear_error(self):
        settings = hvsrpy.HvsrTraditionalSingleAzimuthProcessingSettings(
            fft_settings=dict(n=2048),
            azimuth_in_degrees=np.nan,
        )

        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            compute_fourier_amplitude_spectra(
                self.records,
                settings,
                include_horizontal=True,
            )

    def test_compute_power_spectral_density(self):
        spectra = compute_power_spectral_density(
            self.records,
            self.settings,
            include_horizontal=True,
        )

        self.assertIsInstance(spectra, SpectralResult)
        self.assertTrue("frequency" in spectra)
        self.assertEqual(spectra.spectrum_type, "psd")
        self.assertEqual(spectra["ns"].shape[0], len(self.records))
        self.assertEqual(spectra["ew"].shape, spectra["ns"].shape)
        self.assertEqual(spectra["vt"].shape, spectra["ns"].shape)
        self.assertEqual(spectra["horizontal"].shape, spectra["ns"].shape)
        self.assertTrue(np.all(spectra["ns"] >= 0))

    def test_compute_power_spectral_density_single_azimuth(self):
        settings = hvsrpy.HvsrTraditionalSingleAzimuthProcessingSettings(
            fft_settings=dict(n=2048),
            azimuth_in_degrees=30.0,
        )
        spectra = compute_power_spectral_density(
            self.records,
            settings,
            include_horizontal=True,
        )

        theta = np.radians(settings.azimuth_in_degrees)
        horizontal = (
            self.records[0].ns.amplitude*np.cos(theta)
            + self.records[0].ew.amplitude*np.sin(theta)
        )
        fft_settings = dict(n=2*(spectra["horizontal"].shape[1]-1))
        fft = np.fft.rfft(horizontal, **fft_settings)
        expected_horizontal = 2*np.real(np.conjugate(fft) * fft)
        expected_horizontal /= self.records[0].ns.n_samples
        expected_horizontal /= self.records[0].ns.fs
        self.assertArrayAlmostEqual(spectra["horizontal"][0], expected_horizontal)

    def test_smooth_spectra_psd(self):
        spectra = compute_power_spectral_density(self.records, self.settings)
        self.settings.smoothing = dict(
            operator="konno_and_ohmachi",
            bandwidth=40,
            center_frequencies_in_hz=np.geomspace(0.2, 20, 32),
        )
        smoothed = smooth_spectra(spectra, settings=self.settings)

        self.assertIsInstance(smoothed, SpectralResult)
        self.assertEqual(smoothed.spectrum_type, "psd")
        self.assertTrue(smoothed.is_smoothed)
        self.assertArrayAlmostEqual(
            smoothed["frequency"],
            self.settings.smoothing["center_frequencies_in_hz"],
        )
        self.assertEqual(smoothed["ns"].shape, (len(self.records), 32))

    def test_smooth_spectra_accepts_legacy_dict_input(self):
        spectra = compute_power_spectral_density(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        self.settings.smoothing = dict(
            operator="konno_and_ohmachi",
            bandwidth=40,
            center_frequencies_in_hz=np.geomspace(0.2, 20, 16),
        )

        legacy = spectra.as_dict()
        smoothed = smooth_spectra(legacy, settings=self.settings)

        self.assertIsInstance(smoothed, SpectralResult)
        self.assertEqual(smoothed["horizontal"].shape, (len(self.records), 16))

    def test_plot_spectrum_component_on_provided_axis(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        fig, ax = plt.subplots()
        returned = plot_spectrum_component(
            spectra,
            component="horizontal",
            ax=ax,
            title="Horizontal",
        )
        self.assertIs(returned, ax)
        self.assertEqual(len(ax.get_lines()), len(self.records) + 1)
        self.assertEqual(ax.get_ylabel(), "Fourier Amplitude")
        self.assertEqual(ax.get_title(), "Horizontal")
        plt.close(fig)

    def test_plot_spectrum_component_can_be_composed_across_axes(self):
        spectra = compute_power_spectral_density(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        fig, axes = plt.subplots(2, 2)
        components = ["ns", "ew", "vt", "horizontal"]
        for ax, component in zip(axes.flat, components):
            plot_spectrum_component(
                spectra,
                component=component,
                ax=ax,
                title=component.upper(),
            )
        self.assertEqual(axes[0, 0].get_title(), "NS")
        self.assertEqual(axes[1, 1].get_ylabel(), "Power Spectral Density")
        plt.close(fig)

    def test_plot_spectrum_component_invalid_component(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        with self.assertRaisesRegex(ValueError, "Use 'ns', 'ew', 'vt', or 'horizontal'"):
            plot_spectrum_component(spectra, component="bad_component")

    def test_plot_spectrum_component_missing_horizontal_raises(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=False,
        )
        with self.assertRaisesRegex(ValueError, "horizontal is not present"):
            plot_spectrum_component(spectra, component="horizontal")

    def test_plot_spectrum_results_for_fas(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        fig, axes = plot_spectrum_results(
            spectra,
            spectrum_type="fas",
            include_horizontal=True,
        )
        self.assertEqual(len(axes), 4)
        self.assertEqual(axes[0].get_ylabel(), "Fourier Amplitude")
        self.assertEqual(axes[-1].get_xlabel(), "Frequency (Hz)")
        plt.close(fig)

    def test_plot_spectrum_results_infers_type_from_dataclass(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        fig, axes = plot_spectrum_results(
            spectra,
            include_horizontal=True,
        )
        self.assertEqual(axes[0].get_ylabel(), "Fourier Amplitude")
        plt.close(fig)

    def test_plot_spectrum_results_for_psd(self):
        spectra = compute_power_spectral_density(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        fig, axes = plot_spectrum_results(
            spectra,
            spectrum_type="psd",
            include_horizontal=True,
        )
        self.assertEqual(len(axes), 4)
        self.assertEqual(axes[0].get_ylabel(), "Power Spectral Density")
        plt.close(fig)

    def test_plot_spectrum_results_without_horizontal(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=False,
        )
        fig, axes = plot_spectrum_results(
            spectra,
            spectrum_type="fas",
            include_horizontal=True,
        )
        self.assertEqual(len(axes), 3)
        plt.close(fig)

    def test_plot_spectrum_summary_for_fas(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        fig, ax = plot_spectrum_summary(
            spectra,
            spectrum_type="fas",
            include_horizontal=True,
        )
        self.assertEqual(len(ax.get_lines()), 4)
        self.assertEqual(ax.get_ylabel(), "Fourier Amplitude")
        plt.close(fig)

    def test_plot_spectrum_summary_accepts_legacy_dict_input(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        fig, ax = plot_spectrum_summary(
            spectra.as_dict(),
            spectrum_type="fas",
            include_horizontal=True,
        )
        self.assertEqual(len(ax.get_lines()), 4)
        plt.close(fig)

    def test_plot_spectrum_summary_without_horizontal(self):
        spectra = compute_power_spectral_density(
            self.records,
            self.settings,
            include_horizontal=False,
        )
        fig, ax = plot_spectrum_summary(
            spectra,
            spectrum_type="psd",
            include_horizontal=True,
        )
        self.assertEqual(len(ax.get_lines()), 3)
        self.assertEqual(ax.get_ylabel(), "Power Spectral Density")
        plt.close(fig)

    def test_plot_spectra_for_psd(self):
        spectra = compute_power_spectral_density(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        fig, ax = plot_spectra(
            spectra,
            spectrum_type="psd",
            include_horizontal=True,
        )
        self.assertEqual(len(ax.get_lines()), 4)
        self.assertEqual(ax.get_ylabel(), "Power Spectral Density")
        plt.close(fig)

    def test_plot_spectra_invalid_selection(self):
        spectra = compute_fourier_amplitude_spectra(self.records, self.settings)
        with self.assertRaisesRegex(ValueError, "Use 'fas' or 'psd'"):
            plot_spectra(spectra, spectrum_type="invalid")

    def test_plot_spectrum_results_invalid_selection(self):
        spectra = compute_fourier_amplitude_spectra(self.records, self.settings)
        with self.assertRaisesRegex(ValueError, "Use 'fas' or 'psd'"):
            plot_spectrum_results(spectra, spectrum_type="invalid")

    def test_plot_spectrum_summary_invalid_statistic(self):
        spectra = compute_fourier_amplitude_spectra(self.records, self.settings)
        with self.assertRaisesRegex(ValueError, "Use 'mean' or 'median'"):
            plot_spectrum_summary(
                spectra,
                spectrum_type="fas",
                statistic="mode",
            )

    def test_compute_fourier_amplitude_spectra_with_smoothing(self):
        self.settings.smoothing = dict(
            operator="konno_and_ohmachi",
            bandwidth=40,
            center_frequencies_in_hz=np.geomspace(0.2, 20, 16),
        )
        spectra = compute_fourier_amplitude_spectra(
            self.records, self.settings, include_horizontal=True, smooth=True)
        self.assertEqual(spectra["ns"].shape, (len(self.records), 16))
        self.assertEqual(spectra["horizontal"].shape, (len(self.records), 16))


if __name__ == "__main__":
    unittest.main()
