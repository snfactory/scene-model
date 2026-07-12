"""Minimal SNIFS cube and spectrum I/O compatibility layer.

This module is derived from the public ``pySNIFS`` implementation distributed
by the Nearby Supernova Factory under the MIT license.  It intentionally keeps
only the I/O surface used by scene-model and its installed command-line tools.

Copyright (c) 2016 Nearby Supernova Factory
Original module author: Emmanuel Pecontal
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter


class spectrum:
    """A one-dimensional, optionally variance-bearing spectrum."""

    def __init__(
        self,
        data_file=None,
        var_file=None,
        no=None,
        x=None,
        data=None,
        var=None,
        start=None,
        step=None,
        nx=None,
    ):
        self.file = None
        if data_file is not None:
            self._read(data_file, var_file=var_file, no=no)
        elif x is not None:
            self._from_coordinates(x, data=data, var=var)
        else:
            self._from_regular(data=data, var=var, start=start, step=step, nx=nx)

        self.data = np.asarray(self.data)
        self.x = np.asarray(self.x)
        if self.var is not None:
            self.var = np.asarray(self.var)
        self.has_var = self.var is not None
        if self.len is not None:
            # Retained mutable-record fields used by historical callers.
            self.index_list = np.arange(self.len).tolist()
            self.intervals = [(0, self.len)]
        self.curs_val = []
        self.cid = None

    def _read(self, data_file, *, var_file, no):
        with fits.open(data_file, ignore_missing_end=True) as hdus:
            if "EURO3D" in hdus[0].header:
                if no is None:
                    raise ValueError("The user must provide the spectrum number in the datacube")
                identifiers = hdus[1].data.field("SPEC_ID").tolist()
                if no not in identifiers:
                    self.data = self.var = self.len = self.step = None
                    self.start = self.x = None
                    return
                index = identifiers.index(no)
                self.data = np.array(hdus[1].data.field("DATA_SPE")[index])
                self.var = (
                    np.array(hdus[1].data.field("STAT_SPE")[index])
                    if "STAT_SPE" in hdus[1].columns.names
                    else None
                )
                self.len = hdus[1].data.field("SPEC_LEN")[index]
                self.step = hdus[1].header.get("CDELTS")
                self.start = (
                    hdus[1].header.get("CRVALS")
                    + hdus[1].data.field("SPEC_STA")[index] * self.step
                )
                self.x = np.arange(self.len) * self.step + self.start
                return

            self.data = np.array(hdus[0].data)
            if len(hdus) == 2:
                self.var = np.array(hdus[1].data)
            elif var_file is not None:
                with fits.open(var_file, ignore_missing_end=True) as var_hdus:
                    self.var = np.array(var_hdus[0].data)
            else:
                self.var = None
            if self.var is not None and len(self.var) != len(self.data):
                raise ValueError("Data and variance spectra must have the same length")
            self.len = hdus[0].header.get("NAXIS1")
            self.start = hdus[0].header.get("CRVAL1")
            self.step = hdus[0].header.get("CDELT1")
            self.x = self.start + np.arange(self.len) * self.step

    def _from_regular(self, *, data, var, start, step, nx):
        if data is None:
            if nx is None:
                raise ValueError("Not enough parameters to fill the spectrum data field")
            self.data = np.zeros(nx)
            self.var = np.zeros(nx) if None not in (start, step) else None
            self.len = nx
        else:
            self.data = data
            self.len = len(data)
            if var is not None and len(var) != self.len:
                raise ValueError("data and variance array must have the same length")
            self.var = var
        self.start = 0 if start is None else start
        self.step = 1 if step is None else step
        self.x = self.start + np.arange(self.len) * self.step

    def _from_coordinates(self, x, *, data, var):
        self.start = self.step = None
        self.len = len(x)
        if data is None:
            self.data = np.zeros(self.len)
            self.var = np.zeros(self.len)
        else:
            if len(data) != self.len:
                raise ValueError("x and data arrays must have the same size")
            if var is not None and len(var) != self.len:
                raise ValueError("data and var arrays must have the same size")
            self.data = data
            self.var = var
        self.x = x

    def WR_fits_file(self, filename, header_list=None):
        """Write a regularly sampled spectrum and optional variance."""
        if self.start is None or self.step is None or self.data is None:
            raise ValueError("Only regularly sampled spectra can be saved as fits files.")

        primary = fits.PrimaryHDU(np.asarray(self.data))
        primary.header["CRVAL1"] = self.start
        primary.header["CDELT1"] = self.step
        if header_list is not None:
            excluded_prefixes = ("TUNIT", "TTYPE", "TFORM", "TDISP", "NAXIS", "CRVAL", "CDELT", "CRPIX")
            excluded = {
                "EXTNAME", "XTENSION", "GCOUNT", "PCOUNT", "BITPIX",
                "CTYPES", "CRVALS", "CDELTS", "CRPIXS", "TFIELDS",
            }
            for key, value in header_list:
                if not key.startswith(excluded_prefixes) and key not in excluded:
                    primary.header[key] = value

        hdus = [primary]
        if self.has_var:
            variance = fits.ImageHDU(np.asarray(self.var), name="VARIANCE")
            variance.header["CRVAL1"] = self.start
            variance.header["CDELT1"] = self.step
            hdus.append(variance)
        fits.HDUList(hdus).writeto(filename, overwrite=True)


class SNIFS_cube:
    """SNIFS Euro3D/FITS3D cube used by scene-model extraction."""

    spxSize = 0.43

    def __init__(
        self,
        e3d_file=None,
        fits3d_file=None,
        slices=None,
        lbda=None,
        threshold=1e20,
        nodata=False,
    ):
        interval, stack = self._validate_slices(slices)
        self.data = None
        self.var = None
        if e3d_file is not None:
            self._read_e3d(e3d_file, interval, stack, threshold, nodata)
        elif fits3d_file is not None:
            self._read_fits3d(fits3d_file, interval, stack, nodata)
        else:
            self._empty(lbda)

    @staticmethod
    def _validate_slices(slices):
        if slices is None:
            return None, False
        if not isinstance(slices, list) or len(slices) not in (2, 3):
            raise ValueError("The wavelength range must be given as a list of two or three integer positive values")
        if not all(isinstance(value, int) and value >= 0 for value in slices):
            raise ValueError("The wavelength range must be given as a list of two or three integer positive values")
        values = list(slices)
        if len(values) == 3 and values[2] == 0:
            raise ValueError("The slices step cannot be set to 0")
        if values[0] > values[1]:
            values[0], values[1] = values[1], values[0]
        return values, len(values) == 3

    def _read_e3d(self, path, interval, stack, threshold, nodata):
        with fits.open(path, ignore_missing_end=True) as hdus:
            general = dict(hdus[0].header.items())
            if general.get("EURO3D") not in ("T", True):
                raise ValueError("Invalid E3d file ('EURO3D' keyword)")
            self.from_e3d_file = True
            self.e3d_file = path
            self.e3d_data_header = dict(hdus[1].header.items())
            self.e3d_grp_hdu = hdus[2].copy()
            self.e3d_extra_hdu_list = [hdu.copy() for hdu in hdus[3:]]
            reference_start = hdus[1].header["CRVALS"]
            self.lstep = hdus[1].header["CDELTS"]
            variance = hdus[1].data.field("STAT_SPE") if "STAT_SPE" in hdus[1].columns.names else None
            data = hdus[1].data.field("DATA_SPE")
            starts = hdus[1].data.field("SPEC_STA")
            lengths = hdus[1].data.field("SPEC_LEN")
            common_first, common_last = max(starts), min(lengths + starts)
            common_start = reference_start + common_first * self.lstep
            transposed = np.array([
                data[index][common_first - starts[index]:common_last - starts[index]]
                for index in range(len(data))
            ]).T
            if variance is not None:
                transposed_var = np.array([
                    variance[index][common_first - starts[index]:common_last - starts[index]]
                    for index in range(len(variance))
                ]).T
                transposed_var = np.where(np.abs(transposed_var) > threshold, threshold, transposed_var)
            wavelengths = np.arange(len(transposed)) * self.lstep + common_start

            if interval is None:
                first, last, stride = common_first, common_last, 1
            else:
                first = max(common_first, interval[0])
                last = min(common_last, interval[1])
                stride = interval[2] if len(interval) == 3 else 1
            if last - first < stride:
                raise ValueError("Slice step incompatible with requested slices interval")
            offset = first - common_first
            if not stack:
                if not nodata:
                    self.data = transposed[offset:last - common_first:stride]
                    if variance is not None:
                        self.var = transposed_var[offset:last - common_first:stride]
                self.lbda = wavelengths[offset:last - common_first:stride]
            else:
                center = offset + stride // 2
                stop = last - common_first + stride // 2
                if not nodata:
                    self.data = uniform_filter(transposed, (stride, 1))[center:stop:stride]
                    if variance is not None:
                        self.var = uniform_filter(transposed_var, (stride, 1))[center:stop:stride] / stride
                self.lbda = wavelengths[center:stop:stride]
            self.lstep *= stride
            self.lstart = self.lbda[0]
            self.x = np.array(hdus[1].data.field("XPOS"))
            self.y = np.array(hdus[1].data.field("YPOS"))
            self.no = np.array(hdus[1].data.field("SPEC_ID"))
            if len(hdus) > 3 and hdus[3].name == "TIGERTBL":
                identifiers = hdus[3].data.field("NO").tolist()
                indices = [identifiers.index(value) for value in self.no]
                self.i = np.array(hdus[3].data.field("I")[indices] + 7)
                self.j = np.array(hdus[3].data.field("J")[indices] + 7)
            else:
                self.i = np.round(self.x / self.spxSize).astype("i") + 7
                self.j = np.round(self.y / self.spxSize).astype("i") + 7
        self.nslice = len(self.lbda)
        self.lend = self.lstart + (self.nslice - 1) * self.lstep
        self.nlens = len(self.x)

    def _read_fits3d(self, path, interval, stack, nodata):
        with fits.open(path, ignore_missing_end=True) as hdus:
            header = dict(hdus[0].header.items())
            if header["NAXIS"] != 3:
                raise ValueError(f"Invalid 3D file: NAXIS={header['NAXIS']} != 3")
            self.from_e3d_file = False
            self.e3d_data_header = header
            nslice, nx, ny = header["NAXIS3"], header["NAXIS1"], header["NAXIS2"]
            startx, starty, startl = header["CRVAL1"], header["CRVAL2"], header["CRVAL3"]
            stepx, stepy, self.lstep = header["CDELT1"], header["CDELT2"], header["CDELT3"]
            self.fits3d_file = path
            wavelengths = np.arange(nslice) * self.lstep + startl
            data = np.reshape(hdus[0].data[:, :, ::-1], (nslice, nx * ny))
            if "VARIANCE" in [hdu.name for hdu in hdus]:
                if hdus[0].data.shape != hdus["VARIANCE"].data.shape:
                    raise AssertionError("Variance and primary extensions have different shapes")
                variance = np.reshape(hdus["VARIANCE"].data[:, :, ::-1], (nslice, nx * ny))
            else:
                variance = None

            if interval is None:
                first, last, stride = 0, nslice - 1, 1
            else:
                first, last = max(0, interval[0]), min(nslice - 1, interval[1])
                stride = interval[2] if len(interval) == 3 else 1
            if last - first < stride:
                raise ValueError("Slice step incompatible with requested slices interval")
            # Preserve pySNIFS behavior: a two-item interval is descriptive;
            # only three-item meta-slices alter FITS3D data.
            if not stack:
                if not nodata:
                    self.data = data
                    self.var = variance
                self.lbda = wavelengths
            else:
                center, stop = first + stride // 2, last + stride // 2
                if not nodata:
                    self.data = uniform_filter(data, (stride, 1))[center:stop:stride]
                    if variance is not None:
                        self.var = uniform_filter(variance, (stride, 1))[center:stop:stride] / stride
                self.lbda = wavelengths[center:stop:stride]
            self.lstep *= stride
            self.lstart = self.lbda[0]
            self.i = nx - 1 - np.ravel(np.indices((nx, ny))[1])
            self.j = np.ravel(np.indices((nx, ny))[0])
            self.x = self.i * stepx + startx
            self.y = self.j * stepy + starty
            self.no = nx * (np.arange(nx * ny) % nx) + np.arange(nx * ny) // nx + 1

        if self.data is not None:
            keep = np.where(np.min(np.isnan(self.data.T), axis=1) == False)[0]  # noqa: E712
            self.data = self.data[:, keep]
            if self.var is not None:
                self.var = self.var[:, keep]
            self.i, self.j = self.i[keep], self.j[keep]
            self.x, self.y, self.no = self.x[keep], self.y[keep], self.no[keep]
        self.nslice = len(self.lbda)
        self.lend = self.lstart + (self.nslice - 1) * self.lstep
        self.nlens = len(self.x)

    def _empty(self, lbda):
        self.from_e3d_file = False
        self.nlens = 225
        if lbda is None:
            self.data = np.zeros(self.nlens)
            self.lbda = self.nslice = None
            return
        self.lbda = np.asarray(lbda)
        delta = self.lbda[1:] - self.lbda[:-1]
        self.lstep = delta.mean()
        if not np.allclose(delta, self.lstep):
            raise ValueError("Input wavelength ramp is not linear.")
        self.lstart, self.lend = self.lbda[0], self.lbda[-1]
        self.nslice = len(self.lbda)
        i, j = [array.ravel() for array in np.meshgrid(np.arange(15), np.arange(14, -1, -1))]
        self.x = (i - 7) * self.spxSize
        self.y = (j - 7) * self.spxSize
        self.no = np.arange(1, self.nlens + 1).reshape(15, 15).T.ravel()
        self.data = np.zeros((self.nslice, self.nlens))
        self.i, self.j = i, j

    def slice2d(self, n=None, coord="w", weight=None, var=False, nx=15, ny=15, NAN=True):
        """Return an individual slice or summed wavelength interval as an image."""
        if weight is not None:
            raise NotImplementedError("Weighted slice extraction is not used by scene-model")
        if n is None:
            raise ValueError("Slices to be averaged must be given either as a list or as a weight spectrum")
        if isinstance(n, list):
            if len(n) != 2:
                raise ValueError("The list must have 2 values")
            first_value, last_value = sorted(n)
            if coord == "p":
                first, last = first_value, last_value
            elif coord == "w":
                first = np.argmin((self.lbda - first_value) ** 2, axis=-1)
                last = np.argmin((self.lbda - last_value) ** 2, axis=-1)
            else:
                raise ValueError("Coord. flag should be 'p' or 'w'")
            if first == last:
                last += 1
        else:
            if coord == "p":
                n = int(np.round(n))
                first, last = n, n + 1
            elif coord == "w":
                first = np.argmin((self.lbda - n) ** 2, axis=-1)
                last = first + 1
            else:
                raise ValueError("Coord. flag should be 'p' or 'w'")
        if first < 0 or last > self.data.shape[0]:
            raise IndexError(f"No slice #{n}")
        image = np.zeros((nx, ny), self.data.dtype)
        if NAN:
            image *= np.nan
        if var is True:
            array = self.var
        elif var:
            array = getattr(self, var)
        else:
            array = self.data
        image[np.asarray(self.j), np.asarray(self.i)] = np.sum(array[first:last], axis=0)
        return image

    def WR_e3d_file(self, filename):
        if not self.from_e3d_file:
            raise NotImplementedError("Writing e3d file from scratch not yet implemented")
        _write_e3d_file(
            self.data.T,
            None if self.var is None else self.var.T,
            self.no.tolist(),
            [self.lstart] * self.nlens,
            self.lstep,
            self.x.tolist(),
            self.y.tolist(),
            filename,
            self.e3d_data_header,
            self.e3d_grp_hdu,
            self.e3d_extra_hdu_list,
            nslice=self.nslice,
        )

    def WR_3d_fits(self, filename, header=None, mode="w+"):
        if header is None:
            header = self.e3d_data_header.copy()
            excluded = {
                "XTENSION", "BITPIX", "GCOUNT", "PCOUNT", "TFIELDS", "EXTNAME",
                "TFORM", "TUNIT", "TDISP", "CTYPES", "CRVALS", "CDELTS", "CRPIXS",
            }
            header = {key: value for key, value in header.items() if key not in excluded and not key.startswith(("NAXIS", "TTYPE"))}
        primary = fits.PrimaryHDU(np.asarray([self.slice2d(index, coord="p") for index in range(self.nslice)]))
        for key, value in header.items():
            primary.header[key] = value
        start = -7 * self.spxSize
        for axis, value in ((1, start), (2, start), (3, self.lstart)):
            primary.header[f"CRVAL{axis}"] = value
            primary.header[f"CDELT{axis}"] = self.spxSize if axis < 3 else self.lstep
            primary.header[f"CRPIX{axis}"] = 1
        hdus = [primary]
        if self.var is not None:
            variance = fits.ImageHDU(
                np.asarray([self.slice2d(index, coord="p", var=True) for index in range(self.nslice)]),
                name="VARIANCE",
            )
            for axis, value in ((1, start), (2, start), (3, self.lstart)):
                variance.header[f"CRVAL{axis}"] = value
                variance.header[f"CDELT{axis}"] = self.spxSize if axis < 3 else self.lstep
                variance.header[f"CRPIX{axis}"] = 1
            hdus.append(variance)
        fits.HDUList(hdus).writeto(filename, overwrite=(mode == "w+"))


def _write_e3d_file(
    data_list,
    var_list,
    no_list,
    start_list,
    step,
    xpos_list,
    ypos_list,
    filename,
    data_header,
    grp_hdu,
    extra_hdu_list,
    nslice=None,
):
    """Private Euro3D writer retained for :meth:`SNIFS_cube.WR_e3d_file`."""
    start = max(start_list)
    spec_sta = [int((value - start) / step + 0.5 * np.sign(value - start)) for value in start_list]
    spec_len = [len(values) for values in data_list]
    columns = [
        fits.Column(name="SPEC_ID", format="J", array=no_list),
        fits.Column(name="SELECTED", format="J", array=[0] * len(data_list)),
        fits.Column(name="NSPAX", format="J", array=[1] * len(data_list)),
        fits.Column(name="SPEC_LEN", format="J", array=spec_len),
        fits.Column(name="SPEC_STA", format="J", array=spec_sta),
        fits.Column(name="XPOS", format="E", array=xpos_list),
        fits.Column(name="YPOS", format="E", array=ypos_list),
        fits.Column(name="GROUP_N", format="J", array=[1] * len(data_list)),
        fits.Column(name="SPAX_ID", format="1A1", array=[" "] * len(data_list)),
    ]
    if nslice is None:
        columns.append(fits.Column(name="DATA_SPE", format="PD()", array=np.asarray(data_list, dtype="O")))
        quality = np.asarray([[0 for _ in values] for values in data_list], dtype="O")
        columns.append(fits.Column(name="QUAL_SPE", format="PJ()", array=quality))
        if var_list is not None:
            columns.append(fits.Column(name="STAT_SPE", format="PD()", array=np.asarray(var_list, dtype="O")))
    else:
        columns.append(fits.Column(name="DATA_SPE", format=f"{nslice}D()", array=data_list))
        columns.append(fits.Column(name="QUAL_SPE", format=f"{nslice}J", array=[[0 for _ in values] for values in data_list]))
        if var_list is not None:
            columns.append(fits.Column(name="STAT_SPE", format=f"{nslice}D()", array=var_list))

    table = fits.BinTableHDU.from_columns(columns)
    table.header["CTYPES"] = " "
    table.header["CRVALS"] = start
    table.header["CDELTS"] = step
    table.header["CRPIXS"] = 1
    table.header["EXTNAME"] = "E3D_DATA"
    excluded = {
        "XTENSION", "BITPIX", "GCOUNT", "PCOUNT", "TFIELDS", "EXTNAME",
        "TFORM", "TUNIT", "TDISP", "CTYPES", "CRVALS", "CDELTS", "CRPIXS",
    }
    for key, value in data_header.items():
        if key not in excluded and not key.startswith(("NAXIS", "TTYPE", "TFORM")):
            table.header[key] = value

    primary = fits.PrimaryHDU()
    primary.header["EURO3D"] = True
    primary.header["E3D_ADC"] = False
    primary.header["E3D_VERS"] = "1.0"
    fits.HDUList([primary, table, grp_hdu] + list(extra_hdu_list)).writeto(filename, overwrite=True)
