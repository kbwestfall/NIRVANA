
from IPython import embed

import numpy

from astropy import convolution

from nirvana.models import beam
from nirvana.models import oned
from nirvana.models.geometry import projected_polar
from nirvana.data import manga
from nirvana.tests.util import requires_pyfftw, remote_data_file, requires_remote, dap_test_daptype

def test_convolve():
    """
    Test that the results of the convolution match astropy.
    """
    synth = beam.gauss2d_kernel(73, 3.)
    astsynth = convolution.convolve_fft(synth, synth, fft_pad=False, psf_pad=False,
                                        boundary='wrap')
    intsynth = beam.convolve_fft(synth, synth)
    assert numpy.all(numpy.isclose(astsynth, intsynth)), 'Difference wrt astropy convolution'


def test_beam():
    """
    Test that the convolution doesn't shift the center (at least when
    the kernel is constructed with gauss2d_kernel).

    Note this test fails if you use scipy.fftconvolve because the
    kernels are treated differently.
    """
    n = 50
    synth = beam.gauss2d_kernel(n, 3.)
    _synth = beam.convolve_fft(synth, synth)
    assert numpy.argmax(synth) == numpy.argmax(_synth), \
            'Beam kernel shifted the center for an even image size.'

    n = 51
    synth = beam.gauss2d_kernel(n, 3.)
    _synth = beam.convolve_fft(synth, synth)
    assert numpy.argmax(synth) == numpy.argmax(_synth), \
            'Beam kernel shifted the center for an odd image size.'

@requires_pyfftw
def test_fft():
    synth = beam.gauss2d_kernel(73, 3.)
    synth_fft = numpy.fft.fftn(numpy.fft.ifftshift(synth))
    _convolve_fft = beam.ConvolveFFTW(synth.shape)

    # Compare numpy with direct vs. FFT kernel input
    synth2 = beam.convolve_fft(synth, synth)
    _synth2 = beam.convolve_fft(synth, synth_fft, kernel_fft=True)
    assert numpy.allclose(synth2, _synth2), 'Difference if FFT is passed for numpy'

    # Compare numpy and FFTW with direct input
    _synth2 = _convolve_fft(synth, synth)
    assert numpy.allclose(synth2, _synth2), 'Difference between numpy and FFTW'

    # Compare FFTW with direct vs. FFT kernel input
    synth2 = _convolve_fft(synth, synth_fft, kernel_fft=True)
    assert numpy.allclose(synth2, _synth2), 'Difference if FFT is passed for FFTW'

    # Compare numpy and FFTW with direct input and FFT output
    synth2 = beam.convolve_fft(synth, synth, return_fft=True)
    _synth2 = _convolve_fft(synth, synth, return_fft=True)
    assert numpy.allclose(synth2, _synth2), 'Difference between numpy and FFTW'


@requires_pyfftw
def test_smear():

    n = 51
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(x, y)

    r, theta = projected_polar(x, y, *numpy.radians([45., 30.]))

    rc = oned.HyperbolicTangent(par=numpy.array([100., 1.]))
    sig = oned.Exponential(par=numpy.array([100., 20.]))
    sb = oned.Sersic1D(par=numpy.array([1., 10., 1.]))

    sb_field = sb.sample(r)
    vel_field = rc.sample(r)*numpy.cos(theta)
    sig_field = sig.sample(r)

    cnvlv = beam.ConvolveFFTW(x.shape)
    synth = beam.gauss2d_kernel(n, 3.)
    synth_fft = cnvlv.fft(synth, shift=True)

    vel_smear = beam.smear(vel_field, synth)[1]
    _vel_smear = beam.smear(vel_field, synth, cnvfftw=cnvlv)[1]
    assert numpy.allclose(vel_smear, _vel_smear), 'Velocity-field-only convolution difference.'

    vel_smear = beam.smear(vel_field, synth)[1]
    _vel_smear = beam.smear(vel_field, synth_fft, beam_fft=True)[1]
    assert numpy.allclose(vel_smear, _vel_smear), \
            'Velocity-field difference w/ vs. w/o precomputing the beam FFT for numpy.'

    vel_smear = beam.smear(vel_field, synth, cnvfftw=cnvlv)[1]
    _vel_smear = beam.smear(vel_field, synth_fft, beam_fft=True, cnvfftw=cnvlv)[1]
    assert numpy.allclose(vel_smear, _vel_smear), \
            'Velocity-field difference w/ vs. w/o precomputing the beam FFT for ConvolveFFTW.'

    sb_smear, vel_smear, _ = beam.smear(vel_field, synth, sb=sb_field)
    _sb_smear, _vel_smear, _ = beam.smear(vel_field, synth, sb=sb_field, cnvfftw=cnvlv)
    assert numpy.allclose(sb_smear, _sb_smear), 'SB+Vel convolution difference in SB.'
    assert numpy.allclose(vel_smear, _vel_smear), 'SB+Vel convolution difference in Vel.'

    sb_smear, vel_smear, sig_smear = beam.smear(vel_field, synth, sb=sb_field, sig=sig_field)
    _sb_smear, _vel_smear, _sig_smear = beam.smear(vel_field, synth, sb=sb_field, sig=sig_field,
                                                   cnvfftw=cnvlv)
    assert numpy.allclose(sb_smear, _sb_smear), 'SB+Vel+Sig convolution difference in SB.'
    assert numpy.allclose(vel_smear, _vel_smear), 'SB+Vel+Sig convolution difference in vel.'
    assert numpy.allclose(sig_smear, _sig_smear), 'SB+Vel+Sig convolution difference in sig.'


@requires_pyfftw
def test_deconvolve():
    """
    Test deconvolution
    """
    # Gaussian kernel
    synth = beam.gauss2d_kernel(73, 3.)
    # Convolution object
    cnvfftw = beam.ConvolveFFTW(synth.shape)
    # Get the convolved image
    c = cnvfftw(synth, synth)
    # Deconvolve
    d, m, dc = beam.deconvolve(c, synth, 20, cnvfftw=cnvfftw, return_model=True)
    assert numpy.sum((c-dc)**2) < 1e-9, 'Deconvolution performance worsened.'


@requires_remote
def test_deconvolve_gas():
    # Example gas data
    data_root = remote_data_file()
    kin = manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                 maps_path=data_root)

    # Convolution object
    synth = kin.beam.astype(float)
    cnvfftw = beam.ConvolveFFTW(synth.shape)

#    # Deconvolve
#    c = cnvfftw(synth, synth)
#    d = beam.deconvolve(c, synth, 30, cnvfftw=cnvfftw)
#    # Convolve the deconvolved image with the kernel to compare with original image
#    test_c = cnvfftw(d, synth)

    # Deconvolve
    mask = kin.grid_sb == 0.
    d, d_bpm = beam.deconvolve(kin.grid_sb, synth, 80, mask=mask, cnvfftw=cnvfftw)

    # Convolve the deconvolved image with the kernel to compare with original image
    d_gpm = numpy.logical_not(d_bpm).astype(float)
    cnv_gpm = cnvfftw(d_gpm, synth)
    test_sb = cnvfftw(d * d_gpm, synth) / (cnv_gpm + (cnv_gpm == 0.)) * d_gpm

    # Test total flux recovery
    obs_flux = numpy.sum(kin.grid_sb * d_gpm)
    model_flux = numpy.sum(test_sb)
    assert numpy.absolute(model_flux / obs_flux - 1) < 1e-4, 'Change in total flux too large'

    # Test residual
    model_rms = numpy.sqrt(numpy.mean(numpy.square((kin.grid_sb - test_sb)*d_gpm)))
    assert model_rms < 0.1, 'RMS difference in surface-brightness too large'


@requires_remote
def test_deconvolve_star():

    # Example stellar data
    data_root = remote_data_file()
    kin = manga.MaNGAStellarKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                     maps_path=data_root)

    # Convolution object
    synth = kin.beam.astype(float)
    cnvfftw = beam.ConvolveFFTW(synth.shape)

    # Deconvolve
    mask = kin.grid_sb == 0.
    d, d_bpm = beam.deconvolve(kin.grid_sb, synth, 80, mask=mask, cnvfftw=cnvfftw)

    # Convolve the deconvolved image with the kernel to compare with original image
    d_gpm = numpy.logical_not(d_bpm).astype(float)
    cnv_gpm = cnvfftw(d_gpm, synth)
    test_sb = cnvfftw(d * d_gpm, synth) / (cnv_gpm + (cnv_gpm == 0.)) * d_gpm

    # Test total flux recovery
    obs_flux = numpy.sum(kin.grid_sb * d_gpm)
    model_flux = numpy.sum(test_sb)
    assert numpy.absolute(model_flux / obs_flux - 1) < 1e-3, 'Change in total flux too large'

    # Test residual
    model_rms = numpy.sqrt(numpy.mean(numpy.square((kin.grid_sb - test_sb)*d_gpm)))
    assert model_rms < 0.1, 'RMS difference in surface-brightness too large'


# TODO: Come back to this
#@requires_remote
#def test_deconvolve_star_2():
#
#    # Example stellar data
#    #data_root = remote_data_file()
#    kin = manga.MaNGAStellarKinematics.from_plateifu(7815, 12702) #, cube_path=data_root, maps_path=data_root)
#
#    from matplotlib import pyplot
#
##    obs_sb_i = numpy.ma.MaskedArray(kin.grid_sb, mask=numpy.logical_not(kin.grid_sb>0))
##    obs_sb_i = numpy.ma.MaskedArray(numpy.absolute(kin.grid_sb), mask=kin.grid_sb==0.)
#    obs_sb_i = numpy.ma.MaskedArray(kin.grid_sb, mask=kin.grid_sb==0.)
#
##    offset = numpy.ma.amin(obs_sb_i)
##    obs_sb_i += offset
#    obs_beam_i = kin.beam.astype(float)
#    model_sb_dcnv_i, model_sb_mask_i, model_sb_i \
#                = beam.deconvolve(numpy.ma.absolute(obs_sb_i), obs_beam_i, 80, return_model=True)
#    model_sb_dcnv_i = numpy.ma.MaskedArray(model_sb_dcnv_i, mask=model_sb_mask_i)
#    model_sb_i = numpy.ma.MaskedArray(model_sb_i, mask=model_sb_mask_i)
#
#
#    embed()
#    exit()
#
#    # Convolution object
#    synth = kin.beam.astype(float)
#    cnvfftw = beam.ConvolveFFTW(synth.shape)
#
#    # Deconvolve
#    mask = kin.grid_sb == 0.
#    d, d_bpm = beam.deconvolve(kin.grid_sb, synth, 80, mask=mask, cnvfftw=cnvfftw)
#
#    # Convolve the deconvolved image with the kernel to compare with original image
#    d_gpm = numpy.logical_not(d_bpm).astype(float)
#    cnv_gpm = cnvfftw(d_gpm, synth)
#    test_sb = cnvfftw(d * d_gpm, synth) / (cnv_gpm + (cnv_gpm == 0.)) * d_gpm
#
#    # Test total flux recovery
#    obs_flux = numpy.sum(kin.grid_sb * d_gpm)
#    model_flux = numpy.sum(test_sb)
#
#    embed()
#    exit()
#
#    assert numpy.absolute(model_flux / obs_flux - 1) < 1e-3, 'Change in total flux too large'
#
#    # Test residual
#    model_rms = numpy.sqrt(numpy.mean(numpy.square((kin.grid_sb - test_sb)*d_gpm)))
#    assert model_rms < 0.1, 'RMS difference in surface-brightness too large'
#
#test_deconvolve_star_2()
