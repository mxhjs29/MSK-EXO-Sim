import numpy as np
import pytest

from musclemimic.retargeting.gait120_audit import read_mot, read_trc, y_up_to_z_up


def test_trc_preserves_timestamps_converts_units_and_masks_missing(tmp_path):
    p=tmp_path/'sample.trc'
    p.write_text('PathFileType\t4\t(X/Y/Z)\tsample.trc\n'
                 'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n'
                 '100\t100\t2\t2\tmm\n'
                 'Frame#\tTime\tLASI\t\t\tRASI\t\t\n'
                 '\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n'
                 '485\t4.84\t100\t900\t-500\t0\t0\t0\n'
                 '486\t4.85\t110\t905\t-510\t\t900\t-500\n')
    d=read_trc(p)
    np.testing.assert_allclose(d['time'],[4.84,4.85])
    np.testing.assert_array_equal(d['frames'],[485,486])
    np.testing.assert_allclose(d['points_m'][0,0],[.1,.9,-.5])
    assert d['missing'][:,1].all()
    assert np.isnan(d['points_m'][:,1]).all()
    np.testing.assert_allclose(y_up_to_z_up(d['points_m'])[0,0],[.1,.5,.9])
    rotation=y_up_to_z_up(np.eye(3))
    assert np.linalg.det(rotation)==pytest.approx(1.)


def test_mot_retains_degree_metadata_and_rejects_bad_times(tmp_path):
    p=tmp_path/'angles.mot'
    header='Coordinates\nnRows=2\ninDegrees=yes\nendheader\ntime hip_flexion_r\n'
    p.write_text(header+'2.0 30\n2.01 35\n')
    d=read_mot(p)
    assert d['metadata']['inDegrees']=='yes'
    np.testing.assert_array_equal(d['data'][:,1],[30,35])
    p.write_text(header+'2.0 30\n2.0 35\n')
    with pytest.raises(ValueError,match='timestamps'):read_mot(p)


def test_truncated_mot_cannot_be_silently_used_as_shorter_clip(tmp_path):
    p=tmp_path/'cutoff.mot'
    p.write_text('Coordinates\nnRows=3\nnColumns=2\nendheader\ntime angle\n0.0 1\n0.01 2\n')
    with pytest.raises(ValueError,match='row count mismatch'):read_mot(p)
