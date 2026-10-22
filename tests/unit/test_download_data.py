"""Original-data selection excludes legacy retargeting and verifies Gait120 bytes."""
import hashlib
import json
import sys
import yaml
import pytest
from musclemimic.cli import download_data


def config(tmp_path, revision='a'*40):
    value={'datasets':{'raw_motion_sources':{'repo_id':'mxhjs29/musclemimic-msk','revision':revision,'families':['amass','gait120'],
        'groups':{'amass':{'local_subdir':'musclemimic-msk','allow_patterns':['KIT/**','CMU/**','BABEL/**','LICENSE.txt']},
                  'gait120':{'local_subdir':'.','allow_patterns':['Gait120_101_to_110/**']}}}}}
    path=tmp_path/'manifest.yaml';path.write_text(yaml.safe_dump(value));return path


def test_selects_only_original_amass_directories(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(download_data,'snapshot_download',lambda **kwargs:calls.append(kwargs))
    monkeypatch.setattr(sys,'argv',['download','--family','amass','--manifest',str(config(tmp_path)),'--destination',str(tmp_path/'datasets')])
    download_data.main()
    assert len(calls)==1 and calls[0]['revision']=='a'*40
    assert calls[0]['local_dir']==str(tmp_path/'datasets/musclemimic-msk')
    assert not any('converted' in p or 'hri' in p for p in calls[0]['allow_patterns'])


def gait(tmp_path,broken=False):
    destination=tmp_path/'data';folder=destination/'Gait120_101_to_110';folder.mkdir(parents=True)
    data=b'original';(folder/'source.trc').write_bytes(data)
    sha='0'*64 if broken else hashlib.sha256(data).hexdigest()
    (folder/'file_manifest.json').write_text(json.dumps({'files':[{'path':'Gait120_101_to_110/source.trc','bytes':len(data),'sha256':sha}]}))
    return destination


def test_gait120_download_is_checked_against_original_hashes(tmp_path,monkeypatch):
    destination=gait(tmp_path);calls=[]
    monkeypatch.setattr(download_data,'snapshot_download',lambda **kwargs:calls.append(kwargs))
    monkeypatch.setattr(sys,'argv',['download','--family','gait120','--manifest',str(config(tmp_path)),'--destination',str(destination)])
    download_data.main()
    assert calls[0]['allow_patterns']==['Gait120_101_to_110/**']


def test_gait120_corruption_is_reported(tmp_path,monkeypatch):
    destination=gait(tmp_path,True)
    monkeypatch.setattr(download_data,'snapshot_download',lambda **kwargs:None)
    monkeypatch.setattr(sys,'argv',['download','--family','gait120','--manifest',str(config(tmp_path)),'--destination',str(destination)])
    with pytest.raises(ValueError,match='checksum mismatch'):download_data.main()


def test_unpinned_revision_is_rejected_before_network_access(tmp_path,monkeypatch):
    monkeypatch.setattr(download_data,'snapshot_download',lambda **kwargs:pytest.fail('network attempted'))
    monkeypatch.setattr(sys,'argv',['download','--manifest',str(config(tmp_path,'main'))])
    with pytest.raises(ValueError,match='fixed'):download_data.main()
