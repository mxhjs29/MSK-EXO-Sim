from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from musclemimic.algorithms.ppo import checkpoint


@pytest.mark.parametrize('value,expected',[(0,0),(17,17),(None,-1)])
def test_resume_preserves_zero_training_counters(monkeypatch,value,expected):
    """A transferred checkpoint_0 must not become an unknown (-1) resume."""
    class Manager:
        def __init__(self,**kwargs):pass
        def load_checkpoint(self,path):
            return (None,{}),SimpleNamespace(update_number=value,global_timestep=value)
        def close(self):pass
    monkeypatch.setattr(checkpoint,'UnifiedCheckpointManager',Manager)
    agent=SimpleNamespace(config=OmegaConf.create({'experiment':{}}))
    _,resume=checkpoint.load_checkpoint_for_resume('/test/checkpoint_0',agent)
    assert resume['update_number']==expected
    assert resume['global_timestep']==expected
