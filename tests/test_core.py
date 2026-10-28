import pandas as pd
import torch

from src.models import ForensicBottleneckAdapter, CFD_FSA, GradientReversal
from src.protonet import prototypical_logits
from src.data import add_source_groups


def test_protonet_two_class_prediction():
    support = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits, _ = prototypical_logits(support, labels, query)
    assert logits.argmax(dim=1).tolist() == [0, 1]


def test_fba_shape():
    model = ForensicBottleneckAdapter(512, 128)
    x = torch.randn(4, 512)
    z = model.forensic_embedding(x)
    assert z.shape == (4, 128)


def test_cfd_fsa_shape():
    model = CFD_FSA(512, 128, 512)
    x = torch.randn(4, 512)
    z, content = model(x)
    assert z.shape == (4, 128)
    assert content.shape == (4, 512)


def test_source_grouping_links_ffpp_pairs():
    df = pd.DataFrame({
        "path": ["real/000.mp4", "fake/000_001.mp4", "fake/001_000.mp4", "real/002.mp4"],
        "source_video_key": ["real/000.mp4", "fake/000_001.mp4", "fake/001_000.mp4", "real/002.mp4"],
    })
    out = add_source_groups(df)
    assert out.loc[0, "source_group_key"] == out.loc[1, "source_group_key"]
    assert out.loc[0, "source_group_key"] == out.loc[2, "source_group_key"]
    assert out.loc[3, "source_group_key"] != out.loc[0, "source_group_key"]
