"""Returned reference metadata must correspond to actual full-image attachments."""
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from xcortex.tools.rag_encoder import RAGRetrievalTool


def test_reference_resolution_and_missing_image_are_explicit(settings, tmp_path):
    image = np.arange(600 * 400, dtype=np.uint32).reshape(400, 600).astype(np.uint8)
    path = tmp_path / "reference.png"
    Image.fromarray(image).save(path)
    tool = RAGRetrievalTool(settings)
    tool._encoder = MagicMock()
    tool._store = MagicMock()
    tool._store.count.return_value = 2
    tool._store.query.return_value = [
        {"case_id": "missing", "distance": 0.1, "meta": {"image_path": str(tmp_path / "missing.png")}},
        {"case_id": "available", "distance": 0.2, "meta": {"image_path": str(path), "region": "chest"}},
    ]
    output = tool.run(image, {"region": "chest"}, "")
    assert output.payload["status"] == "partial_images"
    assert output.payload["count"] == 1
    assert output.payload["unavailable_images"][0]["case_id"] == "missing"
    assert output.payload["matches"][0]["image_artifact"] == "ref_0.png"
    assert output.images[0][0] == "ref_0.png"
    assert output.images[0][1].size == (600, 400)
    np.testing.assert_array_equal(np.asarray(output.images[0][1])[:, :, 0], image)
