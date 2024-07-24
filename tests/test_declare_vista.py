from unittest import TestCase
from models import vista_model_registry


class TestVISTADeclare(TestCase):
        def test_function_raises_exception(self):
            _ = vista_model_registry['vit_b'](
                image_size=256, encoder_in_channs=27, patch_embed_3d=True,
                vae=True
            )
