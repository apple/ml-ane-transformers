#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import logging
import unittest
import collections
import numpy as np
import coremltools as ct
import ANEGPTCore.testing_utils as test_utils
import ANEGPTCore.ANE_GPT_Model as ANE_GPT




# =================================
torch.set_grad_enabled(False)




logger = logging.getLogger(__name__)
logger.setLevel('INFO')

# Testing configuration
PSNR_THRESHOLD = 20
SANITY_CHECK_CPUGPU2ANE_SPEEDUP_FACTOR = 10

TEST_BATCH_SIZE = 2
TEST_SRC_SEQ_LEN = 128
TEST_TGT_SEQ_LEN = 256
TEST_EMBED_DIM = 512

TEST_INPUTS = collections.OrderedDict({

    'decoder_input':
    torch.rand(TEST_BATCH_SIZE, TEST_EMBED_DIM, 1, TEST_TGT_SEQ_LEN),

    'decoder_pos_embed':
    torch.rand(TEST_BATCH_SIZE, TEST_EMBED_DIM, 1, TEST_TGT_SEQ_LEN),

    'decoder_k_mask':
    torch.zeros(TEST_BATCH_SIZE, TEST_TGT_SEQ_LEN, 1, 1),

    'decoder_qk_mask':
    torch.zeros(TEST_BATCH_SIZE, TEST_TGT_SEQ_LEN, 1, TEST_TGT_SEQ_LEN),
})

# =======================================================
class TestGPTReferenceImplementation(unittest.TestCase):
    """
    Test conversion success and ANE speed-up of the reference Transformer implementation
    """

    @classmethod
    def setUpClass(cls):
        cls.model = ANE_GPT.AppleNeuralEngineGPT(
            embed_dim=TEST_EMBED_DIM)
        cls.inputs = TEST_INPUTS
        cls.ref_outputs = cls.model(**cls.inputs)

    @classmethod
    def tearDownClass(cls):
        cls.model = None
        cls.inputs = None

    def test_coreml_conversion_and_speedup(self):
        # Conversion from PyTorch module to Torchscript module
        try:
            module_traced = torch.jit.trace(self.model,
                                            list(self.inputs.values()))
        except Exception as e:
            raise RuntimeError("Torchscript tracing failed!") from e

        logger.info("Torchscript tracing is successful")



        # Conversion from Torchscript module to CoreML Model Package
        # Targeting (primarily) ANE without GPU+CPU fallback
        try:
            ANE_Model_CoreML_Model = ct.convert(
                module_traced,
                convert_to='mlprogram',
                inputs=[
                    ct.TensorType(
                        name,
                        shape=tensor.shape,
                        dtype=np.float32,
                    ) for name, tensor in self.inputs.items()
                ],
                compute_units=ct.ComputeUnit.ALL,
            )
            ANE_Model_CoreML_Model.save("ANECoreML.mlpackage")

        except Exception as e:
            raise RuntimeError(
                "CoreML conversion targeting ANE failed!") from e

        logger.info("CoreML conversion targeting ANE is successful")


        # CoreML inference on test inputs
        coreml_out = list(
            ANE_Model_CoreML_Model.predict(
                {k: v.numpy()
                 for k, v in self.inputs.items()}).values())[0]
        coreml_test_outputs = torch.from_numpy(coreml_out).numpy()

        # Test end-to-end parity for the conversion pipeline
        psnr = test_utils.compute_psnr(coreml_test_outputs,
                                                             self.ref_outputs[0].numpy())
        logger.info(
            f"PSNR between original PyTorch module and optimized CoreML ANE forward pass: {psnr:.2f}"
        )
        assert psnr > PSNR_THRESHOLD

        # Speed-up lower bound test
        ane_latency = test_utils.rough_timeit(

            lambda: ANE_Model_CoreML_Model.predict(
                {k: v.numpy()
                 for k, v in self.inputs.items()}),
            n=13)

        # =========================
        noane_latency = test_utils.rough_timeit(
            lambda: ANE_Model_CoreML_Model.predict(
                {k: v.numpy()
                 for k, v in self.inputs.items()}),
            n=13)


        speedup_factor = noane_latency / ane_latency
        logger.info(f"Speed-up factor from GPU+CPU to ANE is at least {speedup_factor:.2f}")


        if speedup_factor < SANITY_CHECK_CPUGPU2ANE_SPEEDUP_FACTOR:
            logger.error(
                f"Expected speed-up (Expected >{SANITY_CHECK_CPUGPU2ANE_SPEEDUP_FACTOR:.2f}, observed {speedup_factor:.2f}) was not observed " \
                "between coremltools.ComputeUnit.ALL and coremltools.ComputeUnit.CPU_AND_GPU. The reason might be, among other things that your Mac " \
                "does not have Apple Silicon (e.g. M1) so ANE is unavailable for this test. This model will still work as efficiently as expected on " \
                "on devices with A14 and newer or M1 or newer chips."
            )


if __name__ == "__main__":
    unittest.main()
