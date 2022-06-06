#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from ane_transformers import testing_utils
import collections
import coremltools as ct
import logging
import numpy as np
import unittest
import time

import torch

torch.set_grad_enabled(False)

import transformers
import distilbert as ane_transformers

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

# Testing configuration
SANITY_CHECK_CPUGPU2ANE_SPEEDUP_FACTOR = 10
TEST_MAX_SEQ_LEN = 256
PSNR_THRESHOLD = 60

SEQUENCE_CLASSIFICATION_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
SEQUENCE_CLASSIFICATION_TEST_SET = collections.OrderedDict([
    ("Today was a good day!", "POSITIVE"),
    ("This is not what I expected!", "NEGATIVE"),
])


class TestDistilBertForSequenceClassification(unittest.TestCase):
    """
    Test PyTorch & CoreML forward pass parity, conversion success, ANE speed-up and trivial example accuracy
    of ANE-optimized distilbert models for sequence classification
    """

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
            SEQUENCE_CLASSIFICATION_MODEL)
        try:
            cls.models = {
                # Instantiate the reference model from an exemplar pre-trained
                # sequence classification model hosted on huggingface.co/models
                'ref':
                transformers.AutoModelForSequenceClassification.from_pretrained(
                    SEQUENCE_CLASSIFICATION_MODEL,
                    return_dict=False,
                    torchscript=True,
                ).eval()
            }
        except Exception as e:
            raise RuntimeError("Failed to download reference model from huggingface.co/models!") from e
        logger.info("Downloaded reference model from huggingface.co/models")

        # Initialize an ANE equivalent model and restore the checkpoint
        cls.models[
            'test'] = ane_transformers.DistilBertForSequenceClassification(
                cls.models['ref'].config).eval()
        cls.models['test'].load_state_dict(cls.models['ref'].state_dict())
        logger.info("Initialized and restored test model")

        # Cache tokenized inputs and forward pass results on both the reference and test networks
        cls.inputs = cls.tokenizer(
            list(SEQUENCE_CLASSIFICATION_TEST_SET.keys()),
            return_tensors='pt',
            max_length=TEST_MAX_SEQ_LEN,
            padding='max_length',
        )

        cls.inputs_list = [
            cls.inputs['input_ids'], cls.inputs['attention_mask']
        ]
        cls.outputs = {
            'ref': cls.models['ref'](*cls.inputs_list)[0].softmax(1).numpy(),
            'test': cls.models['test'](*cls.inputs_list)[0].softmax(1).numpy(),
        }

        # Keep numpy copies of the inputs for CoreML testing convenience
        cls.np_inputs = {
            f"input_{name}": tensor.numpy().astype(np.int32)
            for name, tensor in cls.inputs.items()
        }

    @classmethod
    def tearDownClass(cls):
        cls.tokenizer = None
        cls.models = None
        cls.inputs = None
        cls.inputs_list = None
        cls.np_inputs = None
        cls.outputs = None

    def test_torch_parity(self):
        psnr = testing_utils.compute_psnr(self.outputs['ref'],
                                          self.outputs['test'])
        logger.info(
            f"PSNR between original PyTorch module and optimized PyTorch module forward pass: {psnr}"
        )
        assert psnr > PSNR_THRESHOLD

    def test_torch_accuracy(self):
        id2label = self.models['test'].config.id2label
        for (text,
             label), output in zip(SEQUENCE_CLASSIFICATION_TEST_SET.items(),
                                   self.outputs['test']):
            assert label == id2label[output.argmax().item()]

    def test_coreml_conversion_and_speedup(self):
        # Conversion from PyTorch module to Torchscript module
        try:
            module_traced = torch.jit.trace(self.models['test'],
                                            self.inputs_list)
        except Exception as e:
            raise RuntimeError("Torchscript tracing failed!") from e
        logger.info("Torchscript tracing is successful")

        # Conversion from Torchscript module to CoreML Model Package
        # Targeting (primarily) ANE without GPU+CPU fallback
        try:
            ane_mlpackage_obj = ct.convert(
                module_traced,
                convert_to='mlprogram',
                inputs=[
                    ct.TensorType(
                        f"input_{name}",
                        shape=tensor.shape,
                        dtype=np.int32,
                    ) for name, tensor in self.inputs.items()
                ],
                compute_units=ct.ComputeUnit.ALL,
            )
        except Exception as e:
            raise RuntimeError(
                "CoreML conversion targeting ANE failed!") from e
        logger.info("CoreML conversion targeting ANE is successful")

        # Targeting GPU+CPU but not the ANE
        try:
            noane_mlpackage_obj = ct.convert(
                module_traced,
                convert_to='mlprogram',
                inputs=[
                    ct.TensorType(
                        f"input_{name}",
                        shape=tensor.shape,
                        dtype=np.int32,
                    ) for name, tensor in self.inputs.items()
                ],
                compute_units=ct.ComputeUnit.CPU_AND_GPU,
            )
        except Exception as e:
            raise RuntimeError(
                "CoreML conversion targeting GPU+CPU failed!") from e
        logger.info("CoreML conversion targeting GPU+CPU is successful")

        # CoreML inference on test inputs
        coreml_out = list(ane_mlpackage_obj.predict(
            self.np_inputs).values())[0]
        coreml_test_outputs = torch.from_numpy(coreml_out).softmax(1).numpy()

        # Test end-to-end parity for the conversion pipeline
        psnr = testing_utils.compute_psnr(coreml_test_outputs,
                                          self.outputs['ref'])
        logger.info(
            f"PSNR between original PyTorch module and optimized CoreML ANE forward pass: {psnr}"
        )
        assert psnr > PSNR_THRESHOLD

        # Speed-up lower bound test
        ane_latency = testing_utils.rough_timeit(
            lambda: ane_mlpackage_obj.predict(self.np_inputs), n=13)
        noane_latency = testing_utils.rough_timeit(
            lambda: noane_mlpackage_obj.predict(self.np_inputs), n=13)
        speedup_factor = noane_latency / ane_latency
        logger.info(
            f"Speed-up factor from GPU+CPU to ANE is at least {speedup_factor:.2f}"
        )

        if speedup_factor < SANITY_CHECK_CPUGPU2ANE_SPEEDUP_FACTOR:
            logger.warning(
                f"Expected speed-up (Expected >{SANITY_CHECK_CPUGPU2ANE_SPEEDUP_FACTOR:.2f}, observed {speedup_factor:.2f}) was not observed " \
                "between coremltools.ComputeUnit.ALL and coremltools.ComputeUnit.CPU_AND_GPU. The reason might be, among other things that your Mac " \
                "does not have Apple Silicon (e.g. M1) so ANE is unavailable for this test. This model will still work as efficiently as expected on " \
                "on devices with A14 and newer or M1 or newer chips."
            )


if __name__ == "__main__":
    unittest.main()
