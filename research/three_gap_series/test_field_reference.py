import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from field_reference import validate_reference


class FieldReferenceTests(unittest.TestCase):
    def test_compatible_reuse_does_not_expose_old_rotation_or_truth_evaluation(self):
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=root/'results') as tmp:
            cache=Path(tmp)/'fields.npz';summary=Path(tmp)/'summary.json'
            hashes=dict(checkpoint='cp',latent='z',dataset='wy')
            np.savez(cache,h=np.zeros((3,7,7)),checkpoint_hash='cp')
            args=SimpleNamespace(model='C1',beta=.0005,zdim=4,pca=0,source='formula',estimator='direct',channel='logit',field_reference=str(summary),metric='new_metric')
            protocol=dict(x_mean=[0.,0.,0.],x_sd=[1.,1.,1.])
            record=dict(scope='hessian_only',hashes=hashes,args={k:v for k,v in vars(args).items() if k not in ['field_reference','metric']},
                protocol=protocol,field_cache=str(cache),best={'r':'must not be returned'},evaluation={'matched_mean':999})
            summary.write_text(json.dumps(record),encoding='utf8')
            out,provenance=validate_reference(args,dict(hashes=hashes),protocol,(3,7),None,False)
            self.assertEqual(out,cache)
            self.assertNotIn('best',provenance);self.assertNotIn('evaluation',provenance)
            args.channel='probability'
            with self.assertRaisesRegex(ValueError,'channel'):validate_reference(args,dict(hashes=hashes),protocol,(3,7),None,False)
