import unittest
import numpy as np
import torch
from finrisk.model import FinLLMRisk
from finrisk.losses import objective
from finrisk.conformal import weighted_quantile,base_interval,calibrated_interval
from finrisk.retrieval import DatedRetriever

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data={'structured_dim':8,'text_dim':6,'graph_dim':4,'knowledge_dim':6,'num_classes':5}
        self.cfg={'hidden_dim':32,'short_window':4,'long_stride':3,'num_layers':2,'short_heads':8,'long_heads':4,'dropout':.1}
    def test_forward_and_backward(self):
        b={'structured':torch.randn(3,12,8),'text':torch.randn(3,12,6),
           'graph':torch.randn(3,12,4),'knowledge':torch.randn(3,12,6),
           'rationale':torch.randn(3,6),'labels':torch.tensor([0,1,4]),
           'distress':torch.tensor([.1,.3,.9])}
        model=FinLLMRisk(self.data,self.cfg);o=model(b)
        self.assertEqual(tuple(o['logits'].shape),(3,5))
        self.assertEqual(tuple(o['distress'].shape),(3,))
        loss=objective(o,b,torch.ones(5),{'lambda_reg':.5,'lambda_cot':.1,'lambda_cal':.2})
        loss.backward();self.assertTrue(torch.isfinite(loss))
    def test_conformal(self):
        q=weighted_quantile([.1,.2,.3,.4],[1,2,3,4],alpha=.3,decay=0)
        self.assertGreaterEqual(q,.3)
        l,u=base_interval(torch.tensor([.5]),torch.tensor([.1]))
        a,b=calibrated_interval(l,u,q)
        self.assertTrue((a<=l).all() and (b>=u).all())
    def test_retriever_cutoff(self):
        r=DatedRetriever([[1.,0.],[1.,0.],[0.,1.]],[1,8,3])
        self.assertNotIn(1,[x[0] for x in r.search([1,0],asof=5)])
if __name__=='__main__':unittest.main()
