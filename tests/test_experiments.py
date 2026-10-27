import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.text import tokenize, split_token_stream, make_sequences
from training.experiments import run_experiment, validate_config
from training.experiment_history import append_history, load_history
from training.evaluation import evaluate_rnn,evaluate_bigram

ROOT=Path(__file__).resolve().parents[1]
TOKENS=tokenize((ROOT/"data/sample_corpus.txt").read_text(encoding="utf-8"))

def config(**changes):
    values={"learning_rate":0.1,"hidden_size":8,"sequence_length":3,"epochs":3,
            "train_fraction":0.70,"validation_fraction":0.15,"seed":12,"clip":5.0}
    values.update(changes); return values

class ExperimentTests(unittest.TestCase):
    def test_split_is_ordered_and_windows_do_not_cross_boundaries(self):
        a,b,c=split_token_stream(list(range(100)),0.7,0.15)
        self.assertEqual((a[-1],b[0]),(69,70)); self.assertEqual((b[-1],c[0]),(84,85))
        self.assertTrue(all(max(x)<70 for x,y in make_sequences(a,3)))
        self.assertTrue(all(min(x)>=70 for x,y in make_sequences(b,3)))
    def test_metrics_and_independent_models(self):
        a=run_experiment(TOKENS,config())
        b=run_experiment(TOKENS,config(seed=13))
        self.assertIsNot(a["model"],b["model"])
        self.assertFalse((a["model"].Wxh==b["model"].Wxh).all())
        self.assertGreater(len(a["metrics"]["train_history"]),0)
        self.assertEqual(len(a["metrics"]["validation_history"]),3)
        for key in ("train_loss","validation_loss","test_loss","duration"):
            self.assertGreaterEqual(a["metrics"][key],0)
        self.assertEqual(a["corpus_hash"],b["corpus_hash"])
        training_end=int(len(TOKENS)*0.70)
        self.assertEqual(a["vocab"].words, ["<unk>"]+sorted(set(TOKENS[:training_end])))
    def test_config_validation_and_history(self):
        with self.assertRaises(ValueError): validate_config(config(train_fraction=0.9,validation_fraction=0.1))
        run=run_experiment(TOKENS,config())
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path=Path(folder)/"history.json"
            rows=append_history(path,run)
            self.assertEqual(len(rows),1)
            self.assertEqual(load_history(path)[0]["corpus_hash"],run["corpus_hash"])
    def test_evaluation_and_bigram_baseline(self):
        run=run_experiment(TOKENS,config(epochs=2))
        model_metrics=evaluate_rnn(run["model"],run["test_examples"],run["unknown_counts"]["test"],run["token_counts"]["test"])
        baseline=evaluate_bigram(run["train_examples"],run["test_examples"],run["vocab_size"])
        self.assertEqual(model_metrics["examples"],run["sequence_counts"]["test"])
        self.assertGreaterEqual(model_metrics["top1_accuracy"],0)
        self.assertTrue(model_metrics["perplexity"]>=1)
        self.assertGreaterEqual(baseline["mean_cross_entropy"],0)
        short_context=run_experiment(TOKENS,config(epochs=1,sequence_length=1))
        self.assertGreater(short_context["sequence_counts"]["train"],run["sequence_counts"]["train"])

if __name__=="__main__": unittest.main()
