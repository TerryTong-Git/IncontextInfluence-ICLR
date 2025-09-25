
python experiments.py --label 'final' --datasets "indmmlualg;indmmlumedgen;indmmlupropsy;indmmluforlog;indmmlumordis;indmmlupubrel;indmmlucomsec;indmmluast;indmmlunut;indmmluhighbio;indbigforfal;indbigshu3" --seeds '11' --selectors "cosine;bertscore;bm25;random;bertscoreinfluencepruning;cosineinfluencepruning;bertscoreinfluencepruningsur;cosineinfluencepruningsur;influence;influenceidentity;robertainfluence" --lms "mistral" --lm-batch-size 8 --batch-size 8 --n-shots '3' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results --no-coverage-results --influence-version "proposed"
python run.py run-exps-parallel --paramsfile "params/params-all.jsonl" --gpus "1"

