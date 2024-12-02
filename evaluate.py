# CER, MER

import subprocess

# sclite \
# ${score_opts} \
# -r "${_scoredir}/ref.trn" trn \
# -h "${_scoredir}/hyp.trn" trn \
# -i rm -o all stdout > "${_scoredir}/result.txt"

# https://github.com/espnet/espnet/blob/master/egs2/fleurs/asr1/local/score_lang_id.py

def cer(hyp, ref):
    """Calculate Character Error Rate (CER)"""
    
    subprocess.run([
        'sclite',
        '-r', ref,
        'trn',
        '-h', hyp,
        'trn',
        '-i', 'rm',
        '-o', 'all',
        'stdout', '>', 'result.txt'
    ])
    