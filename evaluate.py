# CER, MER

import subprocess
import unicodedata

cmn_range = (0x4E00, 0x9FFF)

# sclite \
# ${score_opts} \
# -r "${_scoredir}/ref.trn" trn \
# -h "${_scoredir}/hyp.trn" trn \
# -i rm -o all stdout > "${_scoredir}/result.txt"

# https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/scripts/utils/evaluate_asr.sh

# https://troylee2008.blogspot.com/2010/03/using-sclite.html
# https://sources.debian.org/data/main/s/sctk/2.4.10-20151007-1312Z%2Bdfsg2-3.1~deb10u1/doc/infmts.htm#trn_fmt_name_0
"""
trn - Definition of a transcript input file

The transcript format is a file of word sequence records separated by newlines. Each record contains a word sequence, follow by the an utterance ID enclosed in parenthesis. See the '-i' option for a list of accepted utterance id types.

example.

she had your dark suit in greasy wash water all year (cmh_sa01)
Transcript alternations, described above, can be used in the word sequence by using this BNF format:

ALTERNATE :== "{" TEXT ALT+ "}"
ALT :== "/" TEXT
TEXT :== 1 or more whitespace separated words | "@" | ALTERNATE
The "@" represents a NULL word in the transcript. For scoring purposes, an error is not counted if the "@" is aligned as an insertion.

example

i've { um / uh / @ } as far as i'm concerned
"""

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
    