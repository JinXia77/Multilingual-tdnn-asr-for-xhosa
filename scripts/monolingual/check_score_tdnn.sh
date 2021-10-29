#!/usr/bin/env bash
use_pitch=true
lang=xhosa
data=data/$lang

dataset_dir=$data/test
nnet3_dir=exp/nnet3/multi_bnf_sp
decode=$nnet3_dir/decode_test_new

feat_suffix=_hires
    if $use_pitch; then
      feat_suffix=${feat_suffix}_pitch
    fi

steps/score_kaldi.sh --cmd run.pl ${dataset_dir}${feat_suffix} exp/$lang/tri5/graph_monolingual_attempt ${decode}

more ${decode}/scoring_kaldi/best_wer > ${decode}/best_result_${language}_${model}

more ${decode}/best_result_${language}_${model}

