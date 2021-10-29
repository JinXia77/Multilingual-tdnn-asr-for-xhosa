# #!/usr/bin/env bash
# . ./cmd.sh

# model=tri5
# language=xhosa
# path2data=data
# path2lang_data=${path2data}

# # dataset_id=${language}_test_${model}
# # decode=exp/${language}/${model}/decode_${dataset_id}
# decode=exp/${language}/${model}/decode_${language}_test_${model}


# steps/score_kaldi.sh --cmd run.pl ${path2lang_data}/test exp/${language}/${model}/graph ${decode}

# more ${decode}/scoring_kaldi/best_wer

#!/usr/bin/env bash

. ./cmd.sh
language=xhosa
path2data=data
path2lang_data=${path2data}
model_list="mono tri1 tri2 tri3 tri4 tri5"
# model_list=tri5
# lang_select="tri5_ali" # tri5_ali, langp
lang_select="langp"

for model in ${model_list};do

    decode=exp/${language}/${model}/decode_${language}_${lang_select}_${model}

    more ${decode}/best_result_${language}_${model}

done