# #!/usr/bin/env bash

# . ./cmd.sh
# language=xhosa
# path2data=data
# path2lang_data=${path2data}
# model_list="mono tri1 tri2 tri3 tri4 tri5"
# # model_list=tri5
# lang_select="tri5_ali" # tri5_ali, lang


# if [ $lang_select == "tri5_ali" ]
#     # generate langp_test folder.
#     echo "generate tri5_ali langp_test folder"
#     cp -R ${path2lang_data}/langp/tri5_ali/ ${path2lang_data}/langp_test_${lang_select}
#     cp ${path2lang_data}/lang/G.fst ${path2lang_data}/langp_test_${lang_select}
#     touch ${path2lang_data}/langp_test_${lang_select}/.done
#     if

# if [ $lang_select == "lang" ]
#     # generate langp_test folder.
#     echo "generate lang langp_test folder"
#     cp -R ${path2lang_data}/lang/ ${path2lang_data}/langp_test_${lang_select}
#     cp ${path2lang_data}/lang/G.fst ${path2lang_data}/langp_test_${lang_select}
#     touch ${path2lang_data}/langp_test_${lang_select}/.done
# if

# for model in ${model_list};do


#     decode=exp/${language}/${model}/decode_${language}_test_${model}
#     # mkgraph and decode
#     if [ ! -f ${decode}/.done ]; then
#       echo ---------------------------------------------------------------------
#       echo "Spawning decoding with SAT models  on" `date`
#       echo ---------------------------------------------------------------------
#       utils/mkgraph.sh \
#         ${path2lang_data}/langp_test_${lang_select} exp/${language}/${model} exp/${language}/${model}/graph | tee exp/${language}/${model}/mkgraph.log

#       mkdir -p $decode

#       steps/decode_fmllr_extra.sh --nj 4 --cmd "$decode_cmd" exp/${language}/${model}/graph ${path2lang_data}/test ${decode} | tee ${decode}/decode.log

#       touch ${decode}/.done
#     fi

#     # check scores
#     steps/score_kaldi.sh --cmd run.pl ${path2lang_data}/test exp/${language}/${model}/graph ${decode}

#     more ${decode}/scoring_kaldi/best_wer > ${decode}/best_result_${language}_${model}

#     more ${decode}/best_result_${language}_${model}

# done

#!/usr/bin/env bash

. ./cmd.sh
language=xhosa
path2data=data
path2lang_data=${path2data}
model_list="mono tri1 tri2 tri3 tri4 tri5"
# model_list=tri5
# lang_select="tri5_ali" # tri5_ali, langp
lang_select="langp"

if [ $lang_select == "tri5_ali" ]; then
    # generate langp_test folder.
    echo "generate tri5_ali langp_test folder"
    cp -R ${path2lang_data}/langp/tri5_ali/ ${path2lang_data}/langp_test_${lang_select}
    cp ${path2lang_data}/lang/G.fst ${path2lang_data}/langp_test_${lang_select}
    touch ${path2lang_data}/langp_test_${lang_select}/.done
fi

if [ $lang_select == "langp" ]; then
    # generate langp_test folder.
    echo "generate lang langp_test folder"
    cp -R ${path2lang_data}/lang/ ${path2lang_data}/langp_test_${lang_select}
    cp ${path2lang_data}/lang/G.fst ${path2lang_data}/langp_test_${lang_select}
    touch ${path2lang_data}/langp_test_${lang_select}/.done
fi

for model in ${model_list};do

    decode=exp/${language}/${model}/decode_${language}_${lang_select}_${model}
    # mkgraph and decode
    if [ ! -f ${decode}/.done ]; then
      echo ---------------------------------------------------------------------
      echo "Spawning decoding with SAT models  on" `date`
      echo ---------------------------------------------------------------------
      utils/mkgraph.sh \
        ${path2lang_data}/langp_test_${lang_select} exp/${language}/${model} exp/${language}/${model}/graph | tee exp/${language}/${model}/mkgraph.log

      mkdir -p $decode

      steps/decode_fmllr_extra.sh --nj 4 --cmd "$decode_cmd" exp/${language}/${model}/graph ${path2lang_data}/test ${decode} | tee ${decode}/decode.log

      touch ${decode}/.done
    fi

    # check scores
    steps/score_kaldi.sh --cmd run.pl ${path2lang_data}/test exp/${language}/${model}/graph ${decode}

    more ${decode}/scoring_kaldi/best_wer > ${decode}/best_result_${language}_${model}

    more ${decode}/best_result_${language}_${model}

done
