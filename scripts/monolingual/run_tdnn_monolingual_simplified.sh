#!/usr/bin/env bash

echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
# set -e

# nj=32
nj=4


remove_egs=false
cmd=run.pl
srand=0
stage=0
train_stage=-10
get_egs_stage=-10
decode_stage=-10
num_jobs_initial=2
num_jobs_final=8
speed_perturb=true
use_pitch=true  # if true, pitch feature used to train multilingual setup
use_pitch_ivector=false
use_ivector=false
alidir=tri5_ali
ivector_extractor=

bnf_dim=           # If non-empty, the bottleneck layer with this dimension is added at two layers before softmax.
dir=exp/nnet3/multi_bnf

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

echo "starting at stage $stage"

feat_suffix=_hires      # The feature suffix describing features used in
                        # multilingual training
                        # _hires -> 40dim MFCC
                        # _hires_pitch -> 40dim MFCC + pitch
                        # _hires_pitch_bnf -> 40dim MFCC +pitch + BNF


for f in data/xhosa/train/{feats.scp,text} exp/xhosa/$alidir/ali.1.gz exp/xhosa/$alidir/tree; do
   [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if [ "$speed_perturb" == "true" ]; then suffix=_sp; fi
dir=${dir}${suffix}

# $feat_suffix=_hires_pitch
if $use_pitch; then feat_suffix=${feat_suffix}_pitch ; fi


######################################################################
#### Part 1: Copied across from local/nnet3/run_common_langs.sh ######
######################################################################

echo "$0: extract high resolution 40dim MFCC + pitch for speed-perturbed data "
echo "and extract alignment."

generate_alignments=true # If true, it regenerates alignments.
pitch_conf=conf/pitch.conf # Configuration used for pitch extraction.

[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1
. ./conf/common_vars.sh || exit 1;


. ./utils/parse_options.sh
train_set=train

if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    for datadir in train; do
      if [ ! -d data/xhosa/${datadir}_sp ]; then
        ./utils/data/perturb_data_dir_speed_3way.sh data/xhosa/${datadir} data/xhosa/${datadir}_sp
          
        # Extract Plp+pitch feature for perturbed data
        featdir=plp_perturbed/xhosa

        if $use_pitch; then
          steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $nj  data/xhosa/${datadir}_sp exp/xhosa/make_plp_pitch/${datadir}_sp $featdir
        else
          steps/make_plp.sh --cmd "$train_cmd" --nj $nj data/xhosa/${datadir}_sp exp/xhosa/make_plp/${datadir}_sp $featdir
        fi

        steps/compute_cmvn_stats.sh data/xhosa/${datadir}_sp exp/xhosa/make_plp_pitch/${datadir}_sp $featdir || exit 1;

#         steps/compute_cmvn_stats.sh data/xhosa/${datadir}_sp exp/xhosa/make_plp_pitch/${datadir}_sp $featdir
        utils/fix_data_dir.sh data/xhosa/${datadir}_sp

      fi
    done
  fi

  train_set=train_sp
  if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh \
      --nj $nj --cmd "$train_cmd" \
      --boost-silence $boost_sil \
      data/xhosa/$train_set data/xhosa/lang exp/xhosa/tri5 exp/xhosa/tri5_ali_sp || exit 1

  fi
fi


if [ $stage -le 3 ]; then

  hires_config="--mfcc-config conf/mfcc_hires.conf"
  mfccdir=mfcc_hires/xhosa
  mfcc_affix=""
  if $use_pitch; then
    hires_config="$hires_config --online-pitch-config $pitch_conf"
    mfccdir=mfcc_hires_pitch/xhosa
    mfcc_affix=_pitch_online
  fi
  

  for dataset in $train_set ; do
    data_dir=data/xhosa/${dataset}${feat_suffix}
    log_dir=exp/xhosa/make${feat_suffix}/$dataset

    utils/copy_data_dir.sh data/xhosa/$dataset ${data_dir} || exit 1;

    # scale the waveforms, this is useful as we don't use CMVN
    utils/data/perturb_data_dir_volume.sh $data_dir || exit 1;
    

    steps/make_mfcc${mfcc_affix}.sh --nj $nj $hires_config \
      --cmd "$train_cmd" ${data_dir} $log_dir $mfccdir;
    steps/compute_cmvn_stats.sh ${data_dir} $log_dir $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh ${data_dir};

  done
fi



###############################################################
#### Part 2: Set-up Directories and Make Neural Net Config ####
###############################################################


data_dir=data/xhosa/train${suffix}${feat_suffix}
egs_dir=$dir/egs
ali_dir=exp/xhosa/${alidir}${suffix}


num_targets=`tree-info ${ali_dir}/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;
ivector_dim=0
feat_dim=`feat-to-dim scp:${data_dir}/feats.scp -`

if [ $stage -le 8 ]; then
  echo "$0: creating neural net config using the xconfig parser";
  if [ -z $bnf_dim ]; then
    bnf_dim=1024
  fi
  mkdir -p $dir/configs
  ivector_node_xconfig=""
  ivector_to_append=""

  cat <<EOF > $dir/configs/network.xconfig
  $ivector_node_xconfig
  input dim=$feat_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 input=Append(input@-2,input@-1,input,input@1,input@2$ivector_to_append) dim=1024
  relu-renorm-layer name=tdnn2 dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,2) dim=1024
  relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn6 input=Append(-7,2) dim=1024
  relu-renorm-layer name=tdnn_bn dim=$bnf_dim
  
  relu-renorm-layer name=prefinal-affine-lang input=tdnn_bn dim=1024
  output-layer name=output dim=$num_targets max-change=1.5
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/
fi



###########################################
#### Part 3: Generate Egs for Training ####
###########################################

if [ $stage -le 9 ]; then
  . $dir/configs/vars || exit 1;
  
  ### Config Section for Prepare Egs
  get_egs_stage=0
  samples_per_iter=400000
  minibatch_size=512
  num_archives=100
  num_jobs=10
  
  . parse_options.sh || exit 1;
  
  left_context=$model_left_context
  right_context=$model_right_context 
  cmd="$decode_cmd"
  cmvn_opts="--norm-means=false --norm-vars=false"

  data_dir=data/xhosa/train${suffix}${feat_suffix}
  egs_dir=$dir/egs
  ali_dir=exp/xhosa/${alidir}${suffix}

  echo "settings egs to $egs_dir"
  if [ ! -d "$egs_dir" ]; then
    echo "$0: Generate egs for xhosa"

    extra_opts=()
    [ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
    extra_opts+=(--left-context $left_context)
    extra_opts+=(--right-context $right_context)

    echo "$0: calling get_egs.sh"
    steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
      --samples-per-iter $samples_per_iter --stage $get_egs_stage \
      --cmd "$cmd" $egs_opts \
      --generate-egs-scp false \
      --nj $nj \
      $data_dir $ali_dir $egs_dir || exit 1;
  fi
fi


echo Finishing Preparing the tdnn here, now train on GPU
echo Run this script on GPU, after copying across relevent directories but start at stage 11
# exit 0;

##########################################
#### PART 4: TRAIN THE NEURAL NETWORK ####
#### DO THIS STAGE ON GPU             ####
##########################################

export CUDA_VISIBLE_DEVICES=0

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi



if [ $stage -le 11 ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --trainer.samples-per-iter=400000 \
    --trainer.max-param-change=2.0 \
    --trainer.srand=$srand \
    --feat-dir $data_dir \
    --egs.dir $egs_dir \
    --use-dense-targets false \
    --targets-scp $ali_dir \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --use-gpu true \
    --dir=$dir  || exit 1;
fi


if [ $stage -le 12 ]; then
  echo "add transition model and adjust"
  nnet3-am-init $ali_dir/final.mdl $dir/final.raw $dir/final.mdl || exit 1;
  echo "$0: compute average posterior and readjust priors"
  steps/nnet3/adjust_priors.sh --cmd "$decode_cmd" \
    --use-gpu false \
    --iter final --use-raw-nnet false\
    $dir $egs_dir || exit 1;
fi



echo Finished Training on GPU, can decode on CPU 
# exit 0;


########################
#### PART 5: DECODE ####
########################

# >>>>>>>>>>>>>>>>>>>
# decoding different languages
if [ $stage -le 13 ]; then
  
  # decode configs
  decode_data_dir=test
  decode_stage=-1
  use_bnf=false

  . conf/common_vars.sh
  . utils/parse_options.sh

  # more specific params:
  iter=final_adj
  nj=8
  lang=xhosa
  nnet3_dir=$dir
  langconf=conf/$lang/lang.conf
  
  . $langconf
 
  # more setup copied from run_decode_lang
  mfcc=mfcc/$lang
  data=data/$lang

  dataset_dir=$data/$decode_data_dir # data/xhosa/test
  dataset_id=$decode_data_dir # test
  dataset=$(basename $dataset_dir) # test

  mfccdir=mfcc_hires/$lang/$dataset
  mfcc_affix=""
  hires_config="--mfcc-config conf/mfcc_hires.conf"

  nnet3_data_dir=${dataset_dir}_hires
  feat_suffix=_hires
  log_dir=exp/$lang/make_hires/$dataset


  if $use_pitch; then
    mfcc_affix="_pitch_online"
    hires_config="$hires_config --online-pitch-config conf/pitch.conf"
    mfccdir=mfcc_hires_pitch/$lang/$dataset
    nnet3_data_dir=${dataset_dir}_hires_pitch #data/xhosa/test_hires_pitch
    feat_suffix="_hires_pitch"
    log_dir=exp/$lang/make_hires_pitch/$dataset
  fi
 
  # Feature Extraction
  if [ ! -f  $dataset_dir/.done ] ; then
    if [ ! -f ${nnet3_data_dir}/.mfcc.done ]; then

      if [ ! -d ${nnet3_data_dir} ]; then
        utils/copy_data_dir.sh $data/$dataset ${nnet3_data_dir}
      fi

      steps/make_mfcc${mfcc_affix}.sh --nj $nj $hires_config \
          --cmd "$train_cmd" ${nnet3_data_dir} $log_dir $mfccdir;
      steps/compute_cmvn_stats.sh ${nnet3_data_dir} $log_dir $mfccdir;
      utils/fix_data_dir.sh ${nnet3_data_dir};
      touch ${nnet3_data_dir}/.mfcc.done

    fi
    touch $dataset_dir/.done
  fi

  # Make Graph
  if [ ! -f exp/$lang/tri5/graph/HCLG.fst ];then
    utils/mkgraph.sh \
      data/$lang/lang exp/$lang/tri5 exp/$lang/tri5/graph_monolingual_attempt |tee exp/$lang/tri5/mkgraph.log
  fi 

  # Decode
  if [ -f $nnet3_dir/final.mdl ]; then
    decode=$nnet3_dir/decode_${dataset_id}

    feat_suffix=_hires
    if $use_pitch; then
      feat_suffix=${feat_suffix}_pitch
    fi

    if [ ! -f $decode/.done ]; then
      mkdir -p $decode
      score_opts="--skip-scoring false"
      [ ! -z $iter ] && iter_opt="--iter $iter"
#       decode_stage=3
      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" $iter_opt \
          --stage $decode_stage \
          --beam $dnn_beam --lattice-beam $dnn_lat_beam \
          $score_opts \
          exp/$lang/tri5/graph_monolingual_attempt ${dataset_dir}${feat_suffix} $decode | tee $decode/decode.log
          
          
#           steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" $iter_opt           --stage $decode_stage           --beam $dnn_beam --lattice-beam $dnn_lat_beam           $score_opts exp/xhosa/tri5/graph_monolingual_attempt data/xhosa/test_hires_pitch exp/nnet3/multi_bnf_sp/decode_test_new | tee exp/nnet3/multi_bnf_sp/decode_test_new/decode.log
          
#       steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" $iter_opt \
#           --stage $decode_stage \
#           $score_opts \
#           exp/$lang/tri5/graph ${dataset_dir}${feat_suffix} $decode | tee $decode/decode.log

      touch $decode/.done
    fi
  fi
fi
