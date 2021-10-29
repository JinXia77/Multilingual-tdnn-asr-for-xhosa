The dissertation is B174090_MSC_Dissertation.pdf

There is also a folder called "scripts". Below the scripts folder, there are some project scripts and final results under the GMM, monolingual, multilingaul folders. 

For GMM folder,the run-1-main.sh is the main script of GMM section training of this project. The decode.sh script contain the decode section codes about GMM section. The run-1-main.sh should be in the first order, and then run the decode.sh. The check_score.sh is a script which will show the decode results. There are also some figures which shows the final WER results of different GMM models including Xhosa decode results with lang/tri5_ali and mfcc/plp, English decode results with lang and plp feature, Zulu decode result only for tri5 model.

For the monolingual folder, the run_tdnn_monolingual_simplified.sh is the main script of training monolingual system. This script includes both training section and decode section. So the check_score_tdnn.sh is just for check decode score in convenienc.The figure also shows the monolingual WER result of Xhosa.

For multilingual folder, the structure of this folder is very similar to that of monolingual folder but with the multilingual Xhosa-Zulu decode result based on tri3 and tri5. 
