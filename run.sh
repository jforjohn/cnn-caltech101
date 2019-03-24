#!/bin/bash
echo Input: $1

regex='^[1-9]+'
if ! [[ $1 =~ $regex ]] ; then 
  echo "error: Input should be a number > 1" >&2; exit 1
fi

if ! [[ $2 =~ $regex ]] ; then
  echo "error: Input should be a number > 1" >&2; exit 1
fi

if [[ $2 < $1 ]]; then
    echo "error: Input should be a number > $1" >&2; exit 1
fi

for i in `seq $1 $2`;
do
  echo $i
  dir_prefix=augment_model
  # name of directory to save the results
  dir_name="${dir_prefix}_${i}"
  mkdir $dir_name
  echo $dir_name
  echo
  sbatch --output=$dir_name/output.txt --error=$dir_name/error.txt sb_caltech.sh $i $dir_prefix
  # &> model_$i/output.txt
done
