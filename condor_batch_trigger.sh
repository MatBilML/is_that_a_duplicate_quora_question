#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters; Please provide to_trigger file and cpu/gpu as the parameters;"
    exit 1
fi

IFS=$'\n'

# Trigger file format: <run_number>,<python command> one per line
to_trigger_file=$1
pu=$2

for i in `cat $to_trigger_file`; do
        run=`echo $i | cut -d ',' -f1`
        cmd=`echo $i | cut -d ',' -f2`

        echo Run: $run
        echo Command: $cmd

        trigger_file='trigger'$run'.sh'
        echo 'Trigger file name: ' $trigger_file

        echo '#!/bin/bash' > $trigger_file
        echo '' >> $trigger_file
        echo $cmd >> $trigger_file

       ./process_condor.sh run"$run" $pu $trigger_file
done
