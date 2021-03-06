#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Illegal number of parameters; Please provide run_dir and cpu/gpu and optionally trigger.sh file name as the parameters;"
    exit 1
fi

trigger_file='trigger.sh'
if [ "$#" -eq 3 ]; then
    trigger_file=$3
fi

run_dir=$1
pu=$2
mkdir -p $run_dir

condor_file=$run_dir'/condor.script'
cat condor."$pu"_script_head > $condor_file

echo 'arguments = '$trigger_file >> $condor_file
echo 'Log = '$run_dir'/log.log' >> $condor_file
echo 'Output = '$run_dir'/output.log' >> $condor_file
echo 'Error  = '$run_dir'/error.log' >> $condor_file

echo 'Queue 1' >> $condor_file

condor_submit $condor_file
