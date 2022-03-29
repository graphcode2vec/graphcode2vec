#!/bin/bash
postf=.jar
math_jars="/home/wei/eclipse-workspace/code_embedding/data/math_jars"
output=$1
record=$2
if [ ! -d $output ]
then
    mkdir $output
fi

arr=()
store_process=$record
if [ -f $store_process ]
then
	while IFS= read -r line; do
	  arr+=("$line")
	done < $store_process
fi

for f in $( find $math_jars -type f -name "*.jar")
do
	classfiles="${f%$postf}"
	appname=$(basename $classfiles)
	if [[ " ${arr[*]} " == *"$appname"* ]];
	then
	    echo "YES, your arr contains $appname"
	else
	    echo "Process, your arr does not contain $appname"
	fi 
	mkdir -p $output/$appname
	if [ ! -d $classfiles ]; then
		echo "${classfiles} does not exist."
		exit 1
	fi
    java -jar ./target/extracterGraph-1.0-SNAPSHOT.jar -g -i $classfiles -o $output/$appname 
    if [ $? -eq 0 ]; then
    	arr+=($appname)
    	printf "%s\n" "${arr[@]}" > $store_process
    fi 
done
