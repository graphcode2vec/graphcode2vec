#!/bin/bash
postf=.jar
for f in $( find . -type f -name "*.jar")
do
	name="${f%$postf}"
	mkdir $name
	cd $name
	jar xf ../$f
        cd ..	
done
