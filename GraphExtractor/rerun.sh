program=( util-relation_2.12-1.3.0   scala-library-2.13.4 lenses_2.12-0.4.12 jawn-parser_2.12-0.10.4 nearest-neighbour-1.3.10 fastparse_2.12-0.4.2 shaded-scalajson_2.12-1.0.0-M4 log4j-api-2.13.3 scala-java8-compat_2.12-0.9.1 sbinary_2.12-0.5.0  )

for p in "${program[@]}"
do
        #echo $p
        #find . -type f -name "${p}**" -exec cp {} ./rerunjar/ \;
	mkdir rerunjar/${p}
	cd rerunjar/${p}
	jar xvf ../${p}.jar
	cd -
        java  -jar target/extracterGraph-1.0-SNAPSHOT.jar -g -i rerunjar/${p} -o rerun_res/${p}
        echo "java  -jar target/extracterGraph-1.0-SNAPSHOT.jar -g -i rerunjar/${p} -o rerun_res/${p}"	
done
