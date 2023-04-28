#!/bin/bash
# Basic while loop

counter=1
while [ $counter -le 20 ]
do
    python3 create_cpg.py
    echo $counter
    ((counter++))
done

echo All done
