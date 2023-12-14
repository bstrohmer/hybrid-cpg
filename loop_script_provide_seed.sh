#!/bin/bash
# Basic while loop

#seed_array=(7981788)
#Generate random numbers in Python using [np.random.randint(10**7) for i in range(20)]
seed_array=(3006629 5912122 1792795 697622 5887114 2484664 2453096 5666097 2711601 4907026 7937371 4862282 9463088 653905 3576174 1405354 1707361 8496518 9310850 2680292)
counter=1
for seed in "${seed_array[@]}"; do
    python3 create_cpg.py $seed
    #echo $seed
    echo $counter
    ((counter++))
done

echo All done
