#!/usr/bin/env python

def file_gen(time, normalized_rate_coded_agonist, normalized_rate_coded_antagonist, directory, arm, leg, excitation_gain=1):
    # Muscles name must match the OpenSim muscles' names.
    if leg == 1:
        muscles = ['bifemlh_r', 'rect_fem_r']
        file_name = 'leg_excitation.sto'
    if arm == 1:
        muscles = ['TRIlong', 'BIClong']
        file_name = 'arm_excitation.sto'

    # Extract element from the list once the transient is over.
    time = time[5000:]
    normalized_rate_coded_agonist = normalized_rate_coded_agonist[5000:]
    normalized_rate_coded_agonist = [i*excitation_gain for i in normalized_rate_coded_agonist]
    normalized_rate_coded_antagonist = normalized_rate_coded_antagonist[5000:]
    normalized_rate_coded_antagonist = [i*excitation_gain for i in normalized_rate_coded_antagonist]
    
    # .sto file headers.
    headers = ['controls', 'version=', 'nRows=', 'nColumns=', 'inDegrees=', 'endheader']
    ver = str(1)  # Version must be set to 1 for OpenSim > 2.3
    rows = str(len(time))
    cols = str(len(muscles) + 1)
    indeg = 'no'

    with open(directory + '/' + file_name, 'w') as f:
        # Write the header of the .sto file
        for header in headers:
            f.write(header)

            if header == 'version=':
                f.write(ver)
            if header == 'nRows=':
                f.write(rows)
            if header == 'nColumns=':
                f.write(cols)
            if header == 'inDegrees=':
                f.write(indeg)

            f.write('\n')

        # The first column of the data frame is the time, then muscles spaced by tabs
        f.write('time')
        for muscle in muscles:
            f.write('\t' + muscle)

        # Rows begins with seven spaces, excitation values have six spaces
        s7 = '       '
        s6 = '      '
        for t, rc_ag, rc_ant in zip(time, normalized_rate_coded_agonist, normalized_rate_coded_antagonist):
            # Return time in s.
            current_time = t/10000

            # Write the data on the file in the correct format.
            f.write(f'\n{s6}{current_time}{s7}{rc_ant}{s7}{rc_ag}')

        print("The '.sto' file has been generated.")
