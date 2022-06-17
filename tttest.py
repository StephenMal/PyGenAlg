import numpy as np

def float_search(chr, arrs, gene_len):
    # Return nothing if arrs[0] is longer
    if len(arrs[0]) > len(chr):
        return

    # Sum the beginning
    cursum = 0
    for i in range(len(arrs[0])):
        cursum = cursum + chr[i]

    # Stores all the sums of the subarrs
    sums = np.zeros(len(arrs))
    # Iterate through the subarrs
    for indx in range(len(arrs)):
        # Sum it
        sums[indx] = np.sum(arrs[indx])
        # If sum matches the beggining sum, check for matches
        if sums[indx] == cursum:
            # Assume match is true
            match = True
            # Iterate through the subarr
            for a_indx in range(len(arrs[indx])):
                # Compare to chromosome, break if not true
                if arrs[indx][a_indx] != chr[a_indx]:
                    match = False
                    break
            # If match, yield it.
            if match == True:
                yield (indx, len(arrs[indx]), gene_len+len(arrs[indx]))

    # Iterate trhough
    for c_indx in range(1,len(chr)-gene_len):
        # Update the sum
        cursum = cursum - chr[c_indx-1] + chr[c_indx+len(arrs[0])-1]
        # Check if matching sum
        for s_indx in range(len(sums)):
            # If the sum matches see if its a true match
            if sums[s_indx] == cursum:
                match = True # Assume true
                # Iterate through chr and subarr
                for a_indx in range(len(arrs[s_indx])):
                    # \/ Verify length doesn't exceed bounds
                    if (a_indx+c_indx) >= len(chr):
                        match = False
                        break
                    elif arrs[s_indx][a_indx] != chr[a_indx+c_indx]:
                        # Check for a match /\
                        match = False
                        break
                # If a match, yield
                if match == True:
                    yield (s_indx, \
                           c_indx+len(arrs[s_indx]), \
                           c_indx+gene_len+len(arrs[s_indx]))
    return


chr = np.array([0,1,0,1,1,2,0,0,1,0,1,0,0,0,0])

arr0 = np.array([0,1])
arr1 = np.array([1,0])
arr2 = np.array([0,0])
arr3 = np.array([1,1])

arrs = np.array([arr0, arr1, arr2, arr3])

for x in float_search(chr, arrs, 4):
    print(f'gene {x[0]}, [{x[1]}:{x[2]}] = {chr[x[1]:x[2]]}')
