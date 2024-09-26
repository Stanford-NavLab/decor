import numpy as np
import scipy.linalg as spla
import math


##################################################     CHECK FUNCTIONS     ##################################################


def chk_allbin01(seq, eps=1e-8):
    # checks all elements of seq are 0 or 1
    return np.sum(np.abs(seq * (1 - seq))) < eps


def chk_allpm1(seq, eps=1e-8):
    # checks all elements of seq are -1 or +1
    return np.sum(np.abs(np.abs(seq) - 1)) < eps


def chk_samesize(nparr1, nparr2):
    # checks two numpy arrays are the same size
    return nparr1.shape == nparr2.shape


def convert_bin01_to_pm1(seq):
    # converts elements in seq from 0/1 to +1/-1
    assert chk_allbin01(seq)
    return -2 * seq + 1


def convert_pm1_to_bin01(seq):
    # converts elements in seq from +1/-1 to 0/1
    assert chk_allpm1(seq)
    return np.abs(-0.5 * (seq - 1))


def _get_mls_pairs(len_gold):
    # Specify preferred pairs (of maximum-length sequence generators) for generating Gold codes
    #
    # For most preferred pairs, following: https://www.gaussianwaves.com/2015/06/gold-code-generator/
    # For length-31: got pair from GPS textbook
    # For length-1023: got pair from GPS ICD
    # For length-8191: got pair from Gao et al., Aug 2009,
    #                  IEEE Jour. of Selected Topics in Signal Processing (Compass-M1)
    #
    # Note that the taps are assuming register is going from right to left (idx 0 is first to exit)
    # Correspondingly, for length-31 tap1, [1, 0, 1, 0, 0] which is tap [5,3],
    #     will have taps at array idx 1 (python 0) and idx 3 (python 2), and the result will be input to idx 5 (python 5)
    #     (it is weird, yes, the array indices are reversed from the tap indices... this is using the index notation
    #      from the website link, which I find equally confusing, but have confirmed multiple times)

    if len_gold == 31:
        tap1 = [1, 0, 1, 0, 0]
        tap2 = [1, 0, 1, 1, 1]
    elif len_gold == 63:
        tap1 = [1, 0, 0, 0, 0, 1]  # this is tap: [6,1]
        tap2 = [1, 1, 0, 0, 1, 1]  # this is tap: [6,5,2,1]
    elif len_gold == 127:
        tap1 = [1, 0, 0, 0, 1, 1, 1]
        tap2 = [1, 0, 0, 0, 1, 0, 0]
    elif len_gold == 511:
        tap1 = [1, 0, 0, 1, 0, 1, 1, 0, 0]
        tap2 = [1, 0, 0, 0, 0, 1, 0, 0, 0]
    elif len_gold == 1023:
        tap1 = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        tap2 = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    elif len_gold == 2047:
        tap1 = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
        tap2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif len_gold == 8191:
        tap1 = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
        tap2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
    else:
        raise ValueError(
            f"Cannot create Gold codes for specified length: len_gold = {len_gold} "
            "\n(either inconsistent with Gold code definition, or longer than 8191)"
        )

    # return as np arrays
    tap1 = np.array(tap1, dtype=int)
    tap2 = np.array(tap2, dtype=int)

    return tap1, tap2


def _get_lfsr_sequence(tap_array, reg):
    # Returns LFSR sequence as a binary (0,1) sequence
    # tap_array -- taps of the register
    # reg -- initial state of register (can be anything but all 0s)

    # Note: assuming register (reg) will go from right to left
    # (i.e., index 0 of reg will be the first output of the register)
    # so reg (and tap array) should be initialized accordingly

    # Check that inputs are 1-D arrays
    assert tap_array.ndim == 1, "tap_array must be 1-D array, but is of shape " + str(
        tap_array.shape
    )
    assert reg.ndim == 1, "reg must be 1-D array, but is of shape " + str(reg.shape)

    # Check length of taps and initial register
    assert chk_samesize(tap_array, reg), (
        "tap_array and reg not of the same length. tap_array is length-"
        + str(len(tap_array))
        + ", and reg is length-"
        + str(len(reg))
    )

    # Initialize full maximum length sequence
    reg_len = len(reg)
    seq_len = 2 ** len(tap_array) - 1
    full_seq = np.hstack((reg, np.zeros(seq_len - reg_len, dtype=int)))

    # Create the rest of the LFSR sequence
    for idx in range(reg_len, seq_len):
        # Get output of LFSR, put it in the sequence
        nxt_bit = np.sum(reg * tap_array) % 2
        full_seq[idx] = nxt_bit

        # Update the current register
        reg[0 : (reg_len - 1)] = reg[1:(reg_len)]
        reg[reg_len - 1] = nxt_bit

    return full_seq


def _get_gold_code_family_from_taps(tap1, tap2):
    # Returns Gold code family as a codebook of binary (0,1) sequences
    # Note: family is not in any particular order, starting with all relative delays, then adding ML sequences

    # Check that inputs are 1-D arrays
    assert tap1.ndim == 1, "tap1 must be a 1-D arrays, but is of shape " + str(
        tap1.shape
    )
    assert tap2.ndim == 1, "tap2 must be a 1-D arrays, but is of shape " + str(
        tap2.shape
    )

    # Check length of taps are the same
    assert chk_samesize(tap1, tap2), (
        "tap1 and tap2 not of the same length. tap1 is length-"
        + str(len(tap1))
        + ", and tap2 is length-"
        + str(len(tap2))
    )

    # Get sequence length and Gold family size
    nbits = 2 ** len(tap1) - 1
    gold_fam_size = nbits + 2

    # Get maximum length sequences
    mls1 = _get_lfsr_sequence(tap1, np.ones(len(tap1), dtype=int))
    mls2 = _get_lfsr_sequence(tap2, np.ones(len(tap2), dtype=int))
    mls1_pm1 = convert_bin01_to_pm1(mls1)
    mls2_pm1 = convert_bin01_to_pm1(mls2)

    # Initialize codebook
    gold_codebook = np.zeros((gold_fam_size, nbits), dtype=int)

    # Get relative delay between maximum-length codes
    all_delayed_mls1 = spla.circulant(mls1_pm1).T
    gold_codebook[0:nbits, :] = convert_pm1_to_bin01(all_delayed_mls1 * mls2_pm1)

    # Add in original LFSR codes (also as (0,1) binary sequences)
    gold_codebook[nbits, :] = mls1
    gold_codebook[nbits + 1, :] = mls2

    return gold_codebook


def gold_codes(len_gold):
    # Get MLS pairs
    tap1, tap2 = _get_mls_pairs(len_gold)

    # Get and return corresponding Gold codes
    gold_codebook = _get_gold_code_family_from_taps(tap1, tap2)
    return convert_bin01_to_pm1(gold_codebook)


############################################################################################################
##############################################              ################################################
##############################################  Weil Codes  ################################################
##############################################              ################################################
############################################################################################################
def prime(num):
    # 2 is the smallest prime
    if num < 2:
        return False

    # Check if divisible by 2
    if num > 2 and num % 2 == 0:
        return False

    # Check if divisible by other odd numbers
    for i in range(3, math.ceil(math.sqrt(num)), 2):
        if num % i == 0:
            return False
    return True


def get_leg_seq(L):
    # Note that L must be a prime number
    assert prime(L), "L must be a prime number"

    # get the Legendre set (e.g., for length-11: [1, 3, 4, 5, 9])
    legendre_set = np.array([((i * i) % L) for i in range(1, L)])
    legendre_set = np.unique(legendre_set)

    # First element of Legendre sequence is defined as a (binary) 1 -- i.e., -1 for (+/- 1 regime)
    # ^^ this is index 0
    # For indices 1 and higher, values should be +1 for indices directly corresponding to Legendre set
    # (indices 1, 3, 4, 5, and 9 should be +1 if these values are in Legendre set)
    leg_seq = -1 * np.ones(L, dtype=int)
    leg_seq[legendre_set] = 1

    return leg_seq


def weil_codes(L):
    # Returns Weil codes as a set of +/- 1 sequences

    # L must be a prime number for Weil codes
    assert prime(L), "L must be a prime number"

    # get Legendre sequence & number codes
    leg_seq = get_leg_seq(L)
    nWeil = (L - 1) // 2

    # create all relevant shifts of the Legendre sequence
    weil_shifts = np.arange(nWeil) + 1  # shifts should be 1 to nWeil (not 0 to nWeil-1)
    weil_shifted = spla.circulant(leg_seq)[:, weil_shifts].T

    # element-wise multiplication of Legendre sequence with shifted versions
    weil_codebook = leg_seq * weil_shifted

    return weil_codebook
