#
# Code accompanying the paper "diversity in approval-based committee elections under incomplete or
# inaccurate information". Link: https://arxiv.org/pdf/2506.10843
# @author: Martijn Brehm (m.a.brehm@uva.nl)
# @date: 04/08/2025
#

import multiprocessing
import math
import random
import csv
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import os
from filecache import filecache
from bitarray import bitarray
import pickle

DEBUG = False

def cc_score(W,V):
    """
    Compute CC score of (elected) committee W based on voter set V.
    """
    cc = 0
    for voter in V:
        if len([x for x in W if voter[x] == 1]) > 0:
            cc += 1
    return cc

def voters(n, candidates):
    """
    Generates a set V of n voters with preferences as random sized random subsets of 'candidates'.

    Use to test the algorithm when you don't have data.
    """
    # TODO change to bitarray?
    V = [ ]
    for i in range(1,n+1):
        V.append(random.sample(candidates,random.choice(range(1,len(candidates)+1))))
    return V

##########################################################################

############################### GREEDY  ##################################
def greedy_exact(V, k, p=0):
    """
    Runs of the regular greedy max cover algorithm on the given instance.

    Inputs:
    - voter set V, defined as subsets of candidates (approval sets of each voter)
    - committee size k
    - probability that a query is answered incorrectly

    Outputs:
    - the committee
    - list containing the score at each iteration
    """
    # scores will contain CC score at each iteration
    scores = [ ]
    C = list(range(len(V[0])))
    W = [ ]
    V_original = V
    if p > 0:
        V = flip_ballots(V.copy(), p)

    for i in range(k):
        # cc_W_change contains CC(W U c_new \ c) - CC(W)
        cc_W_change = { }
        for c_new in [c for c in C if c not in W]:
            cc_W_change[(c_new)] = cc_score(W + [c_new], V)

        # pick the candidate that corresponds to the largest score
        c_new = max(cc_W_change, key = cc_W_change.get)
        W.append(c_new)
        scores.append(cc_score(W,V_original))
        if DEBUG:
            print(f"greedy exact iteration: {i+1}, score: {scores[-1]}")
    return (W, scores)

def greedy_queries(V,k,t,ell,p=0):
    """
    Incomplete version of the greedy max cover algorithm. Will not access the entire ballot V at
    will, but will instead randomly selected entires of the ballot throughout.

    Inputs:
       - voter set V, defined as subsets of candidates (approval sets of each voter)
       - committee size k
       - query size t
       - amount of voters ell per query
       - probability that a query is answered incorrectly

    Outputs:
    - the committee
    - list containing the score at each iteration
    """
    # scores will contain CC score at each iteration
    scores = []
    m = len(V[0])
    C = list(range(m))
    W = [ ]
    V_original = V
    if p > 0:
        V = flip_ballots(V.copy(), p)

    for i in range(k):

        # per iteration we make (m-k)/(t-k) queries of size t, always containing W, and furthermore t-k other candidates so that all are in one query
        Q = [ ]
        yet_unelected = [x for x in C if x not in W]
        random.shuffle(yet_unelected)
        for j in range(1, math.ceil((m-k)/(t-k))+1):
            Q.append(W + yet_unelected[(t-k)*(j-1):min((t-k)*j,len(yet_unelected))])

        # cc_W_change contains CC(W U c_new \ c) - CC(W)
        cc_W_change = { }
        for query in Q:

            # we compute new scores --> improvements for all possible new candidates for this query
            V_query = random.sample(V,ell) # query ell voters.
            cc_W = cc_score(W, V_query)
            for c_new in [x for x in query if x not in W]:
                cc_W_change[(c_new)] = cc_score(W + [c_new], V_query) - cc_W

        # we pick the candidate that corresponds to the largest improvement on the query // estimated improvement
        c_new = max(cc_W_change, key = cc_W_change.get)
        W.append(c_new)
        scores.append(cc_score(W,V_original))
        if DEBUG:
            print(f"iteration: {i+1}, score: {scores[-1]}")
    return (W, scores)

##########################################################################

######################### LOCAL SEARCH ###################################
def e(k):
    """
    e(k) is a recursive sequence in k, that approximates the Euler constant from above
    """
    som = [1/math.factorial(i) for i in range(k)]
    return sum(som) + 1/((k-1)*math.factorial(k-1))

def alpha_k(j,k):
    """
    for j <= k alpha(j,k) gives the weight of voters represented by j of the k candidates in the committee.
    """
    alphas = [0,1-1/e(k)]
    for i in range(2,j+1):
        alphas.append(i*alphas[-1] - (i-1)*alphas[-2] - 1/e(k))
    return alphas[j]

def alpha(j):
    alphas = [0,1-1/math.e]
    for i in range(2,j+1):
        alphas.append(i*alphas[-1]-(i-1)*alphas[-2]-1/math.e)
    return alphas[j]

alphas = [alpha(i) for i in range(20)]

def local_search_exact(V, k, p=0):
    """
    Regular run of the LS max cover algorithm.

    Inputs:
    - voter set V, defined as subsets of candidates (approval sets of each voter)
    - committee size k
    - probability that a query is answered incorrectly

    Outputs:
    - the committee
    - list containing the score at each iteration
    - number of iterations done
    """
    # initialization
    n, m = len(V), len(V[0])
    C = list(range(m))
    W = random.sample(C,k)
    V_original = V
    if p > 0:
        V = flip_ballots(V.copy(), p)
    gamma = 0.08
    beta = (math.log(2)*gamma)/(k*math.log(k)*(1-1/e(k)-gamma))
    improvement = n*beta
    scores = [ ]
    i = 0
    if DEBUG:
        print(f"initial committee: {W}, score: {cc_score(W,V_original)} stopping cond: {n*beta}")

    # we continue iff the previous change was at least beta large.
    while improvement >= n*beta:
        i = i + 1
        # compute f(W)
        alpha_W = [ ]
        for voter in V:
            alpha_W.append(alphas[len([c for c in W if voter[c] == 1])])
        f_W = sum(alpha_W)
        # in f_W_change we keep track of all f(W U c' \ c)-f(W) scores of this iteration
        f_W_change = { }
        # we compute f-scores --> improvements for all possible swaps
        for c_new in [x for x in C if x not in W]:
            for c in W:
                W_new = [x for x in W if x != c] + [c_new]
                alpha_W_new = [ ]
                for voter in V:
                    alpha_W_new.append(alphas[len([x for x in W_new if voter[x] == 1])])
                f_W_change[(c_new,c)] = sum(alpha_W_new) - f_W
        # we pick the pair that corresponds to the largest improvement
        (c_new,c), improvement = max(f_W_change.items(), key=lambda item: item[1])
        # and implement the change if it is positive
        if improvement > 0:
            W.remove(c)
            W.append(c_new)
        scores.append(cc_score(W,V_original))
        if DEBUG:
            print(f"iteration: {i} improvement (f): {improvement} score: {scores[-1]} W={W} swap {c} for {c_new}")
    return (W, scores, i)

def local_search_queries(V,k,t,ell,iters, p=0):
    """
    Incomplete version of the LS max cover algorithm. Will not access the entire ballot V at
    will, but will instead randomly selected entires of the ballot throughout.

    Inputs:
       - voter set V, defined as subsets of candidates (approval sets of each voter)
       - committee size k
       - query size t
       - amount of voters ell per query
       - number of iterations
       - probability that a query is answered incorrectly

    Outputs:
    - the committee
    - list containing the score at each iteration
    """
    # initialization
    scores = []
    n, m = len(V), len(V[0])
    C = list(range(m))
    W = random.sample(C,k)
    V_original = V
    if p > 0:
        V = flip_ballots(V.copy(), p)

    if DEBUG:
        print("initial committee and score: ", W, cc_score(W,V))

    # per iteration we make (m-k)/(t-k) queries of size t, always containing W, and furthermore t-k other candidates so that all are in one query
    for i in range(iters):
        Q = [ ]
        yet_unelected = [x for x in C if x not in W]
        random.shuffle(yet_unelected)
        for j in range(1, math.ceil((m-k)/(t-k))+1):
            Q.append(W + yet_unelected[(t-k)*(j-1):min((t-k)*j,len(yet_unelected))])

        # in f_W_change we keep track of áll f(W U c' \ c)-f(W) scores of this iteration
        f_W_change = { }
        for query in Q:
            V_query = random.sample(V,ell)

            # compute f(W) for this query
            alpha_W = [ ]
            for voter in V_query:
                alpha_W.append(alphas[len([c for c in W if voter[c] == 1])])
            f_W = sum(alpha_W)
            # we compute f-scores --> improvements for all possible swaps in this query
            for c_new in [x for x in query if x not in W]:
                for c in W:
                    W_new = [x for x in W if x != c] + [c_new]
                    alpha_W_new = [ ]
                    for voter in V_query:
                        alpha_W_new.append(alphas[len([x for x in W_new if voter[x] == 1])])
                    f_W_change[(c_new,c)] = sum(alpha_W_new)-f_W
        # we pick the pair that corresponds to the largest improvement on the query // estimated improvement
        (c_new,c) = max(f_W_change,key = f_W_change.get)
        improvement = f_W_change[(c_new,c)]*(n/ell)
        # and implement the change if it is positive
        if improvement > 0:
            W.remove(c)
            W.append(c_new)
        scores.append(cc_score(W,V_original))
        if DEBUG:
            print(f"iteration: {i} estimated improvement (f): {improvement} score: {scores[-1]}")
    return (W, scores)

###########################################################################

############################### EXPERIMENTS ###############################
# These wrapper functions allow us to cache outcomes of experiments without the entire ballot
# as part of the function call (which blows up the size of the cache too much to be practical).
@filecache(60*60*24*365)
def greedy_queries_wrapper(name, k, t, ell, reps):
    V = read_ballot(name)
    return np.array([np.array(greedy_queries(V, k, t, ell)[1]) for r in range(reps)]) / len(V)

@filecache(60*60*24*365)
def greedy_queries_inaccurate_wrapper(name, k, t, ell, reps, p):
    V = read_ballot(name)
    return np.array([np.array(greedy_queries(V, k, t, ell, p)[1]) for r in range(reps)]) / len(V)

@filecache(60*60*24*365)
def greedy_inaccurate_wrapper(name, k, reps, p):
    V = read_ballot(name)
    return np.array([np.array(greedy_exact(V, k, p)[1]) for r in range(reps)]) / len(V)

@filecache(60*60*24*365)
def greedy_exact_wrapper(name, k):
    V = read_ballot(name)
    return np.array(greedy_exact(V, k)[1]) / len(V)

@filecache(60*60*24*365)
def LS_queries_wrapper(name, k, t, ell, reps):
    V = read_ballot(name)
    return np.array([np.array(local_search_queries(V, k, t, ell, k)[1]) for r in range(reps)]) / len(V)

@filecache(60*60*24*365)
def LS_queries_inaccurate_wrapper(name, k, t, ell, reps, p):
    V = read_ballot(name)
    return np.array([np.array(local_search_queries(V, k, t, ell, k, p)[1]) for r in range(reps)]) / len(V)

@filecache(60*60*24*365)
def LS_inaccurate_wrapper(name, k, reps, p):
    V = read_ballot(name)
    inputs = [(V, k, p)] * reps
    with multiprocessing.Pool(processes = 8) as pool:
        data = pool.starmap(local_search_exact, inputs)
    data = [d[1] for d in data]
    return np.ma.masked_invalid(np.array([row + [np.nan] * (max(map(len, data)) - len(row)) for row in data])) / len(V)

@filecache(60*60*24*365)
def LS_exact_wrapper(name, k, reps):
    V = read_ballot(name)
    inputs = [(V,k)] * reps
    with multiprocessing.Pool(processes = 8) as pool:
        data = pool.starmap(local_search_exact, inputs)
    data = [d[1] for d in data]
    return np.ma.masked_invalid(np.array([row + [np.nan] * (max(map(len, data)) - len(row)) for row in data])) / len(V)

def flip_ballots(V, p):
    """
    Given a ballot V, defined as a list of bitarrays, flip each bit with probability p.
    """
    V_new = [bitarray(l) for l in V]
    for l in V_new:
        flip_mask = bitarray(np.random.binomial(1, p, len(l)).astype(bool).tolist())
        l ^= flip_mask # XOR
    return V_new

def get_ell(n, m, k, t, iters, M):
    """
    Computes the value of ell (number of voters sampled per query) given a
    desired expected number of queries made to each voter.
    """
    return math.floor((n * M * (t - k)) / (iters * (m - k)))

d_app = lambda v, u : sum([abs(sorted(v)[i] - sorted(u)[i]) for i in range(len(v))])
gen_av = lambda p, phi, m : [1-phi + phi * p] * int(np.floor(p * m)) + [phi * p] * int(np.ceil((1 - p) * m))

def compute_p_and_phi(V, s=0.01):
    """
    Given a completed ballot, computes p and phi. p is defined as fraction of votes that are
    approvals. phi is computed by computing the approval vector for the ballot (number of approvals
    per candidate), and then also computing the expected approval vector for our p and various
    values of phi. We then choose phi yielding the approval vector closest to ours (in terms of the
    approvalwise distance).
    """
    # compute p and compute the filled in approval vector
    n, m = len(V), len(V[0])
    p = round(sum([sum(v) for v in V]) / (n * m), 4)
    av = [sum([V[v][c] for v in range(n)]) / n for c in range(m)]

    # use approvalwise vector to find phi closest to it.
    ds = [(d_app(gen_av(p, phi, m), av), phi) for phi in np.arange(0, 1+s, s)]
    return p, sorted(ds)[0][1]

def resample_ballot(n, m, p, phi):
    """
    Samples an approval voting ballot with n voters and m candidates according
    to (p,phi)-resampling model from arXiv:2207.01140v1
    """
    V = []
    C = list(range(m))
    u = random.sample(C, math.floor(p*m))
    for voter in range(n):
        resample = np.random.binomial(1, phi, m)
        V.append(bitarray([1 if (c in u and resample[c] == 0) else 0 if (c not in u and resample[c] == 0) else np.random.binomial(1, p, 1)[0] for c in C]))
    return V

def process_ballot(name, ceiling=1/2):
    """
    Reads in a file containing a partially filled Polis ballot. Removes
    voters that didn't approve any statements, statements that didn't receive
    any approvals nor disapprovals, and statements that were approved by more
    than a 'ceiling' fraction of voters. Then fills in any empty parts of the
    ballot with disapprovals. Outputs the ballot as a list of bitarrays, where
    each bitarray represents one voters ballot, where 1 means approve and 0
    disapprove.

    The second output is a list of statistics containing:
    - n (original number of voters)
    - m (original number of statements)
    - 4-tuple containing original number of approvals, disapprovals, skips and
        empty votes
    - number of voters removed
    - number of candidates removed due to having no approvals
    - number of candidates removed due to having too many approvals
    - 4-tuple containing the same statistics, but after removals.
    """
    with open(f"./openData/{name}/participants-votes.csv", mode='r') as f:
        votes = csv.reader(f, delimiter = ",")
        index = next(votes).index("n-disagree")
        V = np.array([[2 if x == '' else int(x) for x in row[index+1:]] for row in votes], dtype=np.int8)

        n, m = len(V), len(V[0])
        stats = [n, m]
        stats.append((np.sum(V==1), np.sum(V==-1), np.sum(V==0),np.sum(V==2)))

        # Remove voters that didn't approve any statements
        V = V[np.sum(V==1, axis=1) > 0]
        stats.append(n - len(V))
        # Remove candidates that no voter approved or disapproved
        V = V[:, np.sum(V==0, axis=0) < n]
        stats.append(m-len(V[0]))
        m = len(V[0])
        # Remove statements that are too popular
        V = V[:, np.sum(V==1, axis=0) < n*ceiling]
        stats.append(m - len(V[0]))
        stats.append((np.sum(V==1), np.sum(V==-1), np.sum(V==2), np.sum(V==0)))
        # Change neutral and empty votes to disapprove
        V[V == 2] = -1
        V[V == -1] = 0
        return [bitarray(list(v)) for v in V], stats

def read_ballot(name):
    with open(f"./data/{name}", mode='rb') as f:
        return pickle.load(f)

def plot_histogram(V, title="", save=False):
    n, m = len(V), len(V[0])
    metric = [sum([V[v][c] for v in range(n)]) for c in range(m)]
    metric.sort()
    plt.bar(range(m), metric)
    plt.title(f"Relative approval over candidates, $n={n}$, $m={m}$\n{ {title}}")
    if save:
        plt.savefig("./plots/distributions/" + title + ".pdf")
    else:
        plt.show()
    plt.close()

def sort_ballot(V):
    """
    Given a complete ballot as a list of bitarrays, sort the columns (candidates) from least to
    most approved.
    """
    a = np.array([list(v) for v in V])
    a = a[:,np.argsort(np.sum(a,axis=0))[::-1]]
    return [bitarray(list(v)) for v in a]

def main():
    data = list(os.walk("./data"))[0][2]
    data_polis = [b for b in data if not "random" in b]
    data_random = [b for b in data if "random" in b]

    # Compute p and phi
    # vals = np.array([compute_p_and_phi(read_ballot(b), s=0.01) for b in data if not "random" in b])
    # print(f"Average p = {vals[:,0].mean()} all p values: {vals[:,0]}")
    # print(f"Average phi = {vals[:,1].mean()} all phi values: {vals[:,1]}")
    # plt.scatter(vals[:,1], vals[:,0])
    # plt.xlabel("$\phi$")
    # plt.ylabel("$p$")
    # plt.xlim(1,0)
    # plt.ylim(0,1)
    # plt.show()
    # exit()

    # Create artificial data
    # p = 0.0891
    # phi = 0.693
    # n = 1000
    # m = int(n * 0.4)
    # for r in range(100):
    #     V = resample_ballot(n, m, p, phi)
    #     with open(f"data/random_{r}", "wb") as f:
    #         pickle.dump(V, f)

    # Settings for the experiments
    K = 8 # commitee size
    T = 20 # number of statements per query
    M_s = [1,2,3,4,5] # expected number of queries (of T statements) per voter
    p = 1/10 # probability that a query is answered wrong
    REPS = 50 # number of repetitions of random algorithms
    REPS_LS = 20 # number of repetitions of exact LS alg (since this is much slower that other algs) and also is only random in initial selection so less impactful.

    # Experiment 1: compute CC scores of exact greedy, exact LS and approval voting on all data sets
    g_p = np.array([greedy_exact_wrapper(b, K) for b in data_polis])
    g_r = np.array([greedy_exact_wrapper(b, K) for b in data_random])
    ls_p = np.vstack([LS_exact_wrapper(b, K, REPS_LS)[:,:K] for b in data_polis])
    ls_r = np.vstack([LS_exact_wrapper(b, K, REPS_LS)[:,:K] for b in data_random])
    av_p = np.array([[cc_score(range(k), sorted(read_ballot(b), key=lambda x : sum(x)))/len(read_ballot(b)) for k in range(1,K+1)] for b in data_polis])
    av_r = np.array([[cc_score(range(k), sorted(read_ballot(b), key=lambda x : sum(x)))/len(read_ballot(b)) for k in range(1,K+1)] for b in data_random])

    # Fill in nan's in local search results
    replace_nan = lambda x : np.where(np.isnan(x), np.maximum.accumulate(np.where(np.isnan(x), 0, x), axis=1), x)
    ls_r = replace_nan(ls_r)
    ls_p = replace_nan(ls_p)

    # Print mean and stddev of CC score
    print("Mean and std dev of CC scores")
    print(f"\tLocal search polis: {np.mean(ls_p,axis=0),np.std(ls_p,axis=0)}")
    print(f"\tGreedy polis: {np.mean(g_p,axis=0),np.std(g_p,axis=0)}")
    print(f"\tApproval voting search polis: {np.mean(av_p,axis=0),np.std(av_p,axis=0)}")
    print(f"\tLocal search random: {np.mean(ls_r,axis=0),np.std(ls_r,axis=0)}")
    print(f"\tGreedy random: {np.mean(g_r,axis=0),np.std(g_r,axis=0)}")
    print(f"\tApproval voting random: {np.mean(av_r,axis=0),np.std(av_r,axis=0)}")

    # Check if approval voting was ever better than greedy or local search
    print("Does approval voting ever beat the CC score of greedy or local search? If so, what is the ratio between greedy/local search divided by approval voting? LS on polis, greedy on polis, LS on random, greedy on random")
    n_sets_p = len(data_polis)
    n_sets_r = len(data_random)
    av_scores_p = av_p.reshape(n_sets_p, 1, K)
    av_scores_r = av_r.reshape(n_sets_r, 1, K)
    vals = [ls_p.reshape(n_sets_p, REPS_LS, K)/av_scores_p, g_p.reshape(n_sets_p, 1, K)/av_scores_p, ls_r.reshape(n_sets_r, REPS_LS, K)/av_scores_r, g_r.reshape(n_sets_r, 1, K)/av_scores_r]
    for x in vals:
        print(f"\t{np.any(x < 1)} {np.round(x[np.where(x < 1)],3)} Mean/std improvement : {np.round(np.mean(x, axis=(0,1)), 3),np.round(np.std(x,axis=(0,1)), 3)}")

    # Experiment 2: compute CC scores of inexact and/or inaccurate algorithms
    # on the same data. Plot scores relative to exact algorithms.
    plt.figure(figsize=[7.05, 2.17])
    plt.subplots_adjust(top=0.98, left=0.062, right=0.997, bottom=0.09)
    plt.rcParams.update({
        "text.usetex":True,
        "font.family":"Libertine",
        "font.size":9,
        "legend.fontsize":7,
        "errorbar.capsize":2,
        "lines.linewidth":1,
    })
    plt.rcParams['hatch.linewidth'] = 0.6

    plt.ylabel("Relative Max Cover score")
    plt.ylim(0.625, 1)
    plt.xlim(0.4,6.6)

    xs = np.array(range(1, len(M_s) + 2))
    labels = ["$M="+str(M)+"$" for M in M_s] + ["complete info"]
    plt.xticks(list(xs), labels=labels)
    w = 0.111

    h1 = "/"
    h2 = "\\"
    h3 = "x"
    f = 4

    # Greedy polis
    d1 = np.zeros((len(M_s),REPS*len(data_polis))) # incomplete
    d2 = np.zeros((len(M_s),REPS*len(data_polis))) # incomplete and inaccurate
    d3 = np.zeros((REPS*len(data_polis))) # inaccurate
    for i, M in enumerate(M_s):
        for j, b in enumerate(data_polis):
            V = read_ballot(b)
            n, m = len(V), len(V[0])
            score = greedy_exact_wrapper(b, K)[K-1]
            ell = get_ell(n, m, K, T, K, M)
            d1[i][j*REPS:(j+1)*REPS] = greedy_queries_wrapper(b, K, T, ell, REPS)[:,K-1] / score
            d2[i][j*REPS:(j+1)*REPS] = np.array(list(greedy_queries_inaccurate_wrapper(b, K, T, ell, REPS, p)[:,K-1])) / score
            if i == 0:
                d3[j*REPS:(j+1)*REPS] = greedy_inaccurate_wrapper(b, K, REPS, p)[:,K-1] / score
    plt.bar(xs-w/2-3*w, np.append(d1.mean(axis=1), 1), yerr=np.append(d1.std(axis=1)/2, 0), width=w, color='C0', label="\\textsc{greedy-incomplete}, Polis, $p=0$")
    plt.bar(xs-w/2-2*w, np.append(d2.mean(axis=1), d3.mean()), yerr=np.append(d2.std(axis=1)/2, d3.std()/2), width=w, color='C0', hatch=h1*f, label="\\textsc{greedy-incomplete}, Polis, $p=" + str(p) + "$")

    # Greedy random
    d1 = np.zeros((len(M_s),REPS*len(data_random))) # incomplete
    d2 = np.zeros((len(M_s),REPS*len(data_random))) # incomplete and inaccurate
    d3 = np.zeros((REPS*len(data_random))) # inaccurate
    for i, M in enumerate(M_s):
        for j, b in enumerate(data_random):
            V = read_ballot(b)
            n, m = len(V), len(V[0])
            score = greedy_exact_wrapper(b, K)[K-1]
            ell = get_ell(n, m, K, T, K, M)
            d1[i][j*REPS:(j+1)*REPS] = greedy_queries_wrapper(b, K, T, ell, REPS)[:,K-1] / score
            d2[i][j*REPS:(j+1)*REPS] = np.array(list(greedy_queries_inaccurate_wrapper(b, K, T, ell, REPS, p)[:,K-1])) / score
            if i == 0:
                d3[j*REPS:(j+1)*REPS] = greedy_inaccurate_wrapper(b, K, REPS, p)[:,K-1] / score

    plt.bar(xs-w/2-w, np.append(d1.mean(axis=1), 1), yerr=np.append(d1.std(axis=1)/2, 0), width=w, color='C0', hatch=h2*f, label="\\textsc{greedy-incomplete}, synthetic, $p=0$")
    plt.bar(xs-w/2, np.append(d2.mean(axis=1), d3.mean()), yerr=np.append(d2.std(axis=1)/2, d3.std()/2), width=w, color='C0', hatch=h3*f, label="\\textsc{greedy-incomplete}, synthetic, $p=" + str(p) + "$")

    # Local search polis
    d3 = np.zeros((REPS_LS*len(data_polis))) # inaccurate
    for i, M in enumerate(M_s):
        for j, b in enumerate(data_polis):
            n, m = len(V), len(V[0])
            score = np.nanmean(LS_exact_wrapper(b, K, REPS_LS), axis=0)[K-1]
            ell = get_ell(n, m, K, T, K, M)
            d1[i][j*REPS:(j+1)*REPS] = LS_queries_wrapper(b, K, T, ell, REPS)[:,K-1] / score
            d2[i][j*REPS:(j+1)*REPS] = np.array(list(LS_queries_inaccurate_wrapper(b, K, T, ell, REPS, p)[:,K-1])) / score
            if i == 0:
                d3[j*REPS_LS:(j+1)*REPS_LS] = LS_inaccurate_wrapper(b, K, REPS_LS, p)[:,K-1] / score
    plt.bar(xs+w/2, np.append(d1.mean(axis=1), 1), yerr=np.append(d1.std(axis=1)/2, 0), width=w, color='C1', label="\\textsc{ls-complete}, Polis, $p=0$")
    plt.bar(xs+w/2+w, np.append(d2.mean(axis=1), np.nanmean(d3)), yerr=np.append(d2.std(axis=1)/2, np.nanstd(d3)/2), width=w, color='C1', hatch=h1*f, label="\\textsc{ls-complete}, Polis, $p=" + str(p) + "$")

    # Local search random
    d3 = np.zeros((REPS_LS*len(data_random))) # inaccurate
    for i, M in enumerate(M_s):
        for j, b in enumerate(data_random):
            n, m = len(V), len(V[0])
            score = np.nanmean(LS_exact_wrapper(b, K, REPS_LS), axis=0)[K-1]
            ell = get_ell(n, m, K, T, K, M)
            d1[i][j*REPS:(j+1)*REPS] = LS_queries_wrapper(b, K, T, ell, REPS)[:,K-1] / score
            d2[i][j*REPS:(j+1)*REPS] = np.array(list(LS_queries_inaccurate_wrapper(b, K, T, ell, REPS, p)[:,K-1])) / score
            if i == 0:
                d3[j*REPS_LS:(j+1)*REPS_LS] = LS_inaccurate_wrapper(b, K, REPS_LS, p)[:,K-1] / score
    plt.bar(xs+w/2+2*w, np.append(d1.mean(axis=1), 1), yerr=np.append(d1.std(axis=1)/2, 0), width=w, color='C1', hatch=h2*f, label="\\textsc{ls-complete}, synthetic, $p=0$")
    plt.bar(xs+w/2+3*w, np.append(d2.mean(axis=1), np.nanmean(d3)), yerr=np.append(d2.std(axis=1)/2, np.nanstd(d3)/2), width=w, color='C1', hatch=h3*f, label="\\textsc{ls-complete}, synthetic, $p=" + str(p) + "$")

    plt.legend(loc="lower right",ncol=2)
    plt.show()
    plt.savefig("./plots/plot.pdf")
    plt.clf()

if __name__ == '__main__':
    main()
