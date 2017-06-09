# cython: profile=False

import collections

from libc cimport math
import math

import numpy as np
cimport numpy as np

import kenlm


cdef class CharLMBeamDecoder(object):
    """
    A prefix beam decoder with char level LM.
    """

    cpdef double logpsum(self,double a,double b):
        cdef double psum = math.exp(a) + math.exp(b)
        if psum == 0.0:
            return float('-inf')
        else:
            return math.log(psum)

    cdef dict char_to_int
    cdef dict int_to_char
    cdef int  blank_indx
    cdef int  space_indx
    cdef object alphabet
    cdef object lm
    def __init__(self, model_file, alphabet, blank_index=0, space_index=28):
        self.alphabet = alphabet
        self.char_to_int = {}
        self.int_to_char = {}
        for i, c in enumerate(alphabet):
            self.char_to_int[c] = int(i)
            self.int_to_char[int(i)] = c
        print("loading char level KenLM model:{} in CharLMBeamDecoder({})".format(model_file, alphabet))
        self.lm = kenlm.Model(model_file)

    def lm_prefix_state(self, prefix):
        lm_state = kenlm.State()
        self.lm.BeginSentenceWrite(lm_state)
        for c in prefix:
            out_state = kenlm.State()
            self.lm.BaseFullScore(lm_state,c,out_state)
            lm_state = out_state
        return lm_state

    def lm_score_char(self, object lm_state, cha):
        out_state = kenlm.State()
        log10prob = self.lm.BaseFullScore(lm_state,cha,out_state).log_prob
        return log10prob

    def decode(self, double[::1,:] probs,
        double alpha=0.7, double beta=1.7, unsigned int beam=40):
        """
        Do prefix beam search with n-gram KenLM
        """
        cdef double[::1,:] lprobs = np.log10(probs)                                             # use float?
        cdef unsigned int N = probs.shape[0]
        cdef unsigned int T = probs.shape[1]
        cdef unsigned int t, i
        cdef float p_nb, p_b, pp_nb, pp_b, p_blank, p_tot

        scoreFn = lambda x: self.logpsum(x[1][0],x[1][1]) + beta * len(x[0])       # shoud this be this a log(len(x[0]))
        initFn = lambda : [float('-inf'),float('-inf'),0]

        # [prefix, [p_nb, p_b, node, |W|]]
        Aprev = [['',[float('-inf'),0.0,0]]]
        Aold  = collections.defaultdict(initFn)

        # loop over time
        for t in range(T):
            Aprev = dict(Aprev)
            Anext = collections.defaultdict(initFn)

            candidate_chars = self.alphabet #xrange(1,N) # [i for i in range(1,lprobs.shape[0]) if lprobs[i,t] > MIN_PROB]
            for prefix,(p_nb,p_b,score) in Aprev.items():
                p_blank = lprobs[0,t]
                p_tot = self.logpsum(p_nb,p_b)

                valsP = Anext[prefix]
                valsP[1] = self.logpsum(valsP[1], p_blank + p_tot)                                    # blank
                if len(prefix) > 0:
                    valsP[0] = self.logpsum(valsP[0], p_nb + lprobs[self.char_to_int[prefix[-1]],t])  # collapse

                lm_state = self.lm_prefix_state(prefix)

                it = enumerate(candidate_chars)
                next(it) # blank already handled
                for i,c in it:
                    nprefix = prefix + c
                    valsN = Anext[nprefix]

                    lm_prob = alpha * self.lm_score_char(lm_state, c)
                    if len(prefix)==0 or (len(prefix) > 0 and i != prefix[-1]):
                        valsN[0] = self.logpsum(valsN[0], lprobs[i,t] + lm_prob + p_tot)        # add new char
                    else:
                        valsN[0] = self.logpsum(valsN[0], lprobs[i,t] + lm_prob + p_b)          # repeats need a blank between

                    # if it was not updated above - same as above
                    if nprefix not in Aprev:
                        pp_nb, pp_b,_ = Aold[nprefix]
                        valsN[1] = self.logpsum(valsN[1], p_blank + self.logpsum(pp_nb,pp_b))   # blank handlig
                        valsN[0] = self.logpsum(valsN[0], lprobs[i,t] + pp_nb)                  # collapse

            Aold = Anext

            Anext = [[el[0],[el[1][0],el[1][1],scoreFn(el)]] for el in Anext.items()]
            Aprev = sorted(Anext, key=lambda x: x[1][2], reverse=True)[:beam]


        hyp = Aprev[0][0].replace('#',' ')
        return hyp, scoreFn(Aprev[0])
