# FOLLOWING CODE ADAPTED FROM:
# http://www.inference.org.uk/mackay/python/examples/decompose.shtml

answers = []

def decompose( n , candidatelist , answerSoFar=[]):
    """ Decomposes an integer n in as many ways as possible
    using the offered list of candidates, 
    each of which may be used many times.
    It's recommended that the candidatelist be 
    ordered from big to small. WARNING - if the list does 
    not include '1' at the end, bad things could happen!
    """
    assert(n>=0)
    if (n==0):
        # recursion has finished, print the list
        answers.append(list(answerSoFar))
        return
    offset = 0 ; 
    for a in candidatelist:
        if ( a <= n):
            answerSoFar.append(a) 
            decompose( n-a , candidatelist[offset:] , answerSoFar)
            answerSoFar.pop()
            pass
        pass
        offset += 1 ; 
    pass


def decomposeInt(n):
    """
    Decomposes an integer n into sum of integers in as many ways as possible.
    """
    global answers
    answers = []
    assert(n>=1)
    a=[]; i=1;
    # create the list of squares up to n
    while ( i <= n ):
        a.append(i)
        i += 1
        pass
    #a.reverse()
    decompose(n,a)
    return answers
    